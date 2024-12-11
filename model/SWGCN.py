import os
import torch
import torch.nn as nn
from utils.data_utils import convert_sp_mat_to_sp_tensor
from torch_geometric.nn import LGConv

from utils.layer import Attention

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SWGCN(nn.Module):
    def __init__(self, args, train_mats, n_user, n_item):
        super(SWGCN, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_size = args.embedding_size
        self.msg_dropout = args.msg_dropout
        self.n_layer = args.n_layer
        self.n_behavior = args.n_behavior
        self.is_align_target = args.is_align_target
        self.device = args.device
        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2
        self.lamda = args.lamda
        self.self_loop_weight = args.self_loop_weight

        self.train_mats = [convert_sp_mat_to_sp_tensor(train_mat) for train_mat in train_mats]

        self.embedding_dict = self._init_weight()

        self.alpha_list = [nn.Linear(self.embedding_size, 1).to(self.device) for _ in range(self.n_behavior)]
        self.activate = nn.ELU()

        self.graph_encoder = LGConv(args.embedding_size)

        self.attention = Attention(args.embedding_size, num_heads=4)
        self.norm = nn.LayerNorm(args.embedding_size)

        self.dropout = nn.Dropout(p=self.msg_dropout)

    def _init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user': nn.Parameter(initializer(torch.empty(self.n_user, self.n_behavior, self.embedding_size))),
            'item': nn.Parameter(initializer(torch.empty(self.n_item, self.n_behavior, self.embedding_size))),
        })

        return embedding_dict

    def predict(self, user_embeddings, item_embeddings):
        y = torch.sum(torch.mul(user_embeddings, item_embeddings), dim=1)

        return y

    def calculate_interaction_weights(self, edge_index, behavior):
        user_index, item_index = edge_index[0].view(-1), edge_index[1].view(-1)
        user_embeddings = self.embedding_dict['user'][user_index, behavior, :]
        item_embeddings = self.embedding_dict['item'][item_index, behavior, :]

        # The detach operation must be added to prevent the gradient back propagation error
        diff = torch.abs(user_embeddings-item_embeddings).detach()

        activate_values = self.activate(self.alpha_list[behavior](diff).view(-1))
        sp_tensor = torch.sparse_coo_tensor(edge_index, activate_values, (self.n_user, self.n_item), dtype=torch.float, device=self.device, requires_grad=True)
        sp_tensor = torch.sparse.softmax(sp_tensor, dim=1)

        target_preference_weights = sp_tensor.coalesce().values()

        auxiliary_preference_weights = torch.norm(diff, dim=1).pow(2)

        return target_preference_weights, auxiliary_preference_weights

    def create_alignment_loss(self, target_preference_weights_list, auxiliary_preference_weights_list):
        fun_loss_list = []
        l2_loss_list = []
        alignment_behavior = self.n_behavior if self.is_align_target else self.n_behavior-1
        for behavior in range(alignment_behavior):
            target_preference_weight, auxiliary_preference_weight = target_preference_weights_list[behavior], auxiliary_preference_weights_list[behavior]
            fun_loss_list.append(torch.mul(target_preference_weight, auxiliary_preference_weight).mean())

            l2_loss_list.append(target_preference_weight.norm(2).pow(2) / len(target_preference_weight))

        fun_loss = torch.mean(torch.tensor(fun_loss_list))
        l2_loss = torch.mean(torch.tensor(l2_loss_list))
        loss = fun_loss + self.gamma1 * l2_loss

        return loss, fun_loss, l2_loss

    def create_bpr_loss(self, user_embeddings, pos_items_embeddings, neg_items_embeddings):
        pos_scores = self.predict(user_embeddings, pos_items_embeddings)
        neg_scores = self.predict(user_embeddings, neg_items_embeddings)

        fun_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        emb_loss = (1 / 2) * (user_embeddings.norm(2).pow(2) + pos_items_embeddings.norm(2).pow(2) + \
                              neg_items_embeddings.norm(2).pow(2)) / float(len(user_embeddings))

        l2_loss = emb_loss

        loss = fun_loss + self.gamma2 * l2_loss

        return loss, fun_loss, l2_loss

    def create_joint_loss(self, user_embeddings, pos_items_embeddings, neg_items_embeddings, target_preference_weights_list, auxiliary_preference_weights_list):
        bpr_loss,_,_ = self.create_bpr_loss(user_embeddings, pos_items_embeddings, neg_items_embeddings)
        alignment_loss,_,_ = self.create_alignment_loss(target_preference_weights_list, auxiliary_preference_weights_list)

        loss = self.lamda * alignment_loss + (1 -self.lamda) * bpr_loss
        return loss


    def forward(self, users, pos_items, neg_items):
        edge_index_list, edge_weight_list,  = [], []
        target_preference_weights_list, auxiliary_preference_weights_list = [], []
        for behavior in range(self.n_behavior):
            edge_index = torch.as_tensor(self.train_mats[behavior]._indices(), dtype=torch.long, device=self.device)  # 注意这里的dtype
            target_preference_weight, auxiliary_preference_weight = self.calculate_interaction_weights(edge_index, behavior)

            if torch.min(edge_index[1]) < self.n_user:
                # Constructing the adjacency matrix
                edge_index[1] += self.n_user

            # Add self-loop
            self_loop = torch.tensor([range(self.n_user + self.n_item), range(self.n_user + self.n_item)]).to(self.device)
            edge_index = torch.cat([edge_index, self_loop], dim=1)

            self_loop_weight = self.self_loop_weight * torch.ones(len(self_loop[0])).to(self.device)
            edge_weight = torch.cat([target_preference_weight, self_loop_weight], dim=0)

            edge_index_list.append(edge_index)
            edge_weight_list.append(edge_weight)
            target_preference_weights_list.append(target_preference_weight)
            auxiliary_preference_weights_list.append(auxiliary_preference_weight)

        all_embeddings = torch.cat([self.embedding_dict['user'], self.embedding_dict['item']], dim=0)

        for l in range(self.n_layer):
            encoded_embeddings = []
            for behavior in range(self.n_behavior):
                edge_index, edge_weight = edge_index_list[behavior], edge_weight_list[behavior]

                encoded_embedding = self.graph_encoder(all_embeddings[:, behavior, :], edge_index, edge_weight)

                encoded_embeddings.append(encoded_embedding)

            all_embeddings = torch.stack(encoded_embeddings, dim=1)

        att_out = self.attention(all_embeddings, all_embeddings, all_embeddings)
        all_embeddings = att_out + all_embeddings
        all_embeddings = self.norm(all_embeddings)

        if self.msg_dropout > 0.:
            all_embeddings = self.dropout(all_embeddings)

        all_g_embeddings = torch.sum(all_embeddings, dim=1).squeeze()

        u_g_embeddings, i_g_embeddings = torch.split(all_g_embeddings, [self.n_user, self.n_item])

        user_embeddings = u_g_embeddings[users.long(), :]

        pos_item_embeddings = i_g_embeddings[pos_items.long(), :]
        neg_item_embeddings = i_g_embeddings[neg_items.long(), :]

        return user_embeddings, pos_item_embeddings, neg_item_embeddings, target_preference_weights_list, auxiliary_preference_weights_list
