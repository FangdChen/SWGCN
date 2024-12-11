import torch
import torch.nn as nn

class BPRMF(nn.Module):
    def __init__(self, args, n_user, n_item):
        super(BPRMF, self).__init__()
        """
        n_user: number of users;
        n_item: number of items;
        embedding_size: number of predictive factors.
        """
        self.reg = args.gamma2

        self.embed_user = nn.Embedding(n_user, args.embedding_size)
        self.embed_item = nn.Embedding(n_item, args.embedding_size)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def predict(self, user_embeddings, item_embeddings):
        y = torch.sum(torch.mul(user_embeddings, item_embeddings), dim=1)

        return y

    def create_bpr_loss(self, user_embeddings, pos_items_embeddings, neg_items_embeddings):
        pos_scores = self.predict(user_embeddings, pos_items_embeddings)
        neg_scores = self.predict(user_embeddings, neg_items_embeddings)

        fun_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        emb_loss = (1 / 2) * (user_embeddings.norm(2).pow(2) +
                              pos_items_embeddings.norm(2).pow(2) +
                              neg_items_embeddings.norm(2).pow(2)) / float(len(user_embeddings))

        l2_loss = emb_loss

        loss = fun_loss + self.reg * l2_loss

        return loss, fun_loss, emb_loss

    def forward(self, user, pos_item, neg_item):
        user_embeddings = self.embed_user(user)
        pos_item_embeddings = self.embed_item(pos_item)
        neg_item_embeddings = self.embed_item(neg_item)

        return user_embeddings, pos_item_embeddings, neg_item_embeddings
