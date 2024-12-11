import numpy as np
import torch
from utils.constvars import CONST

INF = 1e8

def metric_all(model_name, model, test_loader, n_item, topks, device, multi):
    # 初始化
    metrics = dict()
    for top_k in topks:
        metrics[top_k] = {'HR':[], 'NDCG':[]}
    for users, ground_truth, train_mask in test_loader:
        users = users.to(device)
        ground_truth = ground_truth.to(device)
        train_mask = train_mask.to(device)
        all_items = torch.arange(n_item).to(device)

        placeholder_items = torch.tensor([0]).to(device)
        if model_name == CONST.BPRMF:
            user_embeddings, item_embeddings, _ = model(users, all_items, placeholder_items)
            scores = torch.mm(user_embeddings, item_embeddings.t())
        elif model_name == CONST.SWGCN:
            user_embeddings, item_embeddings, _, _, _= model(users, all_items, placeholder_items)
            scores = torch.mm(user_embeddings, item_embeddings.t())

        scores -= INF*train_mask
        for top_k in topks:
            values, col_indices = torch.topk(scores, top_k, dim=1)
            row_indices = torch.zeros_like(col_indices) + torch.arange(
                scores.shape[0], device=device, dtype=torch.long).view(-1, 1)  # 会自动broadcast
            is_hit = ground_truth[row_indices.view(-1), col_indices.view(-1)].view(-1, top_k)
            HR = is_hit.sum(dim=1).cpu().numpy()
            NDCG = (is_hit.float() / torch.log2(torch.arange(2, top_k+2, device=device, dtype=torch.float32))).sum(dim=1).cpu().numpy()

            metrics[top_k]['HR'].extend(list(HR))
            metrics[top_k]['NDCG'].extend(list(NDCG))

    for top_k in topks:
        metrics[top_k]['HR'] = np.mean(metrics[top_k]['HR']) * multi
        metrics[top_k]['NDCG'] = np.mean(metrics[top_k]['NDCG']) * multi
    return metrics