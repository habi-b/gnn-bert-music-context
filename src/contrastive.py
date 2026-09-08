import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphEncoder(nn.Module):
    
    def __init__(self, hidden=64, embed_dim=128):
        super().__init__()
        self.conv1 = SAGEConv(12, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.proj = nn.Linear(hidden, embed_dim)

    def forward(self, graph_batch):
        x = F.relu(self.conv1(graph_batch.x, graph_batch.edge_index))
        x = F.relu(self.conv2(x, graph_batch.edge_index))
        g = global_mean_pool(x, graph_batch.batch)
        return F.normalize(self.proj(g), dim=-1)


class TextEncoder(nn.Module):
    
    def __init__(self, embed_dim=128, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(self.proj(cls), dim=-1)


def info_nce_loss(graph_embeds, text_embeds, temperature=0.07):
    
    logits = torch.matmul(graph_embeds, text_embeds.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)
    return (loss_g2t + loss_t2g) / 2


def compute_recall_at_k(query_embeds, gallery_embeds, k_values=(1, 5, 10)):
    
    sim_matrix = torch.matmul(query_embeds, gallery_embeds.T)
    n = sim_matrix.size(0)
    correct_idx = torch.arange(n)

    results = {}
    for k in k_values:
        topk = sim_matrix.topk(k, dim=1).indices
        hits = (topk == correct_idx.unsqueeze(1)).any(dim=1)
        results[f"R@{k}"] = hits.float().mean().item()
    return results
