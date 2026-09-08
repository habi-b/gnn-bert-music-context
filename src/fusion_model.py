import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNBERTFusion(nn.Module):
    
    def __init__(self, num_tags=30, gnn_hidden=64, bert_hidden=768,
                 pretrained="bert-base-uncased"):
        super().__init__()
        self.conv1 = SAGEConv(12, gnn_hidden)
        self.conv2 = SAGEConv(gnn_hidden, gnn_hidden)
        self.bert = BertModel.from_pretrained(pretrained)
        self.query_proj = nn.Linear(gnn_hidden, bert_hidden)
        self.key_proj = nn.Linear(bert_hidden, bert_hidden)
        self.value_proj = nn.Linear(bert_hidden, bert_hidden)
        self.classifier = nn.Linear(gnn_hidden + bert_hidden, num_tags)

    def encode_graph(self, graph_batch):
        x = F.relu(self.conv1(graph_batch.x, graph_batch.edge_index))
        x = F.relu(self.conv2(x, graph_batch.edge_index))
        return global_mean_pool(x, graph_batch.batch)

    def forward(self, graph_batch, input_ids, attention_mask, return_attention=False):
        g = self.encode_graph(graph_batch)

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H_text = bert_out.last_hidden_state

        Q = self.query_proj(g).unsqueeze(1)
        K = self.key_proj(H_text)
        V = self.value_proj(H_text)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        # Mask padding tokens so they get ~0 attention weight
        pad_mask = (attention_mask.unsqueeze(1) == 0)
        attn_scores = attn_scores.masked_fill(pad_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended_text = torch.matmul(attn_weights, V).squeeze(1)

        z = torch.cat([g, attended_text], dim=-1)
        logits = self.classifier(z)

        if return_attention:
            return logits, attn_weights.squeeze(1)
        return logits


class BertOnlyAblation(nn.Module):
    

    def __init__(self, num_tags=30, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.classifier = nn.Linear(768, num_tags)

    def forward(self, graph_batch, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


class GNNOnlyAblation(nn.Module):
    
    def __init__(self, num_tags=30, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(12, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_tags)

    def forward(self, graph_batch, input_ids, attention_mask):
        x = F.relu(self.conv1(graph_batch.x, graph_batch.edge_index))
        x = F.relu(self.conv2(x, graph_batch.edge_index))
        g = global_mean_pool(x, graph_batch.batch)
        return self.classifier(g)


class EarlyConcatAblation(nn.Module):
    

    def __init__(self, num_tags=30, gnn_hidden=64, bert_hidden=768,
                 pretrained="bert-base-uncased"):
        super().__init__()
        self.conv1 = SAGEConv(12, gnn_hidden)
        self.conv2 = SAGEConv(gnn_hidden, gnn_hidden)
        self.bert = BertModel.from_pretrained(pretrained)
        self.classifier = nn.Linear(gnn_hidden + bert_hidden, num_tags)

    def forward(self, graph_batch, input_ids, attention_mask):
        x = F.relu(self.conv1(graph_batch.x, graph_batch.edge_index))
        x = F.relu(self.conv2(x, graph_batch.edge_index))
        g = global_mean_pool(x, graph_batch.batch)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t = out.last_hidden_state[:, 0, :]
        z = torch.cat([g, t], dim=-1)
        return self.classifier(z)
