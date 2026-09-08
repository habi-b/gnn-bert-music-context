
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class MelSpectrogramDataset(Dataset):
    
    def __init__(self, labels_df, mel_dir, genre_to_idx):
        self.labels_df = labels_df.reset_index(drop=True)
        self.mel_dir = mel_dir
        self.genre_to_idx = genre_to_idx

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        mel = np.load(os.path.join(self.mel_dir, f"{row['track_id']}.npy"))
        mel_tensor = torch.tensor(mel, dtype=torch.float).unsqueeze(0)
        label = self.genre_to_idx[row["genre"]]
        return mel_tensor, label


class FusionDataset(Dataset):
    
    def __init__(self, df, tag_cols, graph_dir, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tag_cols = tag_cols
        self.graph_dir = graph_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = torch.load(os.path.join(self.graph_dir, f"{row['ytid']}.pt"), weights_only=False)
        tokens = self.tokenizer(row["caption"], padding="max_length", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        label = torch.tensor(row[self.tag_cols].values.astype(np.float32))
        return graph, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0), label


def fusion_collate(batch):
    
    graphs, input_ids, attn_masks, labels = zip(*batch)
    graph_batch = Batch.from_data_list(list(graphs))
    return graph_batch, torch.stack(input_ids), torch.stack(attn_masks), torch.stack(labels)


class ContrastiveDataset(Dataset):
    

    def __init__(self, df, graph_dir, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.graph_dir = graph_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = torch.load(os.path.join(self.graph_dir, f"{row['ytid']}.pt"), weights_only=False)
        tokens = self.tokenizer(row["caption"], padding="max_length", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        return graph, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)


def contrastive_collate(batch):
    graphs, input_ids, attn_masks = zip(*batch)
    return Batch.from_data_list(list(graphs)), torch.stack(input_ids), torch.stack(attn_masks)
