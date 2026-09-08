import re
import torch
import torch.nn as nn
from transformers import BertModel


def clean_filename_to_text(mp3_path):
    name = mp3_path.split("/")[-1].replace(".mp3", "")
    name = re.sub(r"-\d+-\d+$", "", name)
    name = re.sub(r"[-_]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


class BertTagClassifier(nn.Module):
    def __init__(self, num_tags, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)
