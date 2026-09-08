import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def evaluate_multilabel(model, loader, criterion, device, forward_fn):
   
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    import torch
    with torch.no_grad():
        for batch in loader:
            logits, labels = forward_fn(model, batch, device)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    return total_loss / len(loader), macro_f1, micro_f1


def evaluate_singlelabel(model, loader, criterion, device, forward_fn):
    
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    import torch
    with torch.no_grad():
        for batch in loader:
            logits, labels = forward_fn(model, batch, device)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, macro_f1
