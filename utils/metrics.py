import torch


def topk_accuracy(logits, targets, k=5):
    _, topk = logits.topk(k, dim=1)
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()
