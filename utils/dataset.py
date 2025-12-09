import torch
from torch.utils.data import Dataset


class SessionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch, pad_idx=0):
    sequences, labels = zip(*batch)
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)

    padded = []
    for s in sequences:
        padded.append(s + [pad_idx] * (max_len - len(s)))

    padded = torch.tensor(padded, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded, lengths, labels
