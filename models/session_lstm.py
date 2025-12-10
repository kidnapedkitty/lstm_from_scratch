import torch
from torch import nn
from .session_lstm_cell import SessionLSTMCell


class SessionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 num_layers=1, pad_idx=0, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.layers = nn.ModuleList(
            [SessionLSTMCell(embed_dim if l == 0 else hidden_size,
                             hidden_size)
             for l in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, lengths):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        h = [torch.zeros(batch_size, self.hidden_size, device=device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=device)
             for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            valid_mask = (lengths > t).unsqueeze(1)  # only update non-padded positions
            for l, layer in enumerate(self.layers):
                h_new, c_new = layer(x_t, h[l], c[l])
                h[l] = torch.where(valid_mask, h_new, h[l])
                c[l] = torch.where(valid_mask, c_new, c[l])
                x_t = self.dropout(h[l])
            outputs.append(h[-1].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden)
        logits = self.output(outputs)  # (batch, seq_len, vocab_size)
        return logits