import torch
from torch import nn


class SessionLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.W_h = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.b = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x_t, h_prev, c_prev):
        gates = x_t @ self.W_x + h_prev @ self.W_h + self.b
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t
