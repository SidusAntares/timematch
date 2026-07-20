from torch import nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, use_batch_norm=False):
        super().__init__()
        self.output_dim = int(output_dim)
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None

    def forward(self, x, static=None):
        if static is not None:
            x = torch.cat((x, static), dim=1)
        hidden = self.input_fc(x)
        if self.batch_norm is not None:
            hidden = self.batch_norm(hidden)
        output = self.output_fc(F.relu(hidden))
        return torch.sigmoid(output) if self.output_dim == 1 else output
