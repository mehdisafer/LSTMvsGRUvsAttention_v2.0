import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    GRU (input_dim=1, hidden_dim=64, num_layers=5, output_dim=1).
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, hn = self.gru(x)
        last_hid = out[:, -1, :]
        out = self.fc(last_hid)
        return out
