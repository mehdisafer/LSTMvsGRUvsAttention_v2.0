import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM pour input_dim=1 -> output_dim=1 (Close).
    hidden_dim=64, num_layers=5, etc. -> correspond au config.json
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        out, (hn, cn) = self.lstm(x)
        last_hid = out[:, -1, :]
        out = self.fc(last_hid)
        return out
