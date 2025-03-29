import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    """
    LSTM + mécanisme d'attention, input_dim=1 -> on ne prend que Close
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(AttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        out, (hn, cn) = self.lstm(x)
        
        # Calcul des scores d'attention
        energy = torch.tanh(self.attn(out))    # (batch, seq_len, hidden_dim)
        scores = self.v(energy).squeeze(2)     # (batch, seq_len)
        
        alpha = F.softmax(scores, dim=1).unsqueeze(1)  # (batch,1,seq_len)
        context = torch.bmm(alpha, out)                # (batch,1,hidden_dim)
        context = context.squeeze(1)                   # (batch, hidden_dim)
        
        out = self.fc(context)                         # (batch,1)
        return out
