import torch.nn.functional as F
from torch import nn
import torch

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        return self.linear(x)

class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        attention_scores = self.attention(x)  # Shape: (n_samples, n_tiles, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        x = torch.sum(attention_weights * x, dim=1)  # Shape: (n_samples, n_dim)
        x = self.linear(x)  # Shape: (n_samples, output_dim)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout_rate=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AttentionMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout_rate=0):
        super(AttentionMLP, self).__init__()

        self.attention = nn.Linear(input_dim, 1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attention_scores = self.attention(x)  # Shape: (n_samples, n_tiles, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        x = torch.sum(attention_weights * x, dim=1)  # Shape: (n_samples, n_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Shape: (n_samples, output_dim)
        return x