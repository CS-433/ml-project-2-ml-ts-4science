import torch.nn.functional as F
from torch import nn
import torch
import pytorch_lightning as pl

class LinearModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 1e-3

    def forward(self, x):
        x = torch.mean(x, dim=1)
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class Attention(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.linear = nn.Linear(input_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 1e-3

    def forward(self, x):
        attention_scores = self.attention(x)  # Shape: (n_samples, n_tiles, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        x = torch.sum(attention_weights * x, dim=1)  # Shape: (n_samples, n_dim)
        x = self.linear(x)  # Shape: (n_samples, output_dim)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class MLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout_rate=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 1e-3

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        print("[MLP] Training step with ", x, " and ", y)
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class AttentionMLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout_rate=0):
        super(AttentionMLP, self).__init__()

        self.attention = nn.Linear(input_dim, 1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 1e-3

    def forward(self, x):
        attention_scores = self.attention(x)  # Shape: (n_samples, n_tiles, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        x = torch.sum(attention_weights * x, dim=1)  # Shape: (n_samples, n_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Shape: (n_samples, output_dim)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        print("[AttMLP] Training step with ", x, " and ", y)

        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer