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


class GatedAttention(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(GatedAttention, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.ATTENTION_BRANCHES = 1
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.input_dim, self.emb_dim), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.input_dim, self.emb_dim), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.emb_dim, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        
    def forward(self, data):
        """
        Parameters
        ----------
        x_nodes : transform(data.x)
            all node features or embeddings
        data : batch data loaded from torch geometric dataset
            
        Returns
        -------
        Att : attention maps for all nodes
        x_graphs : aggregated graph embeddings (\sum_k Att_k h_k) with h_k node embedding
        """
        # first compute unnormalized attention weights for all samples simultaneously
        
        # parallelized over all instances (nodes)
        A_V = self.attention_V(data.x)  # K x L
        A_U = self.attention_U(data.x)  # K x L
        Att_row = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES

            
        if len(data.y) > 1:
            Att, x_graphs = GatedAttention.batched_attention(data, Att_row)
        else:
            Att = F.softmax(Att_row, dim=0)
            x_graphs = torch.mm(Att.T, data.x)
            
        return Att, x_graphs
    

    def batched_attention(data, Att_row):

        list_x_graphs = []
        list_att = [] # list of normalized attention weights
        # needs to loop over bags to compute attention maps
        for i in range(data.ptr.shape[0] - 1):
            start_curr_graph = data.ptr[i].item()
            end_curr_graph = data.ptr[i + 1].item()
            local_att = F.softmax(
                Att_row[start_curr_graph : end_curr_graph, :], dim=0)
            
            x_graphs = torch.mm(
                local_att.T,
                data.x[start_curr_graph : end_curr_graph, :])
            list_x_graphs.append(x_graphs)
            list_att.append(local_att)
            
        x_graphs = torch.cat(list_x_graphs, dim=0)
        Att = torch.cat(list_att, dim=0)

        return Att, x_graphs

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
    def __init__(self, input_dim, output_dim, batch_dim, hidden_dim=128, dropout_rate=0):
        super(AttentionMLP, self).__init__()

        self.attention = GatedAttention(batch_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 1e-3

    def forward(self, batch):
        
        attention_weights, x = self.attention(batch)  # Shape: (n_samples, n_tiles, 1)
        x = torch.sum(attention_weights * x, dim=1)  # Shape: (n_samples, n_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Shape: (n_samples, output_dim)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        print("[AttMLP] Training step with ", x, " and ", y)

        outputs = self.forward(batch)
        loss = self.criterion(outputs, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    


