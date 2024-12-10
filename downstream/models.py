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
            nn.LazyLinear(out_features=emb_dim), nn.Tanh()  # matrix V
        )

        self.attention_U = nn.Sequential(
            nn.LazyLinear(out_features=emb_dim), nn.Sigmoid()  # matrix U
        )

        self.attention_w = nn.Linear(
            self.emb_dim, self.ATTENTION_BRANCHES
        )  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

    def forward(self, data):
        """
        Parameters
        ----------

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
        Att_row = self.attention_w(
            A_V * A_U
        )  # element wise multiplication # KxATTENTION_BRANCHES

        if len(data.y) > 1:
            Att, x_graphs = GatedAttention.batched_attention(data, Att_row)
        else:
            Att = F.softmax(Att_row, dim=0)
            x_graphs = torch.mm(Att.T, data.x)

        return Att, x_graphs

    def batched_attention(data, Att_row):

        list_x_graphs = []
        list_att = []  # list of normalized attention weights
        # needs to loop over bags to compute attention maps
        for i in range(data.ptr.shape[0] - 1):
            start_curr_graph = data.ptr[i].item()
            end_curr_graph = data.ptr[i + 1].item()
            local_att = F.softmax(Att_row[start_curr_graph:end_curr_graph, :], dim=0)

            x_graphs = torch.mm(local_att.T, data.x[start_curr_graph:end_curr_graph, :])
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
        # print("[MLP] Training step with ", x, " and ", y)
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class AttentionMLP(pl.LightningModule):
    def __init__(
        self, input_dim, output_dim, embed_dim=128, hidden_dim=128, dropout_rate=0
    ):
        super(AttentionMLP, self).__init__()

        self.attention = GatedAttention(input_dim, embed_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 1e-3
        self.num_classes = output_dim

        # Lists to store predictions and targets for validation and test
        self.val_outputs = []
        self.val_targets = []

        self.test_outputs = []
        self.test_targets = []

    def forward(self, batch):
        attention_weights, x = self.attention(batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self.criterion(outputs, batch.y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self.criterion(outputs, batch.y)

        preds = torch.argmax(outputs, dim=1)
        self.val_outputs.append(preds.detach().cpu())
        self.val_targets.append(batch.y.detach().cpu())

        # Log loss per epoch
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.val_outputs, dim=0)
        all_targets = torch.cat(self.val_targets, dim=0)

        # Retrieve the average validation loss (already logged by Lightning)
        val_loss_epoch = self.trainer.callback_metrics.get(
            "val_loss", torch.tensor(float("nan"))
        )

        # Compute accuracy
        val_acc_epoch = (all_preds == all_targets).float().mean()

        # Compute macro F1-score
        val_f1 = self.compute_f1_score(all_preds, all_targets, self.num_classes)

        # Log additional metrics
        self.log("validation_epoch_accuracy", val_acc_epoch, prog_bar=True)
        self.log("validation_epoch_f1", val_f1, prog_bar=True)

        # Print results in a cleaner format
        print(
            f"\n[Validation Epoch End] Avg Loss: {val_loss_epoch:.4f}, "
            f"Avg Acc: {val_acc_epoch:.4f}, F1: {val_f1:.4f}\n"
        )

        # Clear buffers
        self.val_outputs.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self.criterion(outputs, batch.y)

        # Store logits for full test epoch metrics
        self.test_outputs.append(outputs.detach().cpu())
        self.test_targets.append(batch.y.detach().cpu())

        return loss

    def on_test_epoch_end(self):
        # Concatenate all logits and targets
        all_logits = torch.cat(self.test_outputs, dim=0)
        all_targets = torch.cat(self.test_targets, dim=0)

        test_loss = self.criterion(all_logits, all_targets.long())
        all_preds = torch.argmax(all_logits, dim=1)
        test_acc = (all_preds == all_targets).float().mean()
        test_f1 = self.compute_f1_score(all_preds, all_targets, self.num_classes)

        # Log final test metrics
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", test_acc, on_epoch=True, prog_bar=True)
        self.log("test_f1", test_f1, on_epoch=True, prog_bar=True)

        # Print test results in a cleaner format
        print(
            f"\n[Test Epoch End] Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}\n"
        )

        # Clear buffers
        self.test_outputs.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def compute_f1_score(all_preds, all_targets, num_classes):
        # Build confusion matrix: rows = true classes, columns = predicted classes
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        for t, p in zip(all_targets, all_preds):
            conf_matrix[t, p] += 1

        # True positives (TP): diagonal of the confusion matrix
        tp = torch.diag(conf_matrix).float()

        # Calculate FP and FN for each class
        fn = conf_matrix.sum(dim=1).float() - tp
        fp = conf_matrix.sum(dim=0).float() - tp

        epsilon = 1e-10
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1_per_class = 2 * (precision * recall) / (precision + recall + epsilon)

        # Macro F1 is the average F1 over all classes
        f1_macro = f1_per_class.mean()
        return f1_macro


# Make sure you have defined the GatedAttention class previously.
# It needs to implement a forward method returning (attention_weights, x),
# where x is the aggregated embedding.
class GatedAttention(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(GatedAttention, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.ATTENTION_BRANCHES = 1

        self.attention_V = nn.Sequential(nn.LazyLinear(out_features=emb_dim), nn.Tanh())

        self.attention_U = nn.Sequential(
            nn.LazyLinear(out_features=emb_dim), nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.emb_dim, self.ATTENTION_BRANCHES)

    def forward(self, data):
        A_V = self.attention_V(data.x)  # K x L
        A_U = self.attention_U(data.x)  # K x L
        Att_row = self.attention_w(A_V * A_U)

        if len(data.y) > 1:
            Att, x_graphs = GatedAttention.batched_attention(data, Att_row)
        else:
            Att = F.softmax(Att_row, dim=0)
            x_graphs = torch.mm(Att.T, data.x)

        return Att, x_graphs

    @staticmethod
    def batched_attention(data, Att_row):
        list_x_graphs = []
        list_att = []
        for i in range(data.ptr.shape[0] - 1):
            start_curr_graph = data.ptr[i].item()
            end_curr_graph = data.ptr[i + 1].item()
            local_att = F.softmax(Att_row[start_curr_graph:end_curr_graph, :], dim=0)
            x_graphs = torch.mm(local_att.T, data.x[start_curr_graph:end_curr_graph, :])
            list_x_graphs.append(x_graphs)
            list_att.append(local_att)

        x_graphs = torch.cat(list_x_graphs, dim=0)
        Att = torch.cat(list_att, dim=0)
        return Att, x_graphs


def compute_f1(all_preds, all_targets):
    """
    Calcula el F1-score macro.
    all_preds y all_targets son tensores 1D de tamaño N con la clase predicha y la clase real.
    """
    # Obtener la lista de clases (asumimos que van desde 0 hasta num_classes-1)
    classes = torch.unique(all_targets)

    f1_scores = []
    for c in classes:
        # Verdaderos Positivos (TP): pred y target == c
        TP = ((all_preds == c) & (all_targets == c)).sum().item()

        # Falsos Positivos (FP): pred == c pero target != c
        FP = ((all_preds == c) & (all_targets != c)).sum().item()

        # Falsos Negativos (FN): pred != c pero target == c
        FN = ((all_preds != c) & (all_targets == c)).sum().item()

        # Cálculo de precisión y recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        # Cálculo de F1 para la clase c
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)

    # F1 macro: promedio sobre todas las clases
    macro_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0.0
    return macro_f1
