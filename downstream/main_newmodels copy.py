import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
import pandas as pd
import os
import warnings
from collections import Counter
import torchmetrics
import numpy as np

import models
from dataset import EmbeddingsDataset

warnings.filterwarnings("ignore")

# Argparser to set model configuration
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BACH")
parser.add_argument("--augmentation", type=str, default="20")
parser.add_argument("--pooling", type=str, default="GatedAttention", choices=["GatedAttention", "mean"], help="Pooling method")
parser.add_argument("--nlayers_classifier", type=int, default=1, help="Number of layers in classifier head")
parser.add_argument("--dropout_ratio", type=float, default=0.4, help="Dropout ratio in classifier head")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)

# Paths
dataset_name = args.dataset
augmentation = args.augmentation
path = "/mnt/lts4-pathofm/scratch/data/ml4science/" + dataset_name + "/"
data_path = path + "embeddings/embeddings_uni_" + augmentation + "x/"
label_path = path + "labels.csv"

# Create dataset and splits
dataset = EmbeddingsDataset(data_path, label_path, transform=True)
indices = list(range(len(dataset)))
labels_list = [dataset.label_to_index[label] for label in dataset.label_strings]
class_counts = Counter(labels_list)
print("Class distribution in the original dataset:")
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count} samples")

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels_list,
    random_state=42
)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_labels = [labels_list[i] for i in train_indices]
train_dataset_indices = list(range(len(train_dataset)))


# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_fold_indices, val_fold_indices) in enumerate(kf.split(train_dataset_indices, train_labels)):
    print(f"Fold {fold+1}")
    
    train_dataset_fold = Subset(train_dataset_fold, train_fold_indices)
    val_dataset_fold = Subset(train_dataset_fold, val_fold_indices)
    
    
    batch_dim = 64
    train_loader = DataLoader(train_dataset_fold, batch_size=batch_dim, shuffle=True)
    val_loader = DataLoader(val_dataset_fold, batch_size=batch_dim, shuffle=False)
    
    input_dim = 1024
    n_classes = dataset.num_labels

    # Initialize model from the integrated MIL_model
    model = models.MIL_model(
        input_dim=input_dim,
        n_classes=n_classes,
        nlayers_classifier=args.nlayers_classifier,
        dropout_ratio=args.dropout_ratio,
        pooling=args.pooling,
        attention_hidden_dim=128
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Inicializar métricas de torchmetrics
    metric_f1 = torchmetrics.F1Score(task='multiclass', num_classes=n_classes, average='weighted').to(device)

    # Configurar early stopping basado en val_f1
    best_val_f1 = 0.0
    patience = 20
    counter = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # model returns pred_graphs, x_graphs
            preds, _ = model(batch)
            loss = criterion(preds, batch.y)
            loss.backward() 
            optimizer.step()
            train_loss += loss.item() 
            num_batches += 1

        # train_loss /= num_batches

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds, _ = model(batch)
                loss = criterion(preds, batch.y)
                val_loss += loss.item() 

                # Actualizar métricas
                metric_f1.update(preds.softmax(dim=-1), batch.y)

        # val_loss /= len(val_loader.dataset)
        val_f1 = metric_f1.compute().item()
        metric_f1.reset()

        # Comprobar si es el mejor modelo basado en val_f1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping in epoch: {epoch+1}")
                break

        print(f"Epoch [{epoch+1}/{args.epochs}]: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

    # Load best model for this fold
    model.load_state_dict(best_model_state)

    # Save metrics for this fold
    fold_results.append({
        "fold": fold + 1,
        "val_loss": val_loss,
        "val_f1": best_val_f1
    })

    # Save metrics to CSV for this fold
    fold_history_df = pd.DataFrame({
        "epoch": history["epoch"],
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_f1": history["val_f1"]
    })
    path_dataset_result = "results/" + dataset_name + "/" + f"{augmentation}x_{args.pooling}_{args.nlayers_classifier}layers_dropout{args.dropout_ratio}/"
    os.makedirs(path_dataset_result, exist_ok=True)
    fold_output_file = path_dataset_result + f"{dataset_name}_{augmentation}x_fold{fold+1}.csv"
    # fold_history_df.to_csv(fold_output_file, index=False)
    # Create results directory if it does not exist
    print(f"Metrics for fold {fold+1} saved in {fold_output_file}")

val_loss_list = [r["val_loss"] for r in fold_results]
val_f1_list = [r["val_f1"] for r in fold_results]

avg_val_loss = np.mean(val_loss_list)
std_val_loss = np.std(val_loss_list, ddof=1)
avg_val_f1 = np.mean(val_f1_list)
std_val_f1 = np.std(val_f1_list, ddof=1)

print(f"Average Validation Results over {kf.get_n_splits()} folds:")
print(f"Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
print(f"F1 Score: {avg_val_f1:.4f} ± {std_val_f1:.4f}")

# Save all fold results to CSV
fold_results_df = pd.DataFrame({
    "fold": [r["fold"] for r in fold_results],
    "val_loss": val_loss_list,
    "val_f1": val_f1_list
})

fold_results_df.loc["mean"] = ["mean", avg_val_loss, avg_val_f1]
fold_results_df.loc["std"] = ["std", std_val_loss, std_val_f1]

output_file = f"results/{dataset_name}_{augmentation}x_{args.pooling}_{args.nlayers_classifier}layers_dropout{args.dropout_ratio}_kfold.csv"
# fold_results_df.to_csv(output_file, index=False)
print(f"Fold metrics saved in {output_file}")
