import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import os
import warnings
from collections import Counter

import models
from dataset import EmbeddingsDataset

warnings.filterwarnings("ignore")

# Argparser to set model configuration
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BRACS")
parser.add_argument("--augmentation", type=str, default="5")
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

train_indices, temp_indices, train_labels, temp_labels = train_test_split(
    indices, labels_list, test_size=0.3, stratify=labels_list, random_state=42
)
val_indices, test_indices, val_labels, test_labels = train_test_split(
    temp_indices, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

print(f"Number of classes in train set: {len(set(train_labels))}")
print(f"Number of classes in val set: {len(set(val_labels))}")

batch_dim = 64
train_loader = DataLoader(train_dataset, batch_size=batch_dim, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_dim, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_dim, shuffle=False)

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

best_val_loss = float("inf")
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
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # model returns pred_graphs, x_graphs
        preds, _ = model(batch)
        loss = criterion(preds, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.x.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_preds_all = []
    val_targets_all = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds, _ = model(batch)
            loss = criterion(preds, batch.y)
            val_loss += loss.item() * batch.x.size(0)
            val_preds_all.append(torch.argmax(preds, dim=1).cpu())
            val_targets_all.append(batch.y.cpu())

    val_loss /= len(val_loader.dataset)
    val_preds_all = torch.cat(val_preds_all)
    val_targets_all = torch.cat(val_targets_all)
    val_acc = (val_preds_all == val_targets_all).float().mean().item()
    val_f1 = models.compute_f1(val_preds_all, val_targets_all)

    # Check if best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }

    print(f"Epoch [{epoch+1}/{args.epochs}]: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    history["epoch"].append(epoch+1)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

# Load best model
model.load_state_dict(best_model_state["model_state_dict"])

# Test
model.eval()
test_preds_all = []
test_targets_all = []
with torch.no_grad():
    test_loss = 0.0
    for batch in test_loader:
        batch = batch.to(device)
        preds, _ = model(batch)
        loss = criterion(preds, batch.y)
        test_loss += loss.item() * batch.x.size(0)
        test_preds_all.append(torch.argmax(preds, dim=1).cpu())
        test_targets_all.append(batch.y.cpu())

test_loss /= len(test_loader.dataset)
test_preds_all = torch.cat(test_preds_all)
test_targets_all = torch.cat(test_targets_all)
test_acc = (test_preds_all == test_targets_all).float().mean().item()
test_f1 = models.compute_f1(test_preds_all, test_targets_all)

print(f"Test Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")

# Save metrics to CSV in results/
os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame({
    "epoch": history["epoch"],
    "train_loss": history["train_loss"],
    "val_loss": history["val_loss"],
    "val_acc": history["val_acc"],
    "val_f1": history["val_f1"]
})

# Append final test results as a single row at the end
test_summary = pd.DataFrame({
    "epoch": ["final_test"],
    "train_loss": [None],
    "val_loss": [None],
    "val_acc": [None],
    "val_f1": [None],
    "test_loss": [test_loss],
    "test_acc": [test_acc],
    "test_f1": [test_f1]
})
results_df = pd.concat([results_df, test_summary], ignore_index=True)

output_file = f"results/{dataset_name}_{augmentation}x_{args.pooling}_{args.nlayers_classifier}layers_dropout{args.dropout_ratio}.csv"
results_df.to_csv(output_file, index=False)
print(f"Merged metrics saved in {output_file}")
