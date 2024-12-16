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
import faiss

import models
from dataset import EmbeddingsDataset

warnings.filterwarnings("ignore")

# Argparser to set model configuration
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BACH")
parser.add_argument("--augmentation", type=str, default="5")
parser.add_argument("--pooling", type=str, default="GatedAttention", choices=["GatedAttention", "mean"], help="Pooling method")
parser.add_argument("--nlayers_classifier", type=int, default=1, help="Number of layers in classifier head")
parser.add_argument("--dropout_ratio", type=float, default=0.4, help="Dropout ratio in classifier head")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--similarity", type=str, default="cosine", help="Similarity metric")
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
batch_dim = 64

# Create dataset and splits
embeddings, _, labels, _ = EmbeddingsDataset.create_dataset(data_path, label_path)

dataset = []
for i in range(len(embeddings)):
    dataset.append(torch.mean(embeddings[i], axis=0))
dataset = np.array(dataset)

if args.similarity == "cosine":
    faiss.normalize_L2(dataset)

# Convert labels to indices
unique_labels = sorted(set(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
label_strings = labels.copy()  # Keep the original labels as strings
labels = torch.tensor([label_to_index[label] for label in labels])
n_classes = len(unique_labels)

indices = list(range(len(dataset)))
labels_list = [label_to_index[label] for label in label_strings]
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
    
    train_dataset_fold = Subset(train_dataset, train_fold_indices)
    X_train = np.array([train_dataset_fold[i] for i in range(len(train_dataset_fold))])
    y_train = np.array([train_labels[i] for i in train_fold_indices])

    val_dataset_fold = Subset(train_dataset, val_fold_indices)
    X_val = np.array([val_dataset_fold[i] for i in range(len(val_dataset_fold))])
    y_val = np.array([train_labels[i] for i in val_fold_indices])
    y_val = torch.from_numpy(y_val)
    
    input_dim = 1024
    if args.similarity == "cosine":
        faiss_index = faiss.IndexFlatIP(input_dim)
    elif args.similarity == "euclidean":
        faiss_index = faiss.IndexFlatL2(input_dim)
    else:
        raise ValueError("Invalid similarity metric")

    faiss_index.add(X_train)

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    f1_scores = []
    metric_f1 = torchmetrics.F1Score(task='multiclass', num_classes=n_classes, average='weighted')
    similarities, indices = faiss_index.search(X_val, max(k_values))
    
    f1_history = {}
    
    for k in k_values:
        neighbor_weights = similarities[:, :k]
        neighbor_labels = y_train[indices[:, :k]]
        
        weighted_votes = np.zeros((len(y_val), n_classes))
        
        row_idx = np.arange(len(y_val))[:, None].repeat(k, axis=1).ravel()
        col_idx = neighbor_labels.ravel()
        weights = neighbor_weights.ravel()

        np.add.at(weighted_votes, (row_idx, col_idx), weights)
        
        weighted_votes = torch.from_numpy(weighted_votes)
        metric_f1.update(weighted_votes, y_val)
        val_f1 = metric_f1.compute().item()
        metric_f1.reset()

        f1_history[k] = val_f1

        print(f"Fold {fold+1}, k={k}: F1 Score: {val_f1:.4f}")

    fold_results.append({
        "fold": fold + 1,
        "val_f1": f1_history
    })

    # Save metrics to CSV for this fold
    fold_history_df = pd.DataFrame({
        "k": list(f1_history.keys()),
        "val_f1": list(f1_history.values())
    })
    path_dataset_result = "results/" + dataset_name + "/" + f"{augmentation}x_{args.similarity}/"
    os.makedirs(path_dataset_result, exist_ok=True)
    fold_output_file = path_dataset_result + f"{dataset_name}_{augmentation}x_fold{fold+1}.csv"
    fold_history_df.to_csv(fold_output_file, index=False)
    print(f"Metrics for fold {fold+1} saved in {fold_output_file}")

# Calculate average results across folds
k_values = list(fold_results[0]["val_f1"].keys())
avg_f1_by_k = {k: np.mean([fold["val_f1"][k] for fold in fold_results]) for k in k_values}
std_f1_by_k = {k: np.std([fold["val_f1"][k] for fold in fold_results], ddof=1) for k in k_values}

print(f"Average Validation Results over {kf.get_n_splits()} folds:")
for k in k_values:
    print(f"k={k}: F1 Score: {avg_f1_by_k[k]:.4f} ± {std_f1_by_k[k]:.4f}")

# Save all fold results to CSV
fold_results_df = pd.DataFrame({
    "k": k_values,
    "avg_f1": [avg_f1_by_k[k] for k in k_values],
    "std_f1": [std_f1_by_k[k] for k in k_values]
})

output_file = f"results/{dataset_name}_{augmentation}x_{args.similarity}_kfold.csv"
fold_results_df.to_csv(output_file, index=False)
print(f"Fold metrics saved in {output_file}")