import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import models
from dataset import EmbeddingsDataset
from collections import Counter
import pytorch_lightning as pl
import argparse
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# Define argparser for dataset
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BRACS")
parser.add_argument("--augmentation", type=str, default=5)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set the random seed for reproducibility
torch.manual_seed(42)

# Paths
dataset_name = args.dataset
augmentation = args.augmentation
path = "/mnt/lts4-pathofm/scratch/data/ml4science/" + dataset_name + "/"
data_path = path + "embeddings/embeddings_uni_" + augmentation + "x/"
label_path = path + "labels.csv"

# Create the dataset
dataset = EmbeddingsDataset(data_path, label_path, transform=True)

# Extract labels and indices
indices = list(range(len(dataset)))
labels_list = [dataset.label_to_index[label] for label in dataset.label_strings]

# Count the occurrences of each class label
class_counts = Counter(labels_list)

print("Class distribution in the original dataset:")
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count} samples")

# Stratified split into train (70%) and temp (30%)
train_indices, temp_indices, train_labels, temp_labels = train_test_split(
    indices, labels_list, test_size=0.3, stratify=labels_list, random_state=42
)

# Further stratified split of temp into val (15%) and test (15%)
val_indices, test_indices, val_labels, test_labels = train_test_split(
    temp_indices, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# Create Subset objects
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

print(f"Number of classes in train set: {len(set(train_labels))}")
print(f"Number of classes in val set: {len(set(val_labels))}")

batch_dim = 64
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_dim, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_dim, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_dim, shuffle=False)

# Initialize the model
input_dim = 1024
embed_dim = 128  # Internal embed size for Gated Attention
model = models.AttentionMLP(
    input_dim=input_dim,
    output_dim=dataset.num_labels,
    embed_dim=embed_dim,
    dropout_rate=0.4,
).to(device)

# Early stopping and checkpoint callbacks
early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss", dirpath=".", filename="best_model", save_top_k=1, mode="min"
)

# CSV Logger to save metrics
csv_logger = pl.loggers.CSVLogger(
    save_dir="logs", name=f"{dataset_name}_{augmentation}x"
)

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[early_stopping, checkpoint_callback],
    accelerator="cuda" if torch.cuda.is_available() else None,
    devices=1,
    logger=csv_logger,  # Use the CSV logger
)

# Train the model
trainer.fit(model, train_loader, val_loader)

# Test the model using the best checkpoint
trainer.test(ckpt_path="best", dataloaders=test_loader)

# After training and testing, save metrics to CSV
metrics_csv_path = os.path.join(csv_logger.log_dir, "metrics.csv")
metrics_df = pd.read_csv(metrics_csv_path)

# Post-processing to merge rows with the same epoch and step
# Group by 'epoch' and 'step' and take the first non-null value for each metric
merged_metrics_df = metrics_df.groupby(["epoch", "step"]).first().reset_index()


# Save the merged metrics in a CSV file named DATASET_AUGMENTATIONx.csv
output_file = f"results/{dataset_name}_{augmentation}x.csv"
merged_metrics_df.to_csv(output_file, index=False)
print(f"Merged metrics saved in results/{output_file}")
