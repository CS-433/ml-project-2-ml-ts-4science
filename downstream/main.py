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


# Set the random seed for reproducibility
torch.manual_seed(42)

# Create the dataset
data_path = "/scratch/izar/dlopez/ml4science/data/BACH/embeddings/embeddings_uni"
label_path = "/scratch/izar/dlopez/ml4science/data/BACH/labels.csv"

# Create the dataset with the stacked tensors
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
    indices, labels_list, test_size=0.3, stratify=labels_list, random_state=42)

# Further stratified split of temp into val (15%) and test (15%)
val_indices, test_indices, val_labels, test_labels = train_test_split(
    temp_indices, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

# Create Subset objects
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Print the number of classes in each class 
print(f"Number of classes in train set: {len(set(train_labels))}")
print(f"Number of classes in val set: {len(set(val_labels))}")

batch_dim = 4
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_dim, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_dim, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_dim, shuffle=False)



# Initialize the model
input_dim = 1024
model = models.AttentionMLP(input_dim=input_dim, output_dim=dataset.num_labels, batch_dim=batch_dim, dropout_rate=0)

# Early stopping and checkpoint callbacks
early_stopping = pl.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min'
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath='.',
    filename='best_model',
    save_top_k=1,
    mode='min'
)

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[early_stopping, checkpoint_callback]
)

# Train the model
trainer.fit(model, train_loader, val_loader)

# Test the model
trainer.test(model, test_loader)