import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import models
from dataset import EmbeddingsDataset as ds
from collections import Counter

# Set the random seed for reproducibility
torch.manual_seed(42)

# Create the dataset
data_path = "/scratch/izar/dlopez/ml4science/data/BACH/embeddings/embeddings_uni"
label_path = "/scratch/izar/dlopez/ml4science/data/BACH/labels.csv"

dataset = ds(data_path, label_path)

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


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create the model
input_dim = 1024
print(dataset.num_classes)
dropout = 0 # {0, 0.2, 0.5}
model = models.AttentionMLP(input_dim, dataset.num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Early stopping parameters
num_epochs = 100
patience = 5
best_val_loss = float('inf')
trigger_times = 0

# Training loop with early stopping
for epoch in range(num_epochs):

    # Training phase
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        embeddings = batch["embedding"]
        labels = batch["label_idx"]

        # Forward pass
        outputs = model(embeddings)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch["embedding"]
            labels = batch["label_idx"]

            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0

        # Save the model
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        trigger_times += 1
        print(f'EarlyStopping counter: {trigger_times} out of {patience}')
        if trigger_times >= patience:
            print('Early stopping!')
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pt'))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for batch in test_loader:
        embeddings = batch["embedding"]
        labels = batch["label_idx"]

        outputs = model(embeddings)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the test set: {100 * correct / total}%")
