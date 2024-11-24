import torch 
from dataset import EmbeddingsDataset as ds
import models.LinearModel as ln


# Create the dataset 

data_path = "/scratch/izar/dlopez/ml4science/data/BACH/embeddings/embeddings_uni"
label_path = "/scratch/izar/dlopez/ml4science/data/BACH/labels.csv"

dataset = ds(data_path, label_path, transform=True)

# Create dataloader 


train, test = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])

train_dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
# Create the model

model = ln.LinearModel(1024, 4)


# Define the loss function and optimizer

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Train the model

num_epochs = 20

for epoch in range(num_epochs):

    for i, batch in enumerate(train_dataloader):
        embeddings = batch["embedding"]
        labels = batch["label_idx"]

        # Forward pass
        outputs = model(embeddings)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item()}")

# Test the model

test_dataloader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

model.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for batch in test_dataloader:
        embeddings = batch["embedding"]
        labels = batch["label_idx"]

        outputs = model(embeddings)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the test set: {100 * correct / total}%")


