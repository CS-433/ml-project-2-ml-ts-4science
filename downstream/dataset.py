import torch
import torchvision.transforms.functional as F

import numpy as np 
import os

class EmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, coordinates, transform=False):
        self.embeddings = embeddings      # Tensor de forma (N, E, 1024)
        self.coordinates = coordinates    # Tensor de forma (N, E, ...)
        self.transform = transform

    def __len__(self):
        return self.embeddings.shape[0]   # N

    def __getitem__(self, idx):
        sample = {
            "embedding": self.embeddings[idx],      # Forma: (E, 1024)
            "coordinate": self.coordinates[idx],    # Forma: (E, ...)
        }

        if self.transform:
            # Mean-pooling sobre la dimensión E
            sample["embedding"] = torch.mean(sample["embedding"], dim=0)  # Forma: (1024,)

            # L2 Normalización
            # sample["embedding"] = F.normalize(sample["embedding"], p=2.0)

        return sample
    



if __name__ == "__main__":
    # Example usage of the EmbeddingsDataset with BACH
    data_path = "/home/ricoiban/ml-project-2-ml-ts-4science/data/data/BACH/embeddings/embeddings_uni"
    
    # Load all the embeddings and coordinates from separate .npz files
    embeddings = []
    coordinates = []
    for file in os.listdir(data_path):
        if file.endswith(".npz"):
            data = np.load(os.path.join(data_path, file))

            embeddings.append(torch.tensor(data["embeddings"]))
            coordinates.append(torch.tensor(data["coordinates"]))

    # Stack the embeddings and coordinates along the first dimension (N)
    embeddings = torch.stack(embeddings, dim=0)       # Shape: (N, E, 1024)
    coordinates = torch.stack(coordinates, dim=0)    # Shape: (N, E, ...)

    print(f"Embeddings shape: {embeddings.shape}")  # e.g., (N, E, 1024)

    # Create the dataset with the stacked tensors
    dataset = EmbeddingsDataset(embeddings, coordinates, transform=True)
    
    # Test if the dataset works correctly
    print("First sample in the dataset:")
    print(dataset[0])
    
    print(f"Total number of samples in the dataset: {len(dataset)}")
    
    print(f"Shape of the embedding vector in the first sample: {dataset[0]['embedding'].shape}")      # Should be (1024,)
    print(f"Shape of the coordinate data in the first sample: {dataset[0]['coordinate'].shape}")    # Should match the coordinate dimensions
    
    print("Embedding vector of the first sample:")
    print(dataset[0]["embedding"])            # Print the embedding vector
    
    print("Coordinate data of the first sample:")
    print(dataset[0]["coordinate"])           # Print the coordinate data

    # Initialize the DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Test if the DataLoader works correctly
    print("\nTesting DataLoader with a single batch:")
    for batch in dataloader:
        print(f"Batch embedding shape: {batch['embedding'].shape}")      # Should be (batch_size, 1024)
        print(f"Batch coordinate shape: {batch['coordinate'].shape}")    # Should match the coordinate dimensions
        break