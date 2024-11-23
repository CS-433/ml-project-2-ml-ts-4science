import torch
import torchvision.transforms.functional as F

import numpy as np 
import os

class EmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=False):
        self.embeddings, self.coordinates = self.create_dataset(data_path)
        self.transform = transform
        self.data_path = data_path

    def __len__(self):
        return self.embeddings.shape[0]   # N

    def __getitem__(self, idx):
        sample = {
            "embedding": self.embeddings[idx],      # Forma: (E, 1024)
            "coordinate": self.coordinates[idx],    # Forma: (E, ...)
        }

        if self.transform:
            # Mean-pooling over E dimension
            sample["embedding"] = torch.mean(sample["embedding"], dim=0)  # Forma: (1024,)

            # L2 Normalization
            norm = torch.norm(sample["embedding"], p=2)
            eps = 1e-8
            sample["embedding"] = sample["embedding"] / (norm + eps)

        return sample
    


    def create_dataset(self, data_path):
        embeddings = []
        coordinates = []
        for file in os.listdir(data_path):
            if file.endswith(".npz"):
                data = np.load(os.path.join(data_path, file))

                embeddings.append(torch.tensor(data["embeddings"]))
                coordinates.append(torch.tensor(data["coordinates"]))

        embeddings = torch.stack(embeddings, dim=0)       # Shape: (N, E, 1024)
        coordinates = torch.stack(coordinates, dim=0)    # Shape: (N, E, ...)

        return embeddings, coordinates



if __name__ == "__main__":
    # Example usage of the EmbeddingsDataset with BACH
    data_path = "/scratch/izar/dlopez/ml4science/data/BACH/embeddings/embeddings_uni"


    # Create the dataset with the stacked tensors
    dataset = EmbeddingsDataset(data_path, transform=True)
    
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