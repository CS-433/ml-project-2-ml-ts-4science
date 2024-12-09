import torch
import torch.nn.functional as F
import numpy as np
import os
import csv

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset


class EmbeddingsDataset(Dataset):
    def __init__(self, data_path, label_path, transform=False):
        """
        Initializes the dataset with embeddings, coordinates, labels, and file names.

        Args:
            data_path (str): Path to the directory containing .npz files.
            label_path (str): Path to the labels.csv file.
            transform (bool): Whether to apply transformations.
        """
        self.embeddings, self.coordinates, self.labels, self.file_names = self.create_dataset(data_path, label_path)
        self.data_path = data_path

        # Build label mapping
        self.unique_labels = sorted(set(self.labels))
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        # self.num_classes = len(self.unique_labels)

        # Convert labels to indices
        self.label_strings = self.labels.copy()  # Keep the original labels as strings
        self.labels = torch.tensor([self.label_to_index[label] for label in self.labels])

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.embeddings)   # N

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the embedding, coordinate, label index, one-hot label, and file name.
        """
        label_idx = self.labels[idx]
        label_str = self.label_strings[idx]
        # one_hot_label = F.one_hot(label_idx, num_classes=self.num_classes)

        # sample = {
        #     "embedding": self.embeddings[idx],      # Shape: (E, 1024)
        #     "coordinate": self.coordinates[idx],    # Shape: (E, ...)
        #     "label_idx": label_idx,                 # Shape: () - label index
        #     "label_str": label_str,                 # Original label string
        #     "one_hot_label": one_hot_label,         # Shape: (num_classes,)
        #     "file_name": self.file_names[idx]       # Shape: () - Single file name
        # }


        sample = Data(x=torch.FloatTensor(self.embeddings[idx].reshape(1,-1,1024)),
                        y=torch.tensor(label_idx))

        return sample

    @staticmethod
    def my_normalize(tensor, p=2.0, eps=1e-8):
        """
        Applies L2 normalization to the input tensor.

        Args:
            tensor (torch.Tensor): The tensor to normalize.
            p (float): The power of the norm. Default is 2.0 for L2 normalization.
            eps (float): Small value to avoid division by zero.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        norm = torch.norm(tensor, p=p)
        return tensor / (norm + eps)

    def create_dataset(self, data_path, label_path):
        """
        Creates the dataset by loading embeddings, coordinates, labels, and file names.

        Args:
            data_path (str): Path to the directory containing .npz files.
            label_path (str): Path to the labels.csv file.

        Returns:
            tuple: Tensors for embeddings, coordinates, labels, and list of file names.
        """
        embeddings = []
        coordinates = []
        labels = []
        file_names = []

        # Read labels.csv and create a mapping from file_name to class
        label_dict = {}
        with open(label_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                file_name, class_label = row
                label_dict[file_name.strip()] = class_label.strip()

        npz_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
        for file in npz_files:
            with np.load(os.path.join(data_path, file)) as data:
                embeddings.append(torch.from_numpy(data["embeddings"]))


                coordinates.append(torch.from_numpy(data["coordinates"]))

            file_name = file.replace('.npz', '')
            class_label = label_dict.get(file_name)
            labels.append(class_label)

            file_names.append(file_name)

        

        return embeddings, coordinates, labels, file_names


if __name__ == "__main__":
    # Example usage of the EmbeddingsDataset with BACH
    data_path = "/scratch/izar/dlopez/ml4science/data/BACH/embeddings/embeddings_uni"
    label_path = "/scratch/izar/dlopez/ml4science/data/BACH/labels.csv"

    # Create the dataset with the stacked tensors
    dataset = EmbeddingsDataset(data_path, label_path, transform=True)
    
    # Test if the dataset works correctly
    print("First sample in the dataset:")
    print(dataset[0])
    
    print(f"Total number of samples in the dataset: {len(dataset)}")
    
    print(f"Shape of the embedding vector in the first sample: {dataset[0].x.shape}")      # Should be (1024,)
    
    print("Embedding vector of the first sample:")
    print(dataset[0].x)            # Print the embedding vector
    
    # print("Coordinate data of the first sample:")
    # print(dataset[0]["coordinate"])           # Print the coordinate data
    
    # print(f"Label index of the first sample: {dataset[0]['label_idx']}")  # Print the label index
    # print(f"Label string of the first sample: {dataset[0]['label_str']}")  # Print the label string
    print(f"One-hot label of the first sample: {dataset[0].y}")  # Print the one-hot label
    # print(f"File name of the first sample: {dataset[0]['file_name']}")  # Print the file name
    
    # Initialize the DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test if the DataLoader works correctly
    print("\nTesting DataLoader with a single batch:")
    for batch in dataloader:
        print(f"Batch embedding shape: {len(batch)}")      # Should be (batch_size, 1024)
        # print(f"Batch coordinate shape: {batch['coordinate'].shape}")    # Should match the coordinate dimensions
        # print(f"Batch label indices: {batch['label_idx']}")              # Print batch label indices
        print(f"Batch labels: {batch.y}")         # Print batch one-hot labels
        # print(f"Batch file names: {batch['file_name']}")                 # Print batch file names
        break
