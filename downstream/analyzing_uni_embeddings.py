import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

from glob import glob

# ============================
# Initial Configuration
# ============================
dataset = "BACH" # BRACS, ...
magnification = "" #5x, 10x, 20x, 40x
# Directory containing all .npz embedding files
input_dir = f"/scratch/izar/dlopez/ml4science/data/{dataset}/embeddings/embeddings_uni{magnification}/"

# Directory where all visualizations will be saved
output_dir = f"embeddings_analysis_{dataset}/magnification_{magnification}"
os.makedirs(output_dir, exist_ok=True)

# ============================
# 1. Check CUDA Availability
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Compute Device: {device}")

# ============================
# 2. Load and Process All Embeddings
# ============================
# Initialize lists to store embeddings and coordinates
mean_embeddings_list = []
mean_coordinates_list = []

# Get list of all .npz files in the directory
embedding_files = glob(os.path.join(input_dir, "*.npz"))
print(f"Number of files found: {len(embedding_files)}")

for file_idx, file_path in enumerate(embedding_files, 1):
    # Load the .npz file
    embedding_data = np.load(file_path)
    
    # Check available keys
    keys = embedding_data.keys()
    print(f"File {file_idx}: {os.path.basename(file_path)} - Keys: {keys}")
    
    # Extract embeddings and coordinates
    embeddings = embedding_data['embeddings']    # Assume shape (2, 1024)
    coordinates = embedding_data['coordinates']  # Assume shape (2, 1024, 2)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Coordinates shape: {coordinates.shape}")
    
    # Verify if dimensions match
    if embeddings.shape[0] != coordinates.shape[0]:
        raise ValueError(f"The number of embeddings does not match the number of coordinates in the file {file_path}")
    
    # Mean pooling of the two embeddings (average over the first axis)
    mean_embedding = np.mean(embeddings, axis=0)       # Result: (1024,)
    mean_coordinates = np.mean(coordinates, axis=0)   # Result: (1024, 2)
    
    # Store the averaged embeddings and coordinates
    mean_embeddings_list.append(mean_embedding)
    mean_coordinates_list.append(mean_coordinates)
    
    print(f"Mean Embedding shape: {mean_embedding.shape}")
    print(f"Mean Coordinates shape: {mean_coordinates.shape}\n")

# Convert the lists to NumPy arrays
mean_embeddings = np.vstack(mean_embeddings_list)         # Shape: (num_images, 1024)
mean_coordinates = np.vstack(mean_coordinates_list)       # Shape: (num_images, 1024, 2)

print(f"Total embeddings after pooling: {mean_embeddings.shape}")
print(f"Total coordinates after pooling: {mean_coordinates.shape}")

# ============================
# 3. Basic Statistical Analysis
# ============================
mean_embedding_overall = np.mean(mean_embeddings, axis=0)
std_embedding_overall = np.std(mean_embeddings, axis=0)

print(f"Mean of embeddings (per dimension): {mean_embedding_overall[:5]}...")  # Show only the first 5 for brevity
print(f"Standard deviation of embeddings (per dimension): {std_embedding_overall[:5]}...\n")

# ============================
# 4. Embedding Distribution Visualizations
# ============================

# a. Histogram of Embedding Values
plt.figure(figsize=(10, 5))
plt.hist(mean_embeddings.flatten(), bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Embedding Values')
plt.xlabel('Embedding Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "embedding_values_histogram.png"))
plt.close()

# b. Norms of Embeddings
embedding_norms = np.linalg.norm(mean_embeddings, axis=1)  # L2 norm per embedding

plt.figure(figsize=(10, 5))
plt.hist(embedding_norms, bins=50, color='green', alpha=0.7)
plt.title('Distribution of Embedding Norms')
plt.xlabel('Embedding Norm')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "embedding_norms_histogram.png"))
plt.close()

# ============================
# 5. Dimensionality Reduction for Visualization
# ============================

print("Performing PCA for visualization using scikit-learn...")
pca_vis = PCA(n_components=2, random_state=42)
embeddings_pca = pca_vis.fit_transform(mean_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c='blue', alpha=0.5)
plt.title('Embeddings Visualized using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_visualization.png"))
plt.close()


print(f"Exploratory analysis of embeddings completed successfully. All visualizations have been saved in the directory '{output_dir}'.")
