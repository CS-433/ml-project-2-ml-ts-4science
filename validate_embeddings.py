import os
import pandas as pd

# Define file paths
embeddings_dir = 'data/BRACS/embeddings/embeddings_uni_40x'
metadata_file = 'data/BRACS/images_metadata_slurm_40x.csv'

# List all .npz files in the embeddings directory
embedding_files = [
    file.replace('.npz', '') for file in os.listdir(embeddings_dir) if file.endswith('.npz')
]

# Load the metadata file
metadata_df = pd.read_csv(metadata_file)

# Extract image names from the "Image Name" column
metadata_images = metadata_df['Image Name'].tolist()

# Find images in the metadata that don't have a corresponding embedding file
missing_embeddings = [
    image for image in metadata_images if image not in embedding_files
]

# Output the results
print(f"Total images without embeddings: {len(missing_embeddings)}")
print("Missing embeddings:")
print(missing_embeddings)