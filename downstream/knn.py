import numpy as np
import os
import pandas as pd
import faiss

def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy score.
    Args:
        y_true: array-like of true labels
        y_pred: array-like of predicted labels
    Returns:
        float: accuracy score between 0.0 and 1.0
    """
    return np.mean(y_true == y_pred)


def main():
    dataset = "BACH"
    print(f"Loading embeddings from {dataset} dataset")
    embedding_path = f"/home/carlos/ml-project-2-ml-ts-4science/data/{dataset}/embeddings/embeddings_uni"
    npz_files = [f for f in os.listdir(embedding_path) if f.endswith('.npz')]

    dataset = []
    for npz_file in npz_files:
        file_path = os.path.join(embedding_path, npz_file)
        with np.load(file_path) as data:
            embeddings, coordinates = data.values()
            dataset.append(embeddings)
    dataset = np.array(dataset)

    label_path = "/home/carlos/ml-project-2-ml-ts-4science/data/{dataset}/labels.csv"
    labels_df = pd.read_csv(label_path)
    
    labels = []
    for npz_file in npz_files:
        file_name = npz_file.replace('.npz', '')
        label = labels_df[labels_df['file_name'] == file_name][' class'].iloc[0]
        labels.append(label)

    label_dict = {label: i for i, label in enumerate(set(labels))}
    labels = [label_dict[label] for label in labels]
    labels = np.array(labels)

    X = dataset # (n_samples, image_patches, embedding_size)
    y = labels
    print(f"Loaded {len(X)} embeddings with shape: {X.shape}")
    print(f"Loaded {len(y)} labels")

    # Compute mean embedding for each sample
    X_mean = np.mean(X, axis=1) # (n_samples, embedding_size)
    faiss.normalize_L2(X_mean)

    # Using Cosine similarity, weighted k-NN
    faiss_index = faiss.IndexFlatL2(X_mean.shape[1])
    faiss_index.add(X_mean)
    
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    accuracies = []
    similarities, indices = faiss_index.search(X_mean, max(k_values)+1)
    similarities = similarities[:, 1:]
    indices = indices[:, 1:]
    
    for k in k_values:
        neighbor_weights = similarities[:, :k]
        neighbor_labels = y[indices[:, :k]]
        
        n_classes = len(label_dict)
        weighted_votes = np.zeros((len(y), n_classes))
        
        row_idx = np.arange(len(y))[:, None].repeat(k, axis=1).ravel()
        col_idx = neighbor_labels.ravel()
        weights = neighbor_weights.ravel()

        np.add.at(weighted_votes, (row_idx, col_idx), weights)
        
        predictions = np.argmax(weighted_votes, axis=1)

        acc = accuracy_score(y, predictions)
        accuracies.append(acc)
        print(f"Cosine similarity (weighted k-NN) k={k}: accuracy={acc:.3f}")

    # Using Euclidean distance, k-NN
    X_mean = np.mean(X, axis=1) # (n_samples, embedding_size)

    faiss_index = faiss.IndexFlatL2(X_mean.shape[1])
    faiss_index.add(X_mean)

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    accuracies = []
    similarities, indices = faiss_index.search(X_mean, max(k_values)+1)
    similarities = similarities[:, 1:]
    indices = indices[:, 1:]

    for k in k_values:
        neighbor_labels = y[indices[:, :k]]
        predictions = np.array([np.argmax(np.bincount(labels)) for labels in neighbor_labels])

        acc = accuracy_score(y, predictions)
        accuracies.append(acc)
        print(f"Euclidean distance (k-NN) k={k}: accuracy={acc:.3f}")

if __name__ == "__main__":
    main()