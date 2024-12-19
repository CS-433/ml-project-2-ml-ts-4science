import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
import pandas as pd
import os
import warnings
from collections import Counter
import torchmetrics
import numpy as np
import faiss

import models
from dataset import EmbeddingsDataset

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Neural Network or k-NN models with cross-validation."
    )

    # Common arguments
    parser.add_argument(
        "--method",
        type=str,
        default="nn",
        choices=["nn", "knn"],
        help="Method to use: 'nn' for Neural Network, 'knn' for k-Nearest Neighbors",
    )
    parser.add_argument(
        "--dataset", type=str, default="BACH", help="Name of the dataset"
    )
    parser.add_argument(
        "--augmentation", type=str, default="20", help="Augmentation factor"
    )

    # Arguments specific to Neural Network
    parser.add_argument(
        "--pooling",
        type=str,
        default="GatedAttention",
        choices=["GatedAttention", "mean"],
        help="Pooling method for NN",
    )
    parser.add_argument(
        "--nlayers_classifier",
        type=int,
        default=1,
        help="Number of layers in classifier head for NN",
    )
    parser.add_argument(
        "--dropout_ratio",
        type=float,
        default=0.4,
        help="Dropout ratio in classifier head for NN",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )

    # Arguments specific to k-NN
    parser.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Similarity metric for k-NN",
    )

    return parser.parse_args()


def run_nn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NN] Using device: {device}")

    torch.manual_seed(42)

    # Paths
    dataset_name = args.dataset
    augmentation = args.augmentation
    path = f"/mnt/lts4-pathofm/scratch/data/ml4science/{dataset_name}/"
    data_path = os.path.join(path, "embeddings", f"embeddings_uni_{augmentation}x")
    label_path = os.path.join(path, "labels.csv")
    batch_dim = 64

    # Create dataset and splits
    dataset = EmbeddingsDataset(data_path, label_path, transform=True)
    indices = list(range(len(dataset)))
    labels_list = [dataset.label_to_index[label] for label in dataset.label_strings]
    class_counts = Counter(labels_list)
    print("[NN] Class distribution in the original dataset:")
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} samples")

    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=labels_list, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_dim, shuffle=False)

    train_labels = [labels_list[i] for i in train_indices]
    train_dataset_indices = list(range(len(train_dataset)))

    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_fold_indices, val_fold_indices) in enumerate(
        kf.split(train_dataset_indices, train_labels)
    ):
        print(f"[NN] Fold {fold+1}")

        train_dataset_fold = Subset(train_dataset, train_fold_indices)
        val_dataset_fold = Subset(train_dataset, val_fold_indices)

        train_loader = DataLoader(
            train_dataset_fold, batch_size=batch_dim, shuffle=True
        )
        val_loader = DataLoader(val_dataset_fold, batch_size=batch_dim, shuffle=False)

        input_dim = 1024
        n_classes = dataset.num_labels

        # Initialize model
        model = models.MIL_model(
            input_dim=input_dim,
            n_classes=n_classes,
            nlayers_classifier=args.nlayers_classifier,
            dropout_ratio=args.dropout_ratio,
            pooling=args.pooling,
            attention_hidden_dim=128,
        ).to(device)

        criterion = nn.CrossEntropyLoss(reduction="sum")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Initialize metrics
        metric_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=n_classes, average="weighted"
        ).to(device)

        # Early stopping parameters
        best_val_f1 = 0.0
        patience = 20
        counter = 0
        history = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(args.epochs):
            # Training
            model.train()
            train_loss = 0.0
            num_samples_batch = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                preds, _ = model(batch)
                loss = criterion(preds, batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_samples_batch += batch.ptr.size(0) - 1

            train_loss /= num_samples_batch

            # Validation
            model.eval()
            val_loss = 0.0
            num_samples_batch = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    preds, _ = model(batch)
                    loss = criterion(preds, batch.y)
                    val_loss += loss.item()

                    # Update metrics
                    metric_f1.update(preds.softmax(dim=-1), batch.y)
                    num_samples_batch += batch.ptr.size(0) - 1

            val_loss /= num_samples_batch
            val_f1 = metric_f1.compute().item()
            metric_f1.reset()

            # Check for improvement
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"[NN] Early stopping in epoch: {epoch+1}")
                    break

            print(
                f"[NN] Epoch [{epoch+1}/{args.epochs}]: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}"
            )

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)

        # Load best model for this fold
        model.load_state_dict(best_model_state)

        # Save metrics for this fold
        fold_results.append(
            {"fold": fold + 1, "val_loss": val_loss, "val_f1": best_val_f1}
        )

        # Save metrics to CSV for this fold
        fold_history_df = pd.DataFrame(
            {
                "epoch": history["epoch"],
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
                "val_f1": history["val_f1"],
            }
        )
        path_dataset_result = os.path.join(
            "results",
            dataset_name,
            f"{augmentation}x_{args.pooling}_{args.nlayers_classifier}layers_dropout{args.dropout_ratio}",
        )
        os.makedirs(path_dataset_result, exist_ok=True)
        fold_output_file = os.path.join(
            path_dataset_result, f"{dataset_name}_{augmentation}x_fold{fold+1}.csv"
        )
        fold_history_df.to_csv(fold_output_file, index=False)
        print(f"[NN] Metrics for fold {fold+1} saved in {fold_output_file}")

    # Aggregate fold results
    val_loss_list = [r["val_loss"] for r in fold_results]
    val_f1_list = [r["val_f1"] for r in fold_results]

    avg_val_loss = np.mean(val_loss_list)
    std_val_loss = np.std(val_loss_list, ddof=1)
    avg_val_f1 = np.mean(val_f1_list)
    std_val_f1 = np.std(val_f1_list, ddof=1)

    print(f"[NN] Average Validation Results over {kf.get_n_splits()} folds:")
    print(f"Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"F1 Score: {avg_val_f1:.4f} ± {std_val_f1:.4f}")

    # Save all fold results to CSV
    fold_results_df = pd.DataFrame(
        {
            "fold": [r["fold"] for r in fold_results],
            "val_loss": val_loss_list,
            "val_f1": val_f1_list,
        }
    )

    fold_results_df.loc["mean"] = ["mean", avg_val_loss, avg_val_f1]
    fold_results_df.loc["std"] = ["std", std_val_loss, std_val_f1]

    output_file = os.path.join(
        "results",
        f"{dataset_name}_{augmentation}x_{args.pooling}_{args.nlayers_classifier}layers_dropout{args.dropout_ratio}_kfold.csv",
    )
    fold_results_df.to_csv(output_file, index=False)
    print(f"[NN] Fold metrics saved in {output_file}")

    # Load best model (from the last fold)
    model.load_state_dict(best_model_state)

    # Evaluate on test data
    model.eval()
    test_loss = 0.0
    num_samples_batch = 0
    metric_f1 = torchmetrics.F1Score(
        task="multiclass", num_classes=n_classes, average="weighted"
    ).to(device)
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds, _ = model(batch)
            loss = criterion(preds, batch.y)
            test_loss += loss.item()
            metric_f1.update(preds.softmax(dim=-1), batch.y)
            num_samples_batch += batch.ptr.size(0) - 1
    test_loss /= num_samples_batch
    test_f1 = metric_f1.compute().item()
    metric_f1.reset()

    print(f"[NN] Test Results: Loss: {test_loss:.4f}, F1 Score: {test_f1:.4f}")

    # Save test results
    test_results_df = pd.DataFrame({"test_loss": [test_loss], "test_f1": [test_f1]})
    test_output_file = os.path.join(
        path_dataset_result, f"{dataset_name}_{augmentation}x_test_results.csv"
    )
    test_results_df.to_csv(test_output_file, index=False)
    print(f"[NN] Test metrics saved in {test_output_file}")


def run_knn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[k-NN] Using device: {device}")

    torch.manual_seed(42)

    # Paths
    dataset_name = args.dataset
    augmentation = args.augmentation
    path = f"/mnt/lts4-pathofm/scratch/data/ml4science/{dataset_name}/"
    data_path = os.path.join(path, "embeddings", f"embeddings_uni_{augmentation}x")
    label_path = os.path.join(path, "labels.csv")
    batch_dim = 64

    # Create dataset and splits
    embeddings, _, labels, _ = EmbeddingsDataset.create_dataset(data_path, label_path)

    # Average embeddings
    dataset = []
    for i in range(len(embeddings)):
        dataset.append(torch.mean(embeddings[i], axis=0))
    dataset = np.array(dataset)

    # Normalize if cosine similarity
    if args.similarity == "cosine":
        faiss.normalize_L2(dataset)

    # Convert labels to indices
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    label_strings = labels.copy()  # Keep the original labels as strings
    labels = torch.tensor([label_to_index[label] for label in labels])
    n_classes = len(unique_labels)

    indices = list(range(len(dataset)))
    labels_list = [label_to_index[label] for label in label_strings]
    class_counts = Counter(labels_list)
    print("[k-NN] Class distribution in the original dataset:")
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} samples")

    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=labels_list, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_labels = [labels_list[i] for i in train_indices]
    train_dataset_indices = list(range(len(train_dataset)))

    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_fold_indices, val_fold_indices) in enumerate(
        kf.split(train_dataset_indices, train_labels)
    ):
        print(f"[k-NN] Fold {fold+1}")

        train_dataset_fold = Subset(train_dataset, train_fold_indices)
        X_train = np.array(
            [train_dataset_fold[i] for i in range(len(train_dataset_fold))]
        )
        y_train = np.array([train_labels[i] for i in train_fold_indices])

        val_dataset_fold = Subset(train_dataset, val_fold_indices)
        X_val = np.array([val_dataset_fold[i] for i in range(len(val_dataset_fold))])
        y_val = np.array([train_labels[i] for i in val_fold_indices])
        y_val = torch.from_numpy(y_val)

        input_dim = X_train.shape[1]
        if args.similarity == "cosine":
            faiss_index = faiss.IndexFlatIP(input_dim)
        elif args.similarity == "euclidean":
            faiss_index = faiss.IndexFlatL2(input_dim)
        else:
            raise ValueError("Invalid similarity metric")

        faiss_index.add(X_train)

        k_values = [1, 3, 5, 7, 9, 11, 13, 15]
        f1_scores = []
        metric_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=n_classes, average="weighted"
        ).to(device)
        similarities, indices_search = faiss_index.search(X_val, max(k_values))

        f1_history = {}

        for k in k_values:
            neighbor_weights = similarities[:, :k]
            neighbor_labels = y_train[indices_search[:, :k]]

            weighted_votes = np.zeros((len(y_val), n_classes))

            row_idx = np.arange(len(y_val))[:, None].repeat(k, axis=1).ravel()
            col_idx = neighbor_labels.ravel()
            weights = neighbor_weights.ravel()

            np.add.at(weighted_votes, (row_idx, col_idx), weights)

            weighted_votes = torch.from_numpy(weighted_votes).to(device)
            y_val_device = y_val.to(device)
            metric_f1.update(weighted_votes, y_val_device)
            val_f1 = metric_f1.compute().item()
            metric_f1.reset()

            f1_history[k] = val_f1

            print(f"[k-NN] Fold {fold+1}, k={k}: F1 Score: {val_f1:.4f}")

        fold_results.append({"fold": fold + 1, "val_f1": f1_history})

        # Save metrics to CSV for this fold
        fold_history_df = pd.DataFrame(
            {"k": list(f1_history.keys()), "val_f1": list(f1_history.values())}
        )
        path_dataset_result = os.path.join(
            "results", dataset_name, f"{augmentation}x_{args.similarity}"
        )
        os.makedirs(path_dataset_result, exist_ok=True)
        fold_output_file = os.path.join(
            path_dataset_result, f"{dataset_name}_{augmentation}x_fold{fold+1}.csv"
        )
        fold_history_df.to_csv(fold_output_file, index=False)
        print(f"[k-NN] Metrics for fold {fold+1} saved in {fold_output_file}")

    # Calculate average results across folds
    k_values = list(fold_results[0]["val_f1"].keys())
    avg_f1_by_k = {
        k: np.mean([fold["val_f1"][k] for fold in fold_results]) for k in k_values
    }
    std_f1_by_k = {
        k: np.std([fold["val_f1"][k] for fold in fold_results], ddof=1)
        for k in k_values
    }

    print(f"[k-NN] Average Validation Results over {kf.get_n_splits()} folds:")
    for k in k_values:
        print(f"k={k}: F1 Score: {avg_f1_by_k[k]:.4f} ± {std_f1_by_k[k]:.4f}")

    # Save all fold results to CSV
    fold_results_df = pd.DataFrame(
        {
            "k": k_values,
            "avg_f1": [avg_f1_by_k[k] for k in k_values],
            "std_f1": [std_f1_by_k[k] for k in k_values],
        }
    )

    output_file = os.path.join(
        "results", f"{dataset_name}_{augmentation}x_{args.similarity}_kfold.csv"
    )
    fold_results_df.to_csv(output_file, index=False)
    print(f"[k-NN] Fold metrics saved in {output_file}")


def main():
    args = parse_arguments()

    if args.method == "nn":
        run_nn(args)
    elif args.method == "knn":
        run_knn(args)
    else:
        raise ValueError("Invalid method selected. Choose 'nn' or 'knn'.")


if __name__ == "__main__":
    main()
