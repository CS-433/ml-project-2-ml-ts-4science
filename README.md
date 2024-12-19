### ML4Science: The Impact of Whole-Slide Image Resolution on Foundation Model Performance in Computational Pathology

This project focuses on extracting, analysing and using embeddings from histopathology datasets to train and evaluate models for various classification tasks. The ultimate goal is to apply deep learning methods, including ViT-based feature extraction, to facilitate tissue subtyping, pathology classification and related tasks, and to evaluate the relevance of the magnification level of the dataset in the classification results.

---

### Directory Structure

```
CS-433-ml-project-2-ml-ts-4science
├── preprocessing
│   ├── preprocess.py               # Patch generation and metadata creation
│   ├── generate_metadata.py        # Metadata generation for slide images
│   └── generate_bracs_labels.py    # Label extraction for BRACS dataset
├── analysis
│   ├── tiles_analysis.py           # Analysis and visualization of extracted tiles
│   └── output                      # Directory to store output analysis
├── UNI
│   ├── infere_uni.py               # Embedding inference using UNI model
│   ├── download_uni.py             # Script to download UNI model
│   ├── generate.sh                 # Batch script for embedding generation
│   ├── generate_uni_embeddings.sh  # Batch script for distributed embedding inference
│   └── infer_uni_regions.py        # Region-based inference for slides
├── downstream
│   ├── main.py                     # Training script for NN and k-NN methods
│   ├── models.py                   # Implementation of various neural network models
│   ├── analyzing_uni_embeddings.py # Analysis of extracted embeddings
│   └── dataset.py                  # Dataset loader for embeddings and labels
├── requirements.txt                # Python dependencies
├── config.yaml                     # Configuration file for datasets and paths
├── ML4Science.csv                  # Dataset details
└── README.md                       # Project documentation
```

---

### Key Components

#### Preprocessing

1. **Patch Extraction:**
   - The `preprocess.py` script generates image patches based on the specified resolution (MPP) and patch size. It creates metadata containing tile coordinates and details.

2. **Metadata Generation:**
   - `generate_metadata.py` collects slide-level information and saves it as structured metadata.

3. **Label Mapping:**
   - `generate_bracs_labels.py` extracts class labels from filenames for the BRACS dataset.

#### UNI-based Embedding Inference

1. **Embedding Generation:**
   - Scripts like `infere_uni.py` and `infer_uni_regions.py` use the UNI model to compute embeddings for extracted tiles or regions.

2. **Batch Processing:**
   - Batch scripts (`generate.sh`, `generate_uni_embeddings.sh`) are provided for distributed inference across GPUs.

3. **Model Download:**
   - `download_uni.py` downloads the required pretrained UNI model from HuggingFace Hub.

#### Analysis

1. **Tile Analysis:**
   - `tiles_analysis.py` visualizes tiles, checks overlaps, and validates tile extraction.

2. **Embedding Analysis:**
   - `analyzing_uni_embeddings.py` performs statistical analysis and dimensionality reduction on embeddings.

#### Downstream Tasks

1. **Classification Models:**
   - The `main.py` script supports training neural networks and k-NN classifiers using embeddings.

2. **Model Architectures:**
   - `models.py` contains implementations for MLPs, linear models, and attention mechanisms.

3. **Dataset Loader:**
   - `dataset.py` provides a PyTorch-compatible loader for embedding-based datasets.

---

### Datasets

#### Supported Datasets

| Dataset  | Description                                                                                 | Resolution | MPP   | Task                                 |
|----------|---------------------------------------------------------------------------------------------|------------|-------|--------------------------------------|
| BACH     | Subtyping into normal, benign, in situ carcinoma, and invasive carcinoma                    | 20x, 10x, 5x        | 0.42  | Region/RoI-level classification     |
| BRACS    | Subtyping of regions of interest (ROIs) extracted from WSIs                                 | 40x, 20x, 10x, 5x        | 0.25  | Whole-slide image (WSI) classification |
| BreakHis | Histopathological dataset for breast cancer classification                                  | 40x, 20x, 10x, 5x        | 0.25  | Patch-level classification           |

#### Preprocessing Details

- **Resolutions Supported:**
  - 40x, 20x, 10x, and 5x magnifications.
- **Patch Size:**
  - Default: 224x224 pixels.

---

### Methods

#### Mean Pooling
- **Pipeline:**
  | Mean pooling of patch embeddings | → | Linear/MLP classifier |

#### Attention Mechanism
- **Pipeline:**
  | Attention mechanism (weighted patch aggregation) | → | Linear/MLP classifier |

#### Training
- **Neural Networks (NN):**
  - Supports attention pooling and mean pooling.
  - Configurable number of layers and dropout rates.

- **k-Nearest Neighbors (k-NN):**
  - Cosine and Euclidean similarity metrics.
  - Configurable values of `k`.

---

### Usage

#### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CS-433/ml-project-2-ml-ts-4science
   cd CS-433-ml-project-2-ml-ts-4science
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Running Scripts

1. **Preprocessing:**
   ```bash
   python preprocessing/preprocess.py
   ```

2. **Embedding Inference:**
   ```bash
   python UNI/infere_uni.py --data_dir <path_to_data> --model_path <path_to_model>
   ```

3. **Analysis:**
   ```bash
   python analysis/tiles_analysis.py --data_dir <path_to_tiles>
   ```

4. **Training:**
   ```bash
   python downstream/main.py --method nn --dataset BACH --augmentation 20 --pooling GatedAttention --nlayers_classifier 2 --dropout_ratio 0.5 --epochs 100
   ```

   ```bash
   python downstream/main.py --method knn --dataset BACH --augmentation 5 --similarity cosine
   ```

---

### Notes

- Ensure data directories are correctly specified in `config.yaml`.
- Use batch scripts for efficient processing on HPC environments.
- For custom datasets, update `ML4Science.csv` and `config.yaml` accordingly.

---

### Contributors
- Carlos Hurtado Comín
- Mario Rico Ibáñez
- Daniel López Gala

### Supervised by LTS4:
- Sevda Öğüt
- Cédric Vincent-Cuaz
- Vaishnavi Subramanian
