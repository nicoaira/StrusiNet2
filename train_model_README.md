# README for RNA Structure Embedding Training Script

## Overview

This script is designed for training deep learning models to generate embeddings from RNA secondary structures. It supports two types of models: **Siamese ResNet-LSTM** and **GIN (Graph Isomorphism Network)**. The script includes data preprocessing, training with early stopping, and metadata logging.

---

## Features

- **Model Types**:
  - Siamese ResNet-LSTM for sequence-based RNA embedding.
  - GIN for graph-based RNA embedding.
- **Early Stopping**: Stops training if the validation loss does not improve for a specified number of epochs.
- **Triplet Loss**: Utilizes triplet loss for training embeddings.
- **Data Validation**: Removes invalid RNA secondary structures based on dot-bracket notation.
- **Logging**: Tracks training progress, parameters, and execution time.
- **Device Support**: Automatically uses GPU if available.

---

## Prerequisites

### Libraries

Ensure the following Python libraries are installed:

- `torch`
- `torch_geometric`
- `pandas`
- `scikit-learn`
- `tqdm`

Install missing dependencies using:

```bash
pip install torch pandas scikit-learn tqdm
```

For `torch_geometric`, follow the installation guide: [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

### Files and Directory Structure

Place the following modules in the `src/` directory:

- `early_stopping.py`: Implements early stopping logic.
- `gin_rna_dataset.py`: Dataset class for GIN models.
- `model/gin_model.py`: GIN model definition.
- `model/siamese_model.py`: Siamese ResNet-LSTM model definition.
- `triplet_loss.py`: Implements triplet loss function.
- `triplet_rna_dataset.py`: Dataset class for Siamese models.
- `utils.py`: Utility functions like logging and validation.

---

## Usage

### Command-line Arguments

Run the script using the following arguments:

```bash
python train_script.py --input_path <path_to_csv> --model_type <siamese|gin> [options]
```

#### Required Arguments
- `--input_path`: Path to the CSV/TSV file containing RNA secondary structures with `structure_A`, `structure_P`, and `structure_N` columns.
- `--model_type`: Model type (`siamese` or `gin`).

#### Optional Arguments
- `--model_id`: Identifier for the model (default: `siamese_model`).
- `--graph_encoding`: Encoding for GIN model (`standard` or `forgi`). Ignored for Siamese models (default: `standard`).
- `--hidden_dim`: Hidden dimension size for the model (default: `256`).
- `--output_dim`: Output embedding size for GIN model (default: `128`).
- `--batch_size`: Batch size for training and validation (default: `100`).
- `--num_epochs`: Number of epochs for training (default: `10`).
- `--patience`: Patience for early stopping (default: `5`).
- `--lr`: Learning rate for the optimizer (default: `0.001`).
- `--gin_layers`: Number of GIN layers (default: `1`).

---

## Input Data Format

The input file should be a CSV/TSV with the following columns:
- `structure_A`: Dot-bracket notation for anchor structure.
- `structure_P`: Dot-bracket notation for positive structure.
- `structure_N`: Dot-bracket notation for negative structure.

---

## Outputs

1. **Model Checkpoints**: Saved to `output/<model_id>/<model_id>.pth` with optimizer state and epoch information.
2. **Training Logs**: Saved to `output/<model_id>/train.log` containing training progress and parameters.
3. **Embeddings**: Not generated during training but can be inferred post-training using the trained model.

---

## Example

To train a Siamese model:

```bash
python train_script.py --input_path data/rna_structures.csv --model_type siamese --hidden_dim 256 --batch_size 64
```

To train a GIN model:

```bash
python train_script.py --input_path data/rna_structures.csv --model_type gin --hidden_dim 128 --gin_layers 2
```

---

## Training Flow

1. **Data Preprocessing**:
   - Loads and validates RNA structures using `is_valid_dot_bracket`.
   - Splits data into training and validation sets.

2. **Model Initialization**:
   - Creates the specified model (Siamese or GIN).
   - Initializes the optimizer and loss function.

3. **Training**:
   - Trains the model for a specified number of epochs or until early stopping is triggered.
   - Logs progress after each epoch.

4. **Model Saving**:
   - Saves the model and optimizer states to a checkpoint file.

5. **Logging**:
   - Logs training parameters, loss metrics, and execution time.

---

## Notes

- Ensure the `input_path` file contains valid RNA structures in the required format.
- For large datasets, use a GPU for faster training.
- You can extend the script by adding new models or loss functions as needed.