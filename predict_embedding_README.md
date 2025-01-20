# README for RNA Embedding Prediction Script

## Overview

This script predicts RNA embeddings using a pre-trained deep learning model. It supports two model types: **Siamese ResNet-LSTM** and **GIN (Graph Isomorphism Network)**. The script processes RNA secondary structures (in dot-bracket notation) from an input CSV/TSV file, computes embeddings, and saves the results to an output file.

---

## Features

- **Model Types**:
  - **Siamese ResNet-LSTM**: Sequence-based RNA embedding.
  - **GIN**: Graph-based RNA embedding using `standard` or `forgi` graph encoding.
- **Input Validation**: Ensures RNA structures are valid dot-bracket strings.
- **Logging**: Tracks runtime parameters, progress, and execution time.
- **Device Support**: Runs on GPU if available, otherwise defaults to CPU.
- **Dynamic Model Loading**: Automatically downloads the pre-trained model if not found locally.

---

## Prerequisites

### Libraries

Ensure the following Python libraries are installed:

- `torch`
- `pandas`
- `tqdm`

Install missing dependencies with:

```bash
pip install torch pandas tqdm
```

For GIN model support, ensure `torch_geometric` is installed following the [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

---

## Usage

### Command-line Arguments

Run the script as follows:

```bash
python predict_embeddings.py --input <path_to_input_file> --model_type <siamese|gin> [options]
```

#### Required Arguments

- `--input`: Path to the input CSV/TSV file containing RNA secondary structures.
- `--model_type`: Model type (`siamese` or `gin`).

#### Optional Arguments

- `--output`: Path to save the output embeddings (default: `output/<model_id>/<model_id>_embeddings.tsv`).
- `--model_id`: Identifier for the model, used to create the default output path.
- `--structure_column_name`: Name of the column containing RNA secondary structures.
- `--structure_column_num`: Column index (0-indexed) of RNA secondary structures if no header is present.
- `--header`: Specify if the input file has a header (`True` or `False`, default: `True`).
- `--samples`: Number of random samples to process from the input file.
- `--model_path`: Path to the pre-trained model file (default: `saved_model/ResNet-Secondary.pth`).
- `--gin_layers`: Number of GIN layers (default: `1`).
- `--graph_encoding`: Graph encoding type for GIN model (`standard` or `forgi`, default: `standard`).
- `--hidden_dim`: Hidden dimension size for the model (default: `256`).
- `--output_dim`: Output embedding size for the GIN model (ignored for Siamese, default: `128`).

---

## Input Data Format

The input file must be a CSV/TSV with one column containing RNA secondary structures in dot-bracket notation. Specify the column using `--structure_column_name` or `--structure_column_num`.

### Example File

| ID  | secondary_structure         |
|-----|-----------------------------|
| 1   | ..((..))..                 |
| 2   | ..(((...)))..              |

---

## Outputs

1. **Embeddings File**: A TSV file containing RNA structures and their corresponding embedding vectors.
   - Default path: `output/<model_id>/<model_id>_embeddings.tsv`
   - Columns:
     - Original input columns.
     - `embedding_vector`: Comma-separated embedding vector.

2. **Logs**: Progress and parameters are logged to `output/<model_id>/predict_embedding.log`.

---

## Example

### Predict Embeddings Using a Siamese Model

```bash
python predict_embeddings.py --input data/rna_structures.csv --model_type siamese --structure_column_name secondary_structure
```

### Predict Embeddings Using a GIN Model

```bash
python predict_embeddings.py --input data/rna_structures.csv --model_type gin --graph_encoding forgi
```

### Process a Random Sample of 100 Structures

```bash
python predict_embeddings.py --input data/rna_structures.csv --samples 100 --model_type siamese
```

---

## How It Works

1. **Load Model**:
   - Loads the pre-trained model from `--model_path`.
   - If missing, attempts to download the default model.

2. **Input Validation**:
   - Ensures valid RNA structures using dot-bracket notation.

3. **Generate Embeddings**:
   - **Siamese Model**: Converts RNA structures to contact matrices and predicts embeddings.
   - **GIN Model**: Converts RNA structures to graphs (`standard` or `forgi`) and predicts embeddings.

4. **Save Results**:
   - Appends embedding vectors to the input data and saves as a TSV file.

---

## Notes

- Ensure the input file contains valid dot-bracket RNA structures.
- Use GPU for faster predictions with large datasets.
- Extendable: You can add support for new model types or custom preprocessing steps.
