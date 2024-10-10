# StrusiNet2: RNA Structure Embedding Generator

## Introduction
StrusiNet2 is a tool designed to generate embeddings from RNA secondary structures using a pre-trained Siamese Neural Network model. This project takes RNA sequences with their secondary structure in dot-bracket notation, processes them into contact matrices, and then feeds them into a neural network to obtain meaningful embeddings. These embeddings can be used for downstream tasks such as clustering, classification, or other forms of analysis.

## Repository Structure
The repository contains the following key components:

- **`src/model/siamese_model.py`**: Contains the Siamese neural network definition.
- **`src/model/utils.py`**: Utility functions for processing RNA data.
- **`strusinet.py`**: Main script for generating embeddings from RNA secondary structures.
- **`tsne_embedding_tool.py`**: A script to visualize generated embeddings using t-SNE.
- **`tests/test_model.py`** and **`tests/test_strusinet.py`**: Unit tests for ensuring the model and the whole embedding generation process work correctly.

## Installation
To run the StrusiNet2 project, you will need Python and the required dependencies installed.

### Step 1: Clone the Repository
```sh
git clone https://github.com/nicoaira/StrusiNet2.git
cd StrusiNet2
```

### Step 2: Install Dependencies
You can install all necessary dependencies using:

```sh
pip install -r requirements.txt
```

Dependencies include:
- `torch==1.12.0`
- `pandas==1.4.3`
- `torchvision==0.13.0`
- `numpy==1.23.1`
- `scikit-learn==1.1.1`
- `argparse==1.4.0`

Ensure that the versions match those in the `requirements.txt` file&#8203;:contentReference[oaicite:0]{index=0}.

Make sure you also have [Git LFS](https://git-lfs.github.com/) installed if you need to store large models.

### Step 3: Download Pre-trained Model
The pre-trained model file (`ResNet-Secondary.pth`) is not included in this repository due to its size. Please download it using the command below:

```sh
# Download the model from Google Drive and save it in the 'saved_model' directory
mkdir -p saved_model
wget -O saved_model/ResNet-Secondary.pth "https://drive.google.com/uc?export=download&id=1ltrAQ2OfmvrRx8cKxeNKK_oebwVRClEW"
```

## Usage
StrusiNet2 can generate embeddings from RNA sequences stored in a CSV file.

### Input Format
The input file should be a CSV containing at least one column with the RNA secondary structure in dot-bracket notation.

### Running the Embedding Generation Script
To generate embeddings from an RNA dataset:

```sh
python strusinet.py --input_csv example_data/sample_dataset.csv --output_csv example_data/sample_dataset_with_embeddings.csv
```

**Arguments**:
- `--input_csv`: Path to the input CSV file containing RNA secondary structures.
- `--output_csv`: Path to save the output CSV file with embeddings.
- `--structure_column_name`: The column name containing RNA secondary structures (default: 'secondary_structure').
- `--structure_column_num`: (Optional) Column number of RNA secondary structures (0-indexed). If both column name and number are provided, column number will be ignored.
- `--model_path`: Path to the trained model file (default: `saved_model/ResNet-Secondary.pth`).
- `--device`: Device to run the model on (`cpu` or `cuda`, default: `cpu`).
- `--header`: Specify whether the input CSV has a header row (`True` or `False`, default: `True`).

### Example Command
If your CSV doesn't have a header and the secondary structure is in the 6th column:

```sh
python strusinet.py --input_csv example_data/sample_dataset.csv --output_csv example_data/sample_dataset_with_embeddings.csv --structure_column_num 6 --header False --device cuda
```

## Running the t-SNE Embedding Tool
After generating the RNA embeddings, you can use the `tsne_embedding_tool.py` script to visualize the embeddings using t-SNE.

### Example Command
```sh
python tsne_embedding_tool.py --input example_data/sample_dataset_with_embeddings.tsv --output example_data/sample_dataset_with_tsne.tsv --embedding_column_name embedding_vector --n_components 3
```

**Arguments**:
- `--input`: Path to the input TSV file containing embeddings (e.g., `example_data/sample_dataset_with_embeddings.tsv`).
- `--output`: Path to save the output TSV file containing the t-SNE-transformed embeddings.
- `--embedding_column_name`: The name of the column containing the embedding vectors.
- `--n_components`: Number of components for t-SNE (e.g., 2 for 2D visualization, 3 for 3D visualization).

### Explanation
- The script reads the specified `embedding_column_name` from the input file and performs t-SNE transformation with the specified number of components.
- The resulting transformed embeddings are saved to the output file, which can then be used for downstream visualization and analysis.

### Interactive Visualization with Dash
We have also developed an application using Dash to visualize the t-SNE embeddings interactively. You can find the app in the following repository: [https://github.com/nicoaira/embeddings_app](https://github.com/nicoaira/embeddings_app).

## Running the Tests
You can run the tests using:

```sh
python -m unittest discover tests
```

This will run both the unit tests for the model and the integration tests for the embedding generation pipeline.

## Important Notes
- Ensure you have the correct PyTorch version installed that supports your GPU if you're using CUDA.
- If you encounter any issues with the pre-trained model, please make sure to check the Google Drive link and download it correctly.
