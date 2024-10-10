
# StrusiNet2

StrusiNet2 is a neural network-based tool for RNA secondary structure analysis. It allows users to generate embedding vectors from RNA sequences given in dot-bracket notation and perform t-SNE visualization for clustering. This package includes both the model-based embedding generator and a tool for t-SNE analysis.

## Features
- Generate embedding vectors for RNA structures in dot-bracket notation.
- Compute t-SNE projections from embedding vectors for visualizing relationships between different RNA sequences.

## Installation

To use StrusiNet2, you need to have Docker installed. Follow the instructions below to build the Docker container and run the tools.

### Clone the Repository
First, clone this repository to your local machine:
```sh
$ git clone https://github.com/your_username/StrusiNet2.git
$ cd StrusiNet2
```

### Build Docker Container
Use the provided `Dockerfile` to build the container:
```sh
$ docker build -t strusinet2 .
```

### Run the Container
To run the container, use the following command:
```sh
$ docker run --rm -v $(pwd):/app strusinet2 --dot_bracket "((..))((..))" --model_path saved_model/siamese_trained_model.pth
```

- **`--dot_bracket`**: Provide the RNA structure in dot-bracket notation.
- **`--model_path`**: Path to the saved model file.
- **`--max_len`**: (Optional) The maximum length for padding the contact matrix. Defaults to 600.
- **`--device`**: (Optional) Choose the device (`cpu` or `cuda`). Defaults to `cpu`.

### Running t-SNE Tool
StrusiNet2 also provides a tool to perform t-SNE analysis on precomputed embedding vectors. You can run this tool using the following command inside the container:

```sh
$ docker run --rm -v $(pwd):/app strusinet2 python src/tsne/compute_tsne.py --input_csv example_data/sample_embeddings.csv --output_csv output_tsne.csv --embedding_columns embedding_1 embedding_2 embedding_3 --rna_id_column rna_id
```

- **`--input_csv`**: Path to the input CSV file containing embeddings.
- **`--output_csv`**: Path to save the output CSV file with t-SNE coordinates.
- **`--embedding_columns`**: Names of the columns containing the embedding vectors.
- **`--rna_id_column`**: Column to identify rows (e.g., RNA ID).

### Example Data
Example RNA sequences and precomputed embeddings are provided in the `example_data/` directory for testing purposes.

## Development and Testing
This package includes unit tests located in the `tests/` directory. To run the tests, you can execute the following command inside the Docker container:

```sh
$ docker run --rm strusinet2 python -m unittest discover tests
```
