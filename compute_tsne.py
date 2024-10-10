import argparse
import pandas as pd
import numpy as np  # Import NumPy to use arrays
from sklearn.manifold import TSNE

# Function to compute tSNE embeddings and save them to a TSV
def compute_tsne(input_file, output_file, embedding_column_name='embedding_vector', n_components=3, random_state=42):
    # Determine delimiter based on file extension
    delimiter = '\t' if input_file.endswith('.tsv') else ','

    # Load the input TSV/CSV
    df = pd.read_csv(input_file, delimiter=delimiter)

    # Ensure that the embedding column exists
    if embedding_column_name not in df.columns:
        raise ValueError(f"Column '{embedding_column_name}' not found in input file.")

    # Extract the embeddings as lists of floats
    embeddings = df[embedding_column_name].apply(lambda x: list(map(float, x.split(',')))).tolist()

    # Convert the list of embeddings to a NumPy array
    embeddings = np.array(embeddings)

    # Compute tSNE embeddings
    tsne = TSNE(n_components=n_components, random_state=random_state)
    tsne_result = tsne.fit_transform(embeddings)

    # Add tSNE dimensions to the dataframe
    for i in range(n_components):
        df[f'tSNE_{i+1}'] = tsne_result[:, i]

    # Save the output TSV
    df.to_csv(output_file, sep='\t', index=False)
    print(f"tSNE results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute tSNE from embedding vectors and add the result to a TSV or CSV.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input TSV/CSV file containing embedding vectors.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output TSV file with tSNE dimensions.')
    parser.add_argument('--embedding_column_name', type=str, default='embedding_vector', help='Name of the column containing embedding vectors (default: embedding_vector).')
    parser.add_argument('--n_components', type=int, default=3, help='Number of tSNE components (default: 3).')
    args = parser.parse_args()

    # Compute tSNE and save to output
    compute_tsne(
        input_file=args.input,
        output_file=args.output,
        embedding_column_name=args.embedding_column_name,
        n_components=args.n_components
    )
