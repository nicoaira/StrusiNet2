import argparse
import pandas as pd
from sklearn.manifold import TSNE

# Function to compute tSNE embeddings and save them to a CSV
def compute_tsne(input_csv, output_csv, embedding_column_name='embedding_vector', n_components=3, random_state=42):
    # Load the input CSV
    df = pd.read_csv(input_csv)

    # Ensure that the embedding column exists
    if embedding_column_name not in df.columns:
        raise ValueError(f"Column '{embedding_column_name}' not found in input CSV.")

    # Extract the embeddings
    embeddings = df[embedding_column_name].apply(lambda x: eval(x)).tolist()

    # Compute tSNE embeddings
    tsne = TSNE(n_components=n_components, random_state=random_state)
    tsne_result = tsne.fit_transform(embeddings)

    # Add tSNE dimensions to the dataframe
    for i in range(n_components):
        df[f'tSNE_{i+1}'] = tsne_result[:, i]

    # Save the output CSV
    df.to_csv(output_csv, index=False)
    print(f"tSNE results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute tSNE from embedding vectors and add the result to a CSV.")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing embedding vectors.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output CSV file with tSNE dimensions.')
    parser.add_argument('--embedding_column_name', type=str, default='embedding_vector', help='Name of the column containing embedding vectors (default: embedding_vector).')
    parser.add_argument('--n_components', type=int, default=3, help='Number of tSNE components (default: 3).')
    args = parser.parse_args()

    # Compute tSNE and save to output
    compute_tsne(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        embedding_column_name=args.embedding_column_name,
        n_components=args.n_components
    )