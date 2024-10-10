# src/tsne/compute_tsne.py

import pandas as pd
from sklearn.manifold import TSNE
import argparse

def compute_tsne(df, embedding_columns, rna_id_column, n_components=3, perplexity=30.0, random_state=42):
    """
    Compute t-SNE for given embeddings in a DataFrame and returns a DataFrame with t-SNE results.

    Args:
        df (pd.DataFrame): DataFrame containing the embeddings.
        embedding_columns (list): List of column names containing the embedding vectors.
        rna_id_column (str): The column name used to identify rows (e.g., RNA IDs).
        n_components (int): Number of dimensions for t-SNE projection (default is 3).
        perplexity (float): t-SNE perplexity parameter (default is 30.0).
        random_state (int): Random seed for reproducibility (default is 42).
    
    Returns:
        pd.DataFrame: DataFrame containing RNA IDs and t-SNE coordinates.
    """
    # Extract embeddings from the DataFrame
    embeddings = df[embedding_columns].values
    
    # Compute t-SNE projections
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(embeddings)

    # Create a new DataFrame for t-SNE results
    tsne_df = df[[rna_id_column]].copy()
    tsne_df['tSNE_1'] = tsne_results[:, 0]
    tsne_df['tSNE_2'] = tsne_results[:, 1]

    # Add third dimension if applicable
    if n_components == 3:
        tsne_df['tSNE_3'] = tsne_results[:, 2]
    
    return tsne_df

def main():
    parser = argparse.ArgumentParser(description="Compute t-SNE for RNA embeddings and save results to a CSV file.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file with embeddings.")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to save the output CSV with t-SNE coordinates.")
    parser.add_argument('--embedding_columns', type=str, nargs='+', required=True, help="Names of the columns containing the embeddings.")
    parser.add_argument('--rna_id_column', type=str, required=True, help="Name of the column to identify rows (e.g., RNA ID).")
    parser.add_argument('--n_components', type=int, default=3, help="Number of dimensions for t-SNE (default is 3).")
    parser.add_argument('--perplexity', type=float, default=30.0, help="t-SNE perplexity parameter (default is 30.0).")
    parser.add_argument('--random_state', type=int, default=42, help="Random seed for reproducibility (default is 42).")

    args = parser.parse_args()

    # Load the input CSV file
    df = pd.read_csv(args.input_csv)

    # Compute t-SNE
    tsne_df = compute_tsne(df, args.embedding_columns, args.rna_id_column, args.n_components, args.perplexity, args.random_state)

    # Save the t-SNE results to the output CSV file
    tsne_df.to_csv(args.output_csv, index=False)
    print(f"t-SNE results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
