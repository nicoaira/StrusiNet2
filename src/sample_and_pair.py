import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import argparse
import os
from multiprocessing import Pool, cpu_count

def sample_rows(input_path, n=None, f=None):
    df = pd.read_csv(input_path, sep='\t')
    if n is not None:
        sampled_df = df.sample(n=n, random_state=42).reset_index(drop=True)
    elif f is not None:
        sampled_df = df.sample(frac=f, random_state=42).reset_index(drop=True)
    else:
        raise ValueError("Either 'n' or 'f' must be provided.")
    return sampled_df

def calculate_distance_batch(args):
    batch, embeddings_tensor, metric = args
    results = []
    for i, j in batch:
        if metric == 'cosine':
            distance = 1 - torch.nn.functional.cosine_similarity(embeddings_tensor[i], embeddings_tensor[j], dim=0).item()
        else:  # squared distance
            distance = torch.sum((embeddings_tensor[i] - embeddings_tensor[j]) ** 2).item()
        results.append((i, j, distance))
    return results

def calculate_distances(embeddings, metric='squared', num_workers=1, batch_size=1000):
    embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
    num_embeddings = embeddings_tensor.shape[0]
    
    total_pairs = num_embeddings * (num_embeddings - 1) // 2
    pairs = [(i, j) for i in range(num_embeddings) for j in range(i + 1, num_embeddings)]
    
    # Split pairs into batches
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    args_list = [(batch, embeddings_tensor, metric) for batch in batches]
    
    distances = []
    with Pool(num_workers) as pool:
        with tqdm(total=total_pairs, desc="Calculating distances") as pbar:
            for result in pool.imap_unordered(calculate_distance_batch, args_list):
                distances.extend(result)
                pbar.update(len(result))
    
    return distances

def generate_pairs(sampled_df, distances):
    pairs = []
    for i, j, distance in distances:
        row_i = sampled_df.iloc[i]
        row_j = sampled_df.iloc[j]
        pair = {'distance': distance}
        for col in sampled_df.columns:
            pair[f'{col}_1'] = row_i[col]
            pair[f'{col}_2'] = row_j[col]
        pairs.append(pair)
    return pd.DataFrame(pairs)

def main(input_path, n, f, metric, num_workers, batch_size):
    if n is not None and f is not None:
        raise ValueError("Both 'n' and 'f' cannot be provided at the same time.")
    
    # Sample rows
    sampled_df = sample_rows(input_path, n, f)

    # Extract embedding vectors
    embeddings = sampled_df['embedding_vector'].apply(lambda x: np.array([float(i) for i in x.split(',')])).tolist()

    # Calculate distances
    distances = calculate_distances(embeddings, metric, num_workers, batch_size)

    # Generate pairs
    pairs_df = generate_pairs(sampled_df, distances)

    # Output file
    output_path = os.path.splitext(input_path)[0] + '_pairs.tsv'
    pairs_df.to_csv(output_path, sep='\t', index=False)
    print(f"Pairs saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample rows from a TSV file, generate pairwise combinations, and calculate distances or similarities between embedding vectors.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input TSV file.')
    parser.add_argument('--n', type=int, help='Number of rows to sample.')
    parser.add_argument('--f', type=float, help='Fraction of rows to sample.')
    parser.add_argument('--metric', type=str, choices=['squared', 'cosine'], default='squared', help='Distance metric to use (default: squared).')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help='Number of worker processes to use for multiprocessing (default: number of CPU cores).')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for distance calculations (default: 1000).')
    args = parser.parse_args()

    main(args.input, args.n, args.f, args.metric, args.num_workers, args.batch_size)
