import argparse
import os
import torch
import numpy as np
import pandas as pd
import torch
import numpy as np
import plotly.express as px
import plotly.io as pio
import random
from sklearn.manifold import TSNE

def str_to_tensor(embedding_str):
    embedding_list = embedding_str.split(',')

    # Convert the list of strings to a NumPy array of floats
    embedding_array = np.array([float(x) for x in embedding_list])

    #Convert the NumPy array back to a PyTorch tensor
    return torch.tensor(embedding_array)

def project_embeddings(df):
    embeddings = [str_to_tensor(e) for e in df['embedding_vector']]

    # Ensure the embeddings are 2D (i.e., shape is [num_samples, 256])
    embeddings = torch.stack(embeddings).numpy()

    # Check if the embeddings are already flattened to (num_samples, 256)
    if embeddings.ndim == 3:
        embeddings = embeddings.squeeze(1)  # Remove the singleton dimension if necessary

    # Extract RNA class labels
    rfam = df['rfam'].values

    ### 1. t-SNE ###
    # Perform t-SNE for dimensionality reduction to 2D
    tsne = TSNE(n_components=3, random_state=42)
    embedding_tsne = tsne.fit_transform(embeddings)

    return embedding_tsne

def save_df_with_tsne_embedding(df, embedding_tsne, output_path):
    df['tSNE_1'] = embedding_tsne[:, 0]
    df['tSNE_2'] = embedding_tsne[:, 1]
    df['tSNE_3'] = embedding_tsne[:, 2]
    df.to_csv(output_path)

def save_scatter_2d(output_folder, df, embedding_tsne, column = 'rfam'):
    column_values = df[column].values

    df_tsne = pd.DataFrame({
        'tSNE_1': embedding_tsne[:, 0],
        'tSNE_2': embedding_tsne[:, 1],
        # 'tSNE_3': embedding_tsne[:, 3],
        column: column_values
    })

    # Plot the t-SNE projection using Plotly
    fig_tsne12 = px.scatter(
        df_tsne,
        x='tSNE_1',
        y='tSNE_2',
        color=column,
        labels={'color': 'RNA Class'},
        title='t-SNE projection of RNA embeddings',
        hover_data=[column]
    )

    # Define file paths
    html_output_path = f"{output_folder}/scatter_tsne_{column}.html"
    png_output_path = f"{output_folder}/scatter_tsne_{column}.png"

    # Save as an interactive HTML file
    fig_tsne12.write_html(html_output_path)
    print(f"Scatter plot saved as HTML to {html_output_path}")

    # Save as a static PNG image
    pio.write_image(fig_tsne12, png_output_path, format='png', width=800, height=600)
    print(f"Scatter plot saved as PNG to {png_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--predict_embedding_path', type=str, required=True)
    parser.add_argument('--output_name', default = "output", type=str)
    parser.add_argument('--sample_num', default = 10000, type=int)
    args = parser.parse_args()


    df = pd.read_csv(args.predict_embedding_path, sep="\t")

    random_indices = random.sample(range(len(df)), args.sample_num)
    df_random_sample = df.iloc[random_indices].copy()

    embedding_tsne = project_embeddings(df_random_sample)

    output_folder = f"output/{args.output_name}"
    os.makedirs(output_folder, exist_ok=True)
    save_scatter_2d(output_folder, df_random_sample, embedding_tsne, column = 'rfam')
    save_scatter_2d(output_folder, df_random_sample, embedding_tsne, column = 'rna_type')

    df_random_sample['tSNE_1'] = embedding_tsne[:, 0]
    df_random_sample['tSNE_2'] = embedding_tsne[:, 1]
    df_random_sample['tSNE_3'] = embedding_tsne[:, 2]
    
    projected_embeddings_path = f"{output_folder}/projected_embeddings.csv"
    df_random_sample.to_csv(projected_embeddings_path)
    print(f"Saved projections to {projected_embeddings_path}")
    