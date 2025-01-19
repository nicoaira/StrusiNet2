import argparse
import random
import torch
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os

from tqdm import tqdm

from src.embedding_evaluation.utils import square_dist
from src.gin_rna_dataset import GINRNADataset
from src.model.gin_model_single_layer import GINModelSingleLayer
from src.model.gin_model import GINModel
from src.model.siamese_model import SiameseResNetLSTM
from src.triplet_rna_dataset import TripletRNADataset
from src.utils import is_valid_dot_bracket

def remove_invalid_structures(df):
    valid_structures = (
        df["structure_A"].apply(is_valid_dot_bracket) & 
        df["structure_P"].apply(is_valid_dot_bracket) & 
        df["structure_N"].apply(is_valid_dot_bracket)
    )
    return df[valid_structures]

def load_trained_model(
        model_path,
        model_type="siamese",
        graph_encoding="standard",
        hidden_dim=256,
        output_dim=128,
        gin_layers=1,
        lstm_layers=1,
        device='cpu'
    ):
    if model_type == "siamese":
        model = SiameseResNetLSTM(
            input_channels=1, hidden_dim=hidden_dim, lstm_layers=lstm_layers)

    elif model_type == "gin_1":
        model = GINModelSingleLayer(graph_encoding=graph_encoding,
                         hidden_dim=hidden_dim, output_dim=output_dim)
    
    elif model_type == "gin":
        model = GINModel(hidden_dim=hidden_dim, output_dim=output_dim, graph_encoding=graph_encoding, gin_layers = gin_layers)

    # Load the checkpoint that contains multiple states (epoch, optimizer, and model state_dict)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Load only the model's state_dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to the appropriate device (CPU or GPU)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def get_dataset_loader(model_type,val_df):
    if model_type == "siamese":
        max_len = max(
            max(val_df['structure_A'].str.len()),
            max(val_df['structure_P'].str.len()),
            max(val_df['structure_N'].str.len())
        )
        val_dataset = TripletRNADataset(val_df, max_len=max_len)
        val_loader = TorchDataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)

    elif "gin_" in model_type:
        val_dataset = GINRNADataset(val_df)
        val_loader = GeoDataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)

    return val_loader

def generate_validation_embeddings(model, validation_loader):
   
  anchor_embeddings = []
  positive_embeddings = []
  negative_embeddings = []

  with torch.no_grad():
      progress_bar_val = tqdm(enumerate(validation_loader), total=len(validation_loader))
      for _, batch in progress_bar_val:
          anchor, positive, negative = batch

          # Forward pass
          anchor_out, positive_out, negative_out = model(
              anchor, positive, negative)

          # Collect embeddings
          anchor_embeddings.append(anchor_out)
          positive_embeddings.append(positive_out)
          negative_embeddings.append(negative_out)

  # Concatenate the results into single tensors
  anchor_embeddings = torch.cat(anchor_embeddings)
  positive_embeddings = torch.cat(positive_embeddings)
  negative_embeddings = torch.cat(negative_embeddings)
  return anchor_embeddings, positive_embeddings, negative_embeddings

def save_model_histograms(model, validation_loader, model_id):
  
  def save_histogram(output_folder, anchor_embeddings, positive_embeddings, negative_embeddings, metric):
      
    if metric == 'cosine':
      anchor_positive_similarity = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1)
      anchor_negative_similarity = F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=1)
    elif metric == 'square_dist':
      anchor_positive_similarity = square_dist(anchor_embeddings, positive_embeddings)
      anchor_negative_similarity = square_dist(anchor_embeddings, negative_embeddings)
      
    # Plot the histograms
    plt.figure(figsize=(10, 6))

    # Plot for anchor-positive distances (blue)
    plt.hist(anchor_positive_similarity.numpy(), bins=30, alpha=0.5, label='Anchor-Positive', color='blue')

    # Plot for anchor-negative distances (red)
    plt.hist(anchor_negative_similarity.numpy(), bins=30, alpha=0.5, label='Anchor-Negative', color='red')

    # Add labels and title
    # Add labels and title
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Anchor-Positive and Anchor-Negative Distances ({metric.capitalize()} Metric)')
    plt.legend()

    output_path = os.path.join(output_folder, f"histogram_{metric}.png")
    plt.savefig(output_path)
    plt.close()  # Close the plot to free memory
    print(f"Saved {metric} histogram to {output_path}")
  
  output_folder = f"output/{model_id}"
  os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists
  anchor_embeddings, positive_embeddings, negative_embeddings = generate_validation_embeddings(model, validation_loader)
  save_histogram(output_folder, anchor_embeddings, positive_embeddings, negative_embeddings, 'cosine')
  save_histogram(output_folder, anchor_embeddings, positive_embeddings, negative_embeddings, 'square_dist')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_id', default = "gin_2", type=str)
    parser.add_argument('--model_type', type=str, choices=["siamese", "gin_1", "gin"])
    parser.add_argument('--graph_encoding', type=str, choices=['standard', 'forgi'], default='standard',
                        help='Encoding to use for the transformation to graph. Only used in case of gin modeling')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size for the model.')
    parser.add_argument('--output_dim', type=int, default=128, help='Output embedding size for the GIN model (ignored for siamese).')
    parser.add_argument('--val_dataset_path', default ="data/example_data/val_dataset.csv", type=str)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--gin_layers', type=int)
    args = parser.parse_args()


    val_df = pd.read_csv(args.val_dataset_path)
    val_df = remove_invalid_structures(val_df)

    if args.samples:
        random_indices = random.sample(range(len(val_df)), args.samples)
        val_df = val_df.iloc[random_indices].copy()

    if args.model_type == "siamese":
        max_len = max(
            max(val_df['structure_A'].str.len()),
            max(val_df['structure_P'].str.len()),
            max(val_df['structure_N'].str.len())
        )
        val_dataset = TripletRNADataset(val_df, max_len=max_len)
        val_loader = TorchDataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)
    else:
        val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding)
        val_loader = GeoDataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)
    
    model = load_trained_model(
        args.model_path,
        args.model_type,
        gin_layers=args.gin_layers,
        graph_encoding=args.graph_encoding,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    save_model_histograms(model, val_loader, args.model_id)

    