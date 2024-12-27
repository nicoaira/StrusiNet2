import argparse
import random
import torch
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

from src.embedding_evaluation.utils import save_model_histograms
from src.gin_rna_dataset import GINRNADataset
from src.model.gin_model_2_layers import GINModel2Layers
from src.model.gin_model_3_layers import GINModel3Layers
from src.model.gin_model_single_layer import GINModel
from src.model.siamese_model import SiameseResNetLSTM
from src.model.gin_model import GINModelGeneral
from src.triplet_rna_dataset import TripletRNADataset
from src.utils import is_valid_dot_bracket

def remove_invalid_structures(df):
    valid_structures = (
        df["structure_A"].apply(is_valid_dot_bracket) & 
        df["structure_P"].apply(is_valid_dot_bracket) & 
        df["structure_N"].apply(is_valid_dot_bracket)
    )
    return df[valid_structures]

def load_trained_model(model_path, model_type="siamese", graph_encoding="allocator", hidden_dim=256, gin_layers=1, lstm_layers=1, device='cpu'):
    if model_type == "siamese":
        model = SiameseResNetLSTM(
            input_channels=1, hidden_dim=hidden_dim, lstm_layers=lstm_layers)
    elif model_type == "gin_1":
        model = GINModel(graph_encoding=graph_encoding,
                         hidden_dim=256, output_dim=128)

    elif model_type == "gin_2":
        model = GINModel2Layers(hidden_dim=256, output_dim=128)

    elif model_type == "gin_3":
        model = GINModel3Layers(hidden_dim=256, output_dim=128)
    
    elif model_type == "gin":
        model = GINModelGeneral(hidden_dim=256, output_dim=128, graph_encoding=graph_encoding, gin_layers = gin_layers)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_name', default = "output", type=str)
    parser.add_argument('--model_type', default ="gin_1", type=str)
    parser.add_argument('--graph_encoding', type=str, choices=['allocator', 'forgi'], default='allocator',
                        help='Encoding to use for the transformation to graph. Only used in case of gin modeling')
    parser.add_argument('--val_dataset_path', default ="example_data/val_dataset.csv", type=str)
    parser.add_argument('--sample_num', type=int)
    parser.add_argument('--gin_layers', type=int)
    args = parser.parse_args()


    val_df = pd.read_csv(args.val_dataset_path)
    val_df = remove_invalid_structures(val_df)

    if args.sample_num:
        random_indices = random.sample(range(len(val_df)), args.sample_num)
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
    
    model = load_trained_model(args.model_path, args.model_type, gin_layers=args.gin_layers, graph_encoding=args.graph_encoding)
    save_model_histograms(model, val_loader, args.output_name)

    