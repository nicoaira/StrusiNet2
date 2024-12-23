import os
import torch
from torch import optim
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from tqdm import tqdm
from src.early_stopping import EarlyStopping
from src.gin_rna_dataset import GINRNADataset
from src.model.gin_model_single_layer import GINModel
from src.model.gin_model_2_layers import GINModel2Layers
from src.model.gin_model_3_layers import GINModel3Layers
from src.model.gin_model import GINModelGeneral
from src.model.siamese_model import SiameseResNetLSTM
from src.triplet_loss import TripletLoss
from src.triplet_rna_dataset import TripletRNADataset
from src.utils import is_valid_dot_bracket
import time

def remove_invalid_structures(df):
    valid_structures = (
        df["structure_A"].apply(is_valid_dot_bracket) & 
        df["structure_P"].apply(is_valid_dot_bracket) & 
        df["structure_N"].apply(is_valid_dot_bracket)
    )
    return df[valid_structures]

import torch
from datetime import datetime

def save_model_to_local(model, optimizer, epoch, output_name):
    """
    Save the model's state_dict, optimizer's state_dict, and the current epoch to local,
    including a timestamp in the file name.

    Args:
    - model: The PyTorch model to save.
    - optimizer: The optimizer used during training.
    - epoch: The current epoch to save.
    - model_save_path: Path to save the model (without file extension).
    """
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/{output_name}/{output_name}_{timestamp}.pth"

    # Append the timestamp to the file name
    file_name_with_timestamp = output_path

    # Create the checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # Save the checkpoint
    torch.save(checkpoint, file_name_with_timestamp)
    print(f"Model saved to {file_name_with_timestamp}")



def train_model_with_early_stopping(
        model,
        output_name,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs,
        patience,
        device
):
    """
    Train either a GIN model or a Siamese model with early stopping.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The DataLoader for training data.
        val_loader (DataLoader): The DataLoader for validation data.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        num_epochs (int): Number of epochs for training.
        patience (int): Early stopping patience.
        device (str): Device to use ('cuda' or 'cpu').
    """
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
        for i, batch in progress_bar:
            optimizer.zero_grad()
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": running_loss / (i + 1)})

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")
            for i, batch in progress_bar_val:
                anchor, positive, negative = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_out, positive_out, negative_out = model(anchor, positive, negative)
                loss = criterion(anchor_out, positive_out, negative_out)
                val_loss += loss.item()
                progress_bar_val.set_postfix({"Val Loss": val_loss / (i + 1)})

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")

        # Early stopping
        early_stopping(val_loss / len(val_loader))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Training complete.")

    output_folder = f"output/{output_name}"
    os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists
    save_model_to_local(model, optimizer, epoch, output_name)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate embeddings from RNA secondary structures using a trained Siamese or GIN model.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input CSV/TSV file containing RNA secondary structures.')
    parser.add_argument('--output_name', type=str, default='siamese_model', help='Output name')
    parser.add_argument('--model_type', type=str, default='siamese', required=True, choices=['siamese', 'gin_1','gin_2','gin_3', 'gin'], help="Type of model to use: 'siamese' or 'gin'.")
    parser.add_argument('--graph_encoding', type=str, choices=['allocator', 'forgi'], default='allocator', help='Encoding to use for the transformation to graph. Only used in case of gin modeling')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size for the model.')
    parser.add_argument('--output_dim', type=int, default=128, help='Output embedding size for the GIN model (ignored for siamese).')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training and validation.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (e.g., "cpu" or "cuda").')
    parser.add_argument('--gin_layers', type=int, default=1, help='Number of gin layers.')
    args = parser.parse_args()

    # Load data
    dataset_path = args.input_path
    df = pd.read_csv(dataset_path)
    df = remove_invalid_structures(df)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Instantiate model, criterion, optimizer, and dataset based on model type
    if args.model_type == "siamese":
        max_len = max(
            max(df['structure_A'].str.len()),
            max(df['structure_P'].str.len()),
            max(df['structure_N'].str.len())
        )
        model = SiameseResNetLSTM(input_channels=1, hidden_dim=args.hidden_dim, lstm_layers=1)
        train_dataset = TripletRNADataset(train_df, max_len=max_len)
        val_dataset = TripletRNADataset(val_df, max_len=max_len)
        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    elif args.model_type == "gin_1":
        model = GINModel(graph_encoding=args.graph_encoding, hidden_dim=args.hidden_dim, output_dim=args.output_dim)
        train_dataset = GINRNADataset(train_df, graph_encoding=args.graph_encoding)
        val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding)
        train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = GeoDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    elif args.model_type == "gin_2":
        model = GINModel2Layers(hidden_dim=args.hidden_dim, output_dim=args.output_dim)
        train_dataset = GINRNADataset(train_df, graph_encoding=args.graph_encoding)
        val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding)
        train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = GeoDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    elif args.model_type == "gin_3":
        model = GINModel3Layers(hidden_dim=args.hidden_dim, output_dim=args.output_dim)
        train_dataset = GINRNADataset(train_df, graph_encoding=args.graph_encoding)
        val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding)
        train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = GeoDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    elif args.model_type == "gin":
        model = GINModelGeneral(hidden_dim=args.hidden_dim, output_dim=args.output_dim, graph_encoding=args.graph_encoding, gin_layers = args.gin_layers)
        train_dataset = GINRNADataset(train_df, graph_encoding=args.graph_encoding)
        val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding)
        train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = GeoDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        

    # Set up criterion
    criterion = TripletLoss(margin=1.0)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()

    # Train the model with early stopping
    train_model_with_early_stopping(
        model,
        args.output_name,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=args.num_epochs,
        patience=args.patience,
        device=args.device
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60

    print(f"Finished. Total execution time: {execution_time_minutes:.6f} minutes")

if __name__ == "__main__":
    main()