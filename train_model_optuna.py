import argparse
import optuna
import os
import torch
from torch import optim
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from src.model.gin_model import GINModel
from src.early_stopping import EarlyStopping
from src.gin_rna_dataset import GINRNADataset
from src.model.gin_model_single_layer import GINModelSingleLayer
from src.model.siamese_model import SiameseResNetLSTM
from src.triplet_loss import TripletLoss
from src.triplet_rna_dataset import TripletRNADataset
from src.utils import is_valid_dot_bracket, log_information, log_setup
import time
from datetime import datetime

def remove_invalid_structures(df):
    valid_structures = (
        df["structure_A"].apply(is_valid_dot_bracket) & 
        df["structure_P"].apply(is_valid_dot_bracket) & 
        df["structure_N"].apply(is_valid_dot_bracket)
    )
    return df[valid_structures]

def save_model_to_local(model, optimizer, epoch, output_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/{output_name}/{output_name}_{timestamp}.pth"
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, output_path)
    print(f"Model saved to {output_path}")

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
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    for epoch in range(num_epochs):
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

        early_stopping(val_loss / len(val_loader))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return val_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from RNA secondary structures using a trained Siamese or GIN model.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input CSV/TSV file containing RNA secondary structures.')
    parser.add_argument('--optimisation_id', type=str, default='gin_optimisation', help='Model id')
    parser.add_argument('--n_trials', type=int, default='50', help='Number of trials')
    parser.add_argument('--model_type', type=str, default='gin', required=True, choices=['siamese', 'gin_1', 'gin'], help="Type of model to use: 'siamese' or 'gin'.")
    parser.add_argument('--graph_encoding', type=str, choices=['allocator', 'forgi'], default='forgi', help='Encoding to use for the transformation to graph. Only used in case of gin modeling')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training and validation.')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to train the model.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden dimension size for the model. If not provided, it will be optimized.')
    parser.add_argument('--output_dim', type=int, default=None, help='Output embedding size. If not provided, it will be optimized.')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate. If not provided, it will be optimized.')
    parser.add_argument('--gin_layers', type=int, default=None, help='Number of GIN layers. If not provided, it will be optimized.')
    args = parser.parse_args()

    # Create output folder and log setup
    output_folder = f"output/{args.optimisation_id}"
    os.makedirs(output_folder, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{output_folder}/run_{current_time}.log"
    log_setup(log_path)

    # Define the Optuna objective function
    def objective(trial):
        # Sample or fix hyperparameters based on command-line arguments
        hidden_dim = args.hidden_dim if args.hidden_dim is not None else trial.suggest_categorical('hidden_dim', [128, 256, 512])
        output_dim = args.output_dim if args.output_dim is not None else trial.suggest_categorical('output_dim', [32, 64, 128])
        lr = args.lr if args.lr is not None else trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        gin_layers = args.gin_layers if args.gin_layers is not None else trial.suggest_int('gin_layers', 1, 5)

        # Load data
        df = pd.read_csv(args.input_path)
        df = remove_invalid_structures(df)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # Instantiate model based on hyperparameters
        if args.model_type == "siamese":
            model = SiameseResNetLSTM(input_channels=1, hidden_dim=hidden_dim, lstm_layers=1)
            train_dataset = TripletRNADataset(train_df, max_len=max(df['structure_A'].str.len()))
            val_dataset = TripletRNADataset(val_df, max_len=max(df['structure_A'].str.len()))
            train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        elif args.model_type == "gin_1":
            model = GINModelSingleLayer(graph_encoding=args.graph_encoding, hidden_dim=hidden_dim, output_dim=output_dim)
            train_dataset = GINRNADataset(train_df, graph_encoding=args.graph_encoding)
            val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding)
            train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
            val_loader = GeoDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        elif args.model_type == "gin":
            model = GINModel(hidden_dim=hidden_dim, output_dim=output_dim, graph_encoding=args.graph_encoding, gin_layers=gin_layers)
            train_dataset = GINRNADataset(train_df, graph_encoding=args.graph_encoding)
            val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding)
            train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
            val_loader = GeoDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        # Set up criterion and optimizer
        criterion = TripletLoss(margin=1.0)
        optimizer = optim.Adam(model.parameters(), lr=lr)


        trial_params = {
            "trial_number": trial.number,  # Optuna's unique trial number
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "lr": lr,
            "gin_layers": gin_layers,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "patience": args.patience,
            "model_type": args.model_type,
            "graph_encoding": args.graph_encoding,
        }
        # Run training with early stopping and return the validation loss
        val_loss = train_model_with_early_stopping(
            model,
            args.optimisation_id,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            num_epochs=args.num_epochs,
            patience=args.patience,
            device="cuda" if torch.cuda.is_available() else "cpu"
        ),

        trial_params["val_loss"] = val_loss
        log_information(log_path, trial_params)

        return val_loss

    # Create and optimize the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Log the best trial
    log_information(log_path, {"Best trial": study.best_trial})

if __name__ == "__main__":
    main()
