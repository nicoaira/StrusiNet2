import optuna
import os
import torch
from torch import optim
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from src.early_stopping import EarlyStopping
from src.gin_rna_dataset import GINRNADataset
from src.model.gin_model_single_layer import GINModel
from src.model.gin_model_2_layers import GINModel2Layers
from src.model.gin_model_3_layers import GINModel3Layers
from src.model.siamese_model import SiameseResNetLSTM
from src.triplet_loss import TripletLoss
from src.triplet_rna_dataset import TripletRNADataset
from src.utils import is_valid_dot_bracket
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

    save_model_to_local(model, optimizer, epoch, output_name)
    return val_loss / len(val_loader)

def objective(trial):
    # Sample hyperparameters from the search space
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_epochs = trial.suggest_int('num_epochs', 5, 15)
    patience = trial.suggest_int('patience', 3, 7)

    # Load data
    dataset_path = 'data/rna_structures.csv'  # Replace with the actual path
    df = pd.read_csv(dataset_path)
    df = remove_invalid_structures(df)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Choose the model type
    model_type = 'siamese'  # or 'gin_1', 'gin_2', 'gin_3'
    
    # Instantiate model based on hyperparameters
    if model_type == "siamese":
        model = SiameseResNetLSTM(input_channels=1, hidden_dim=hidden_dim, lstm_layers=1)
        train_dataset = TripletRNADataset(train_df, max_len=max(df['structure_A'].str.len()))
        val_dataset = TripletRNADataset(val_df, max_len=max(df['structure_A'].str.len()))
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        model = GINModel(graph_encoding='allocator', hidden_dim=hidden_dim, output_dim=128)
        train_dataset = GINRNADataset(train_df, graph_encoding='allocator')
        val_dataset = GINRNADataset(val_df, graph_encoding='allocator')
        train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = GeoDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Set up criterion and optimizer
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Run training with early stopping and return the validation loss
    val_loss = train_model_with_early_stopping(
        model,
        'optuna_model',
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=num_epochs,
        patience=patience,
        device='cpu'
    )
    return val_loss

def main():
    # Create an Optuna study to optimize hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # Run 50 trials

    # Print the best trial
    print("Best trial:", study.best_trial)

if __name__ == "__main__":
    main()
