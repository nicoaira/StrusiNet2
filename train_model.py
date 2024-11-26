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
from src.model.siamese_model import SiameseResNetLSTM
from src.triplet_loss import TripletLoss
from src.triplet_rna_dataset import TripletRNADataset
from src.utils import is_valid_dot_bracket

def remove_invalid_structures(df):
    valid_structures = (
        df["structure_A"].apply(is_valid_dot_bracket) & 
        df["structure_P"].apply(is_valid_dot_bracket) & 
        df["structure_N"].apply(is_valid_dot_bracket)
    )
    return df[valid_structures]

def save_model_to_local(model, optimizer, epoch, model_save_path):
    """
    Save the model's state_dict, optimizer's state_dict, and the current epoch to local.

    Args:
    - model: The PyTorch model to save.
    - optimizer: The optimizer used during training.
    - epoch: The current epoch to save.
    - path: Path to save the model
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, model_save_path)
    print(f"Model saved to {model_save_path}")


def train_model_with_early_stopping(
        model,
        model_save_path,
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

    save_model_to_local(model, optimizer, epoch, model_save_path)




def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate embeddings from RNA secondary structures using a trained Siamese or GIN model.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input CSV/TSV file containing RNA secondary structures.')
    parser.add_argument('--output_path', type=str, default='saved_model/trained_model.pth', help='Output path of the trained model.')
    parser.add_argument('--model_type', type=str, default='siamese', required=True, choices=['siamese', 'gin'], help="Type of model to use: 'siamese' or 'gin'.")
    parser.add_argument('--graph_encoding', type=str, choices=['allocator', 'forgi'], default='allocator', help='Encoding to use for the transformation to graph. Only used in case of gin modeling')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension size for the model.')
    parser.add_argument('--output_dim', type=int, default=128, help='Output embedding size for the GIN model (ignored for siamese).')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation.')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train the model.')
    parser.add_argument('--patience', type=int, default=1, help='Patience for early stopping.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (e.g., "cpu" or "cuda").')
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

    elif args.model_type == "gin":
        model = GINModel(graph_encoding=args.graph_encoding, hidden_dim=args.hidden_dim, output_dim=args.output_dim)
        train_dataset = GINRNADataset(train_df, graph_encoding=args.graph_encoding)
        val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding)
        train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = GeoDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Set up criterion
    criterion = TripletLoss(margin=1.0)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model with early stopping
    train_model_with_early_stopping(
        model,
        args.output_path,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=args.num_epochs,
        patience=args.patience,
        device=args.device
    )

if __name__ == "__main__":
    main()