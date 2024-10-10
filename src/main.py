import torch
import pandas as pd
import argparse
from model.siamese_model import SiameseResNetLSTM
from model.utils import pad_and_convert_to_contact_matrix

# Load the trained model
def load_trained_model(model_path, input_channels=1, hidden_dim=256, lstm_layers=1, device='cpu'):
    model = SiameseResNetLSTM(input_channels=input_channels, hidden_dim=hidden_dim, lstm_layers=lstm_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Function to get embedding from dot-bracket structure
def get_embedding(dot_bracket_structure, model, max_len=641, device='cpu'):
    # Convert dot-bracket to padded contact matrix
    contact_matrix = pad_and_convert_to_contact_matrix(dot_bracket_structure, max_len)
    contact_tensor = torch.tensor(contact_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, max_len, max_len)

    # Get embedding from the model
    with torch.no_grad():
        embedding = model.forward_once(contact_tensor)
    return embedding.squeeze().cpu().numpy()

# Main function to handle inputs and generate embeddings
def main():
    parser = argparse.ArgumentParser(description="Generate RNA embeddings from dot-bracket notations using a trained Siamese model.")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the CSV file containing RNA dot-bracket structures.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the CSV with generated embeddings.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--structure_column', type=str, default='structure', help='Column name in CSV containing dot-bracket structures.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., "cpu" or "cuda").')
    args = parser.parse_args()

    # Load the model
    model = load_trained_model(args.model_path, device=args.device)

    # Load the input CSV
    df = pd.read_csv(args.input_csv)
    
    if args.structure_column not in df.columns:
        raise ValueError(f"The specified column '{args.structure_column}' does not exist in the input CSV.")

    # Generate embeddings for each structure
    embeddings = []
    for structure in df[args.structure_column]:
        embedding = get_embedding(structure, model, device=args.device)
        embeddings.append(embedding)

    # Add embeddings to the dataframe
    embedding_columns = [f'embedding_{i+1}' for i in range(len(embeddings[0]))]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)
    result_df = pd.concat([df, embeddings_df], axis=1)

    # Save the output CSV
    result_df.to_csv(args.output_csv, index=False)
    print(f"Embeddings saved to {args.output_csv}")

if __name__ == "__main__":
    main()