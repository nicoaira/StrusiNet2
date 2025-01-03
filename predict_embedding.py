#!/usr/bin/env python3

import random
import time
import torch
import pandas as pd
from tqdm import tqdm
import argparse
from src.model.gin_model import GINModel
from src.model.siamese_model import SiameseResNetLSTM
from src.model.gin_model_single_layer import GINModelSingleLayer
from src.utils import dotbracket_to_forgi_graph, forgi_graph_to_tensor, log_information, log_setup, pad_and_convert_to_contact_matrix, dotbracket_to_graph, graph_to_tensor
import os
import subprocess
from pathlib import Path

# Load the trained model


def load_trained_model(
        model_path,
        model_type="siamese",
        graph_encoding="allocator",
        hidden_dim=256,
        output_dim=128,
        lstm_layers=1,
        device='cpu',
        gin_layers = 1
):
    # Check if the model file exists, if not provide instruction to download it
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Attempting to download...")
        model_url = "https://drive.google.com/uc?export=download&id=1ltrAQ2OfmvrRx8cKxeNKK_oebwVRClEW"
        download_command = f"wget -O {model_path} \"{model_url}\""
        try:
            subprocess.run(download_command, shell=True, check=True)
            print(f"Model downloaded successfully and saved at {model_path}")
        except subprocess.CalledProcessError:
            raise FileNotFoundError(
                f"Failed to download the model file. Please download it manually from {model_url} and place it in the 'saved_model/' directory."
            )

    # Instantiate the model
    if model_type == "siamese":
        model = SiameseResNetLSTM(
            input_channels=1, hidden_dim=hidden_dim, lstm_layers=lstm_layers)
    elif model_type == "gin_1":
        model = GINModelSingleLayer(graph_encoding=graph_encoding,
                         hidden_dim=hidden_dim, output_dim=output_dim)
    
    elif model_type == "gin":
        model = GINModel(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            graph_encoding=graph_encoding,
            gin_layers = gin_layers
        )

    # Load the checkpoint that contains multiple states (epoch, optimizer, and model state_dict)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Load only the model's state_dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to the appropriate device (CPU or GPU)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to get embedding from contact matrix


def get_siamese_embedding(model, structure, max_len, device='cpu'):
    contact_matrix = pad_and_convert_to_contact_matrix(structure, max_len)
    contact_tensor = torch.tensor(contact_matrix, dtype=torch.float32).unsqueeze(
        0).unsqueeze(0).to(device)  # Shape: (1, 1, max_len, max_len)

    with torch.no_grad():
        embedding = model.forward_once(contact_tensor)
    return ','.join(f'{x:.6f}' for x in embedding.cpu().numpy().flatten())

# Function to get embedding from graph


def get_gin_embedding(model, graph_encoding, structure, device):
    if graph_encoding == "allocator":
        graph = dotbracket_to_graph(structure)
        tg = graph_to_tensor(graph)
    elif graph_encoding == "forgi":
        graph = dotbracket_to_forgi_graph(structure)
        tg = forgi_graph_to_tensor(graph)

    tg.to(device)
    model.eval()
    with torch.no_grad():
        embedding = model.forward_once(tg)
    return ','.join(f'{x:.6f}' for x in embedding.cpu().numpy().flatten())


# Function to validate dot-bracket structure
def validate_structure(structure):
    if not isinstance(structure, str):
        raise ValueError(
            "The secondary structure must be a string containing valid characters for dot-bracket notation.")
    valid_characters = "()[]{}<>AaBbCcDd."
    if not all(char in valid_characters for char in structure):
        raise ValueError(f"Invalid characters found in the column used for secondary structure: '{structure}'. Valid characters are: {valid_characters}")

# Main function to generate embeddings from CSV or TSV


def generate_embeddings(
        input_df,
        output_path,
        model_type,
        model_path,
        log_path,
        structure_column,
        max_len=641,
        device='cpu',
        graph_encoding='allocator',
        gin_layers=1,
        hidden_dim=256,
        output_dim=128,
):
    # Load the trained model
    model = load_trained_model(
        model_path,
        model_type,
        graph_encoding,
        device=device,
        gin_layers= gin_layers,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )


    # Initialize list for storing embeddings
    embeddings = []

    # Iterate over rows and calculate embeddings using tqdm for progress bar
    with torch.no_grad():
        for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Embeddings"):
            structure = row[structure_column]
            # Validate the dot-bracket structure
            validate_structure(structure)
            if model_type == "siamese":
                embedding = get_siamese_embedding(
                    model, structure, max_len, device=device)
            elif "gin" in model_type:
                embedding = get_gin_embedding(model, graph_encoding, structure, device)

            # Convert list to comma-separated string
            embeddings.append(embedding)

    # Add the embeddings to the DataFrame
    input_df['embedding_vector'] = embeddings

    # Save the output TSV
    input_df.to_csv(output_path, sep='\t', index=False)
    print(f"Embeddings saved to {output_path}")
    save_log = {
        "Embeddings saved path": output_path
    }
    log_information(log_path, save_log)

def read_input_data(input, samples, structure_column_num, header):
    delimiter = '\t' if input.endswith('.tsv') else ','

    # Load the input CSV based on whether there is a header or not
    if header:
        df = pd.read_csv(input, delimiter=delimiter)
    else:
        if structure_column_num is None:
            raise ValueError(
                "When header is False, structure_column_num must be specified.")
        df = pd.read_csv(input, delimiter=delimiter, header=None)
        
    if samples:
        random_indices = random.sample(range(len(df)), samples)
        df = df.iloc[random_indices].copy()
    return df

def get_structure_column_name(input_df, header,structure_column_name, structure_column_num):
    if header:
        if structure_column_name:
            structure_column = structure_column_name
        elif args.structure_column_num is not None and not structure_column_name:
            structure_column = input_df.columns[structure_column_num]
        else:
            # default value = secondary_structure
            structure_column = "secondary_structure"
    return structure_column

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings from RNA secondary structures using a trained Siamese model.")
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input CSV/TSV file containing RNA secondary structures.')
    parser.add_argument('--samples', type=int)
    
    parser.add_argument('--output', type=str, help='Output path of the embedding')
    parser.add_argument('--model_id', type=str, help='If output path not defined, store in output/{model_id}/{model_id}_embedding.tsv')
    
    parser.add_argument('--structure_column_name', type=str,
                        help='Name of the column with the RNA secondary structures.')
    parser.add_argument('--structure_column_num', type=int,
                        help='Column number of the RNA secondary structures (0-indexed). If both column name and number are provided, column number will be ignored.')

    # Allows the default model_path to be dynamic
    script_directory = Path(__file__).resolve().parent
    default_model_path = script_directory / 'saved_model' / 'ResNet-Secondary.pth'
    parser.add_argument('--model_path', type=str, default=str(default_model_path),
                        help=f'Path to the trained model file (default: {default_model_path}).')

    parser.add_argument('--model_type', type=str, choices=['siamese', 'gin_1', 'gin'], default='siamese', help='Model type to run (e.g., "siamese" or "gin").')

    parser.add_argument('--gin_layers', type=int, default=1, help='Number of gin layers.')

    parser.add_argument('--graph_encoding', type=str, choices=['allocator', 'forgi'], default='allocator',
                        help='Encoding to use for the transformation to graph. Only used in case of gin modeling')

    parser.add_argument('--header', type=str, default='True',
                        help='Specify whether the input CSV file has a header (default: True). Use "True" or "False".')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size for the model.')
    parser.add_argument('--output_dim', type=int, default=128, help='Output embedding size for the GIN model (ignored for siamese).')
    args = parser.parse_args()

    # Validate the header argument
    if args.header.lower() not in ['true', 'false']:
        raise ValueError(
            "Invalid value for --header. Please use 'True' or 'False'.")
    args.header = args.header.lower() == 'true'

    if args.output:
        output_path = args.output
    elif args.model_id:
        output_path = f"output/{args.model_id}/{args.model_id}_embeddings.tsv"
    else:
        raise "Either output path or output name must be defined"
    
    input_df = read_input_data(args.input, args.samples, args.structure_column_num, args.header)

    # Determine which column to use for structure
    structure_column = get_structure_column_name(input_df, args.header, args.structure_column_name, args.structure_column_num)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_folder = os.path.dirname(output_path) 
    os.makedirs(output_folder, exist_ok=True)

    log_path = f"{output_folder}/predict_embedding.log"
    log_setup(log_path)

    predict_params = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "device": device,
        "test_data_path": args.input,
        "samples_test_data": input_df.shape[0]
    }
    if args.model_type == "gin":
        predict_params["gin_layers"] = args.gin_layers
        predict_params["graph_encoding"] = args.graph_encoding

    log_information(log_path, predict_params, "Predict params")
    
    start_time = time.time()
    # Generate embeddings
    generate_embeddings(
        input_df,
        output_path,
        args.model_type,
        args.model_path,
        log_path,
        structure_column,
        device=device,
        graph_encoding=args.graph_encoding,
        gin_layers=args.gin_layers,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60

    print(f"Finished. Total execution time: {execution_time_minutes:.6f} minutes")
    execution_time = {
        "Total execution time" : f"{execution_time_minutes:.6f} minutes"
    }
    log_information(log_path, execution_time, "Execution time")
