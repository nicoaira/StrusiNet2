import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import Counter
import networkx as nx
from torch_geometric.utils import from_networkx

# Padding Functions

def center_pad_matrix(db_structure, max_len, padding_value='.'): 
    """
    Takes the dot bracket structure, and pads it with dots (single-stranded bases),
    centering the structure within the padded region.
    """
    structure_len = len(db_structure)

    if structure_len < max_len:
        to_pad = max_len - structure_len
        left_pad = to_pad // 2
        right_pad = to_pad - left_pad  # Ensures proper padding for odd differences

        # Pad the structure with the specified padding value ('.' for unpaired bases)
        padded_structure = [padding_value] * left_pad + list(db_structure) + [padding_value] * right_pad
    else:
        padded_structure = list(db_structure)  # No padding if already max_len

    return ''.join(padded_structure)

# Padding and Converting Function

def pad_and_convert_to_contact_matrix(db_structure, max_len, padding_value='.'):
    """
    Pads the dot-bracket structure to the max_len and converts it into a contact matrix.
    Handles multiple types of base pairs, including pseudoknots.

    Args:
    - db_structure (str): Dot-bracket structure with pseudoknots.
    - max_len (int): The length to which the structure should be padded.
    - padding_value (str): The character used to pad the structure (default: '.').

    Returns:
    - contact_matrix (np.ndarray): A binary contact matrix (size max_len x max_len).
    """
    # Step 1: Center pad the structure to the desired length
    padded_structure = center_pad_matrix(db_structure, max_len, padding_value)

    # Step 2: Create a contact matrix (max_len x max_len)
    contact_matrix = np.zeros((max_len, max_len), dtype=int)

    # Dictionary to hold the stack for each pairing symbol
    look_for = [('(', ')'), ('[', ']'), ('{', '}'),
                 ('<', '>'), ('A', 'a'), ('B', 'b'),
                 ('C', 'c'), ('D', 'd')]
    
    stacks = {pair[0]: [] for pair in look_for}

    # Step 3: Fill the contact matrix by iterating over the padded structure
    for i, char in enumerate(padded_structure):
        # Check if the character is an opening bracket in the pairing symbols
        for open_symbol, close_symbol in look_for:
            if char == open_symbol:
                stacks[open_symbol].append(i)  # Push the index onto the corresponding stack
            elif char == close_symbol and stacks[open_symbol]:
                j = stacks[open_symbol].pop()  # Pop the last opening bracket index
                # Fill in the contact matrix symmetrically
                contact_matrix[i, j] = 1
                contact_matrix[j, i] = 1

    return contact_matrix

def dot_bracket_to_contact_matrix(dot_bracket):
    """
    Converts dot-bracket notation into a contact matrix representation where
    1 represents base pairing between positions and 0 represents no interaction.
    """
    n = len(dot_bracket)
    contact_matrix = np.zeros((n, n), dtype=int)
    stack = []

    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            contact_matrix[i, j] = 1
            contact_matrix[j, i] = 1  # Ensure symmetry

    return contact_matrix

# Similarity Functions

def cos_similarity(emb_1, emb_2):
    """
    Compute cosine similarity between two embeddings.
    """
    cos_similarity = F.cosine_similarity(emb_1, emb_2, dim=1)
    return cos_similarity.item()

def square_dist(emb1, emb2):
    """
    Compute the squared distance between two embeddings.
    """
    return torch.sum((emb1 - emb2) ** 2).item()

def euclidean_dist(emb1, emb2):
    """
    Compute the Euclidean distance between two embedding vectors.
    """
    return torch.norm(emb1 - emb2).item()

def dotbracket_to_graph(dotbracket):
    G = nx.Graph()
    bases = []

    # Add nodes and edges based on dot-bracket structure
    # TODO: Node information is redundant, should it be removed?
    for i, c in enumerate(dotbracket):
        if c == '(':
            bases.append(i)
            G.add_node(i, label='unpaired')
        elif c == ')':
            if bases:
                neighbor = bases.pop()
                G.add_edge(i, neighbor, edge_type='base_pair')
                G.nodes[i]['label'] = 'paired'
                G.nodes[neighbor]['label'] = 'paired'
            else:
                print("Mismatched parentheses in input!")
                return None
        elif c == '.':
            G.add_node(i, label='unpaired')
        else:
            print("Input is not in dot-bracket notation!")
            return None

        # Adding sequential (adjacent) edges
        if i > 0:
            G.add_edge(i, i - 1, edge_type='adjacent')
    
    return G


def graph_to_tensor(G):
    # Convert node labels to feature vectors
    for node in G.nodes:
        # Example encoding: paired = [1, 0], unpaired = [0, 1]
        if G.nodes[node]['label'] == 'paired':
            G.nodes[node]['x'] = [1, 0]
        else:
            G.nodes[node]['x'] = [0, 1]
    
    # Convert edge types to a numerical format if needed (optional)
    for u, v, attr in G.edges(data=True):
        if attr['edge_type'] == 'base_pair':
            G.edges[u, v]['edge_attr'] = [1, 0]
        else:  # adjacent
            G.edges[u, v]['edge_attr'] = [0, 1]

    # Convert NetworkX graph to PyTorch Geometric Data object
    data = from_networkx(G)

    # Set node features
    # TODO: Node information is redundant, should it be removed?
    data.x = torch.tensor([G.nodes[node]['x'] for node in G.nodes], dtype=torch.float)

    # If you need edge attributes
    if 'edge_attr' in data:
        data.edge_attr = torch.tensor([G.edges[u, v]['edge_attr'] for u, v in G.edges], dtype=torch.float)

    return data