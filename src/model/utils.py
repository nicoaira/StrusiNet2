import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import Counter

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
    look_for = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>'), ('A', 'a')]
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