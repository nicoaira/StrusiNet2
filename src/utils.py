import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
import forgi.graph.bulge_graph as fgb

def is_valid_dot_bracket(structure):
    """
    Check if a dot-bracket RNA structure has matching open and closing parentheses.
    """
    stack = []
    for char in structure:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    
    # If stack is empty, all parentheses are matched
    return len(stack) == 0

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


def graph_to_tensor(g):
    x = torch.Tensor([[0] if g.nodes[node]['label'] == 'unpaired' else [1] for node in g.nodes])
    edge_index = torch.LongTensor(list(g.edges())).t().contiguous()

    # Graph to Data object
    data = Data(x=x, edge_index=edge_index)

    return data

def dotbracket_to_forgi_graph(dotbracket):
    bulge_graph = fgb.BulgeGraph.from_dotbracket(dotbracket)

    g = nx.Graph()
    
    nodes = list(bulge_graph.edges.keys())

    for i,n in enumerate(nodes):
        g.add_node(i, structure = n[0], length = bulge_graph.element_length(n))
        for c in bulge_graph.edges[n]:
            g.add_edge(i,nodes.index(c))
    return g

def forgi_graph_to_tensor(g):
    element_type_map = {
        'f': [1, 0, 0, 0, 0, 0],
        't': [0, 1, 0, 0, 0, 0],
        's': [0, 0, 1, 0, 0, 0],
        'i': [0, 0, 0, 1, 0, 0],
        'm': [0, 0, 0, 0, 1, 0],
        'h': [0, 0, 0, 0, 0, 1]
    }
    x = torch.Tensor([element_type_map.get(g.nodes[node]["structure"]) + [g.nodes[node]["length"]] for node in g.nodes])
    edge_index = torch.LongTensor(list(g.edges())).t().contiguous()

    # Graph to Data object
    data = Data(x=x, edge_index=edge_index)

    return data