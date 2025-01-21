import numpy as np
import forgi
from ViennaRNA import fold
import math
import igraph as ig
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import tempfile
import random
import matplotlib.pyplot as plt
import forgi.visual.mplotlib as fvm
import os
import logging

DEBUG_MODE = True


def validate_structure(structure, operation, **kwargs):
    """
    Validates the dot-bracket notation of an RNA secondary structure.

    Args:
        structure (str): The RNA structure in dot-bracket notation.
        operation (str): The name of the operation being validated.
        **kwargs: Additional debugging information.

    Returns:
        bool: True if the structure is valid, False otherwise.
    """
    if DEBUG_MODE:
        # Log additional debugging information
        for key, value in kwargs.items():
            logging.debug(f"[DEBUG] {key}: {value}")

    stack = []
    for index, char in enumerate(structure):
        if char == '(':  # Open bracket, push to stack
            stack.append(char)
        elif char == ')':  # Close bracket, check stack
            if not stack or stack[-1] != '(':
                if DEBUG_MODE:
                    logging.debug(f"[DEBUG] Mismatched parenthesis at position {index}")
                return False
            stack.pop()
        elif char != '.':  # Invalid character in structure
            if DEBUG_MODE:
                logging.debug(f"[DEBUG] Invalid character '{char}' at position {index}")
            return False

    # Ensure no unmatched opening brackets remain
    if stack:
        if DEBUG_MODE:
            logging.debug(f"[DEBUG] Unmatched opening brackets after {operation}: {structure}")
        return False

    return True


def update_nodes_dict(dot_bracket_mapper, nodes_dict):
    """
    Updates the nodes dictionary to map the range of each structural element.

    Args:
        dot_bracket_mapper (list): Mapping of structural elements.
        nodes_dict (dict): Current mapping of nodes to their ranges.

    Returns:
        dict: Updated nodes dictionary with ranges.
    """
    updated_nodes_dict = {key: [] for key in nodes_dict.keys()}

    prev_element = 'START'

    for pos, element in enumerate(dot_bracket_mapper):
        if pos < len(dot_bracket_mapper) - 1:
            next_element = dot_bracket_mapper[pos + 1]
        else:
            next_element = 'END'

        if element != prev_element:
            updated_nodes_dict[element].append(pos + 1)

        if element != next_element:
            updated_nodes_dict[element].append(pos + 1)

        prev_element = element

    for node, range_vals in updated_nodes_dict.items():
        if not range_vals:
            updated_nodes_dict[node] = ''

    return updated_nodes_dict


def structure_transformation(structure, case_found, node_A_internals=None, node_A_externals=None, node_B_internals=None, node_B_externals=None):
    """
    Transforms the RNA structure based on the case found during shuffling.

    Args:
        structure (str): Current RNA structure.
        case_found (int): Identifier for the transformation case.
        node_A_internals (tuple): Internal range of node A.
        node_A_externals (tuple): External range of node A.
        node_B_internals (tuple): Internal range of node B.
        node_B_externals (tuple): External range of node B.

    Returns:
        str: Transformed RNA structure.
    """
    if case_found in [1, 2]:
        # Define connector ranges based on internal and external nodes
        connector_1_range = (node_A_internals[0] + 1, node_B_externals[0] - 1)
        connector_2_range = (node_B_externals[1] + 1, node_A_internals[1] - 1)

        # Reassemble the structure with swapped connectors
        structure_temp = structure[:node_A_internals[0]]
        structure_temp += structure[connector_2_range[0] - 1:connector_2_range[1]]
        structure_temp += structure[node_B_externals[0] - 1:node_B_externals[1]]
        structure_temp += structure[connector_1_range[0] - 1:connector_1_range[1]]
        structure_temp += structure[node_A_internals[1] - 1:]

    elif case_found in [3, 4]:
        # Define connector range for different case
        connector_range = (node_A_externals[1] + 1, node_B_externals[0] - 1)

        # Reassemble the structure with different connectors
        structure_temp = structure[:node_A_externals[0] - 1]
        structure_temp += structure[node_B_externals[0] - 1:node_B_externals[1]]
        structure_temp += structure[connector_range[0] - 1:connector_range[1]]
        structure_temp += structure[node_A_externals[0] - 1:node_A_externals[1]]
        structure_temp += structure[node_B_externals[1]:]

    return structure_temp


def structure_shuffle(pos_structure, pos_sequence, node1, node2, graph, nodes_dict, dot_bracket_mapper):
    """
    Shuffles parts of the RNA structure based on specified nodes.

    Args:
        pos_structure (str): Current positive structure.
        pos_sequence (str): Current positive sequence.
        node1 (int): First node identifier.
        node2 (int): Second node identifier.
        graph (Graph): Graph representation of the RNA structure.
        nodes_dict (dict): Mapping of nodes to their ranges.
        dot_bracket_mapper (list): Mapping of structural elements.

    Returns:
        tuple: Updated structure, sequence, and dot-bracket mapper.
    """
    node_1_name = graph.vs['name'][node1]
    node_2_name = graph.vs['name'][node2]

    node_1_internals = (nodes_dict[node_1_name][1], nodes_dict[node_1_name][2])
    node_1_externals = (nodes_dict[node_1_name][0], nodes_dict[node_1_name][3])

    node_2_internals = (nodes_dict[node_2_name][1], nodes_dict[node_2_name][2])
    node_2_externals = (nodes_dict[node_2_name][0], nodes_dict[node_2_name][3])

    # Determine the case based on node positions
    if node_1_externals[0] < node_2_externals[0] and node_1_externals[1] > node_2_externals[1]:
        case_found = 1
        node_A_internals = node_1_internals
        node_A_externals = node_1_externals
        node_B_internals = node_2_internals
        node_B_externals = node_2_externals
    elif node_2_externals[0] < node_1_externals[0] and node_2_externals[1] > node_1_externals[1]:
        case_found = 2
        node_A_internals = node_2_internals
        node_A_externals = node_2_externals
        node_B_internals = node_1_internals
        node_B_externals = node_1_externals
    elif node_1_externals[1] < node_2_externals[0]:
        case_found = 3
        node_A_internals = node_1_internals
        node_A_externals = node_1_externals
        node_B_internals = node_2_internals
        node_B_externals = node_2_externals
    elif node_2_externals[1] < node_1_externals[0]:
        case_found = 4
        node_A_internals = node_2_internals
        node_A_externals = node_2_externals
        node_B_internals = node_1_internals
        node_B_externals = node_1_externals

    # Transform the structure based on the identified case
    pos_structure = structure_transformation(
        pos_structure, case_found,
        node_A_internals, node_A_externals,
        node_B_internals, node_B_externals
    )

    validate_structure(pos_structure, "structure shuffle", dot_bracket_mapper=dot_bracket_mapper)

    # Update the dot-bracket mapper after transformation
    dot_bracket_mapper = structure_transformation(
        dot_bracket_mapper, case_found,
        node_A_internals, node_A_externals,
        node_B_internals, node_B_externals
    )

    # Transform the sequence similarly
    pos_sequence = structure_transformation(
        pos_sequence, case_found,
        node_A_internals, node_A_externals,
        node_B_internals, node_B_externals
    )

    return pos_structure, pos_sequence, dot_bracket_mapper


def stem_indels(dot_bracket_mapper, structure, sequence, modifications_counter, min_size=3, max_modifications=np.inf):
    """
    Introduces insertions or deletions in the stems of the RNA structure.

    Args:
        dot_bracket_mapper (list): Mapping of structural elements.
        structure (str): Current RNA structure.
        sequence (str): Current RNA sequence.
        modifications_counter (dict): Counter for modifications per stem.
        min_size (int): Minimum stem size for modification.
        max_modifications (int): Maximum modifications allowed per stem.

    Returns:
        tuple: Updated mapper, structure, sequence, and modifications counter.
    """
    # Count occurrences of each structural element
    elements_sizes = {key: 0 for key in dot_bracket_mapper}
    for element in elements_sizes.keys():
        elements_sizes[element] = dot_bracket_mapper.count(element)

    # Filter stems based on minimum size
    stem_sizes = {key: value for key, value in elements_sizes.items() if (key.startswith('s') and value / 2 > min_size)}

    if not stem_sizes:
        return dot_bracket_mapper, structure, sequence, modifications_counter

    total_value = sum(stem_sizes.values())
    random_num = random.randint(0, total_value - 1)

    # Select a stem proportionally to its size
    for stem, size in stem_sizes.items():
        random_num -= size
        if random_num < 0:
            selected_stem = stem
            break

    if modifications_counter[selected_stem] == max_modifications:
        return dot_bracket_mapper, structure, sequence, modifications_counter

    modifications_counter[selected_stem] += 1
    selected_option = random.choice(['insert', 'delete'])

    if selected_option == 'delete':
        # Remove bases from the stem
        first_index = dot_bracket_mapper.index(selected_stem)
        dot_bracket_mapper.pop(first_index)
        structure = structure[:first_index] + structure[first_index + 1:]
        sequence = sequence[:first_index] + sequence[first_index + 1:]
        last_index = len(dot_bracket_mapper) - dot_bracket_mapper[::-1].index(selected_stem) - 1
        dot_bracket_mapper.pop(last_index)
        structure = structure[:last_index] + structure[last_index + 1:]
        sequence = sequence[:last_index] + sequence[last_index + 1:]

    elif selected_option == 'insert':
        # Insert complementary bases into the stem
        selected_base_1 = random.choice(['C', 'G', 'A', 'U'])
        selected_base_2 = {'C': 'G', 'G': 'C', 'A': 'U', 'U': 'A'}[selected_base_1]
        first_index = dot_bracket_mapper.index(selected_stem)
        dot_bracket_mapper.insert(first_index, selected_stem)
        structure = structure[:first_index] + '(' + structure[first_index:]
        sequence = sequence[:first_index] + selected_base_1 + sequence[first_index:]
        last_index += 1
        dot_bracket_mapper.insert(last_index + 1, selected_stem)
        structure = structure[:last_index] + ')' + structure[last_index:]
        sequence = sequence[:last_index] + selected_base_2 + sequence[last_index:]

    validate_structure(structure, "stem indel", dot_bracket_mapper=dot_bracket_mapper, selected_option=selected_option)

    return dot_bracket_mapper, structure, sequence, modifications_counter


def hairpin_loop_indels(dot_bracket_mapper, structure, sequence, modifications_counter, min_size=4, max_size=8, max_modifications=np.inf):
    """
    Introduces insertions or deletions in the hairpin loops of the RNA structure.

    Args:
        dot_bracket_mapper (list): Mapping of structural elements.
        structure (str): Current RNA structure.
        sequence (str): Current RNA sequence.
        modifications_counter (dict): Counter for modifications per hairpin loop.
        min_size (int): Minimum hairpin loop size.
        max_size (int): Maximum hairpin loop size.
        max_modifications (int): Maximum modifications allowed per loop.

    Returns:
        tuple: Updated mapper, structure, sequence, and modifications counter.
    """
    # Count occurrences of each structural element
    elements_sizes = {key: 0 for key in dot_bracket_mapper}
    for element in elements_sizes.keys():
        elements_sizes[element] = dot_bracket_mapper.count(element)

    # Filter hairpin loops based on size constraints
    loops_sizes = {key: value for key, value in elements_sizes.items() if (key.startswith('h') and min_size <= value <= max_size)}

    if not loops_sizes:
        return dot_bracket_mapper, structure, sequence, modifications_counter

    selected_loop = random.choice(list(loops_sizes.keys()))

    if modifications_counter[selected_loop] == max_modifications:
        return dot_bracket_mapper, structure, sequence, modifications_counter

    modifications_counter[selected_loop] += 1
    selected_option = random.choice(['insert', 'delete'])

    # Ensure modifications stay within size limits
    if (loops_sizes[selected_loop] == min_size and selected_option == 'delete') or \
       (loops_sizes[selected_loop] == max_size and selected_option == 'insert'):
        return dot_bracket_mapper, structure, sequence, modifications_counter

    if selected_option == 'delete':
        first_index = dot_bracket_mapper.index(selected_loop)
        dot_bracket_mapper.pop(first_index)
        structure = structure[:first_index] + structure[first_index + 1:]
        sequence = sequence[:first_index] + sequence[first_index + 1:]
    elif selected_option == 'insert':
        selected_base = random.choice(['C', 'G', 'A', 'U'])
        first_index = dot_bracket_mapper.index(selected_loop)
        dot_bracket_mapper.insert(first_index, selected_loop)
        structure = structure[:first_index] + '.' + structure[first_index:]
        sequence = sequence[:first_index] + selected_base + sequence[first_index:]

    validate_structure(structure, "hairpin loop indel", dot_bracket_mapper=dot_bracket_mapper, selected_option=selected_option)

    return dot_bracket_mapper, structure, sequence, modifications_counter


def internal_loop_indels(dot_bracket_mapper, structure, sequence, modifications_counter, nodes_dict, internal_loops_list, min_size=1, max_size=8, max_modifications=np.inf):
    """
    Introduces insertions or deletions in the internal loops of the RNA structure.

    Args:
        dot_bracket_mapper (list): Mapping of structural elements.
        structure (str): Current RNA structure.
        sequence (str): Current RNA sequence.
        modifications_counter (dict): Counter for modifications per internal loop.
        nodes_dict (dict): Mapping of nodes to their ranges.
        internal_loops_list (list): List of internal loops.
        min_size (int): Minimum internal loop size.
        max_size (int): Maximum internal loop size.
        max_modifications (int): Maximum modifications allowed per loop.

    Returns:
        tuple: Updated mapper, structure, sequence, and modifications counter.
    """
    loop_sizes = {}
    for loop in internal_loops_list:
        loop_ranges = nodes_dict[loop]
        loop_A_size = loop_ranges[1] - loop_ranges[0] + 1
        if min_size <= loop_A_size <= max_size:
            loop_sizes[loop + '-A'] = loop_A_size

        loop_B_size = loop_ranges[3] - loop_ranges[2] + 1
        if min_size <= loop_B_size <= max_size:
            loop_sizes[loop + '-B'] = loop_B_size

    if not loop_sizes:
        return dot_bracket_mapper, structure, sequence, modifications_counter

    selected_loop = random.choice(list(loop_sizes.keys()))
    selected_side = selected_loop.split('-')[1]
    selected_loop_code = selected_loop.split('-')[0]

    if modifications_counter[selected_loop] == max_modifications:
        return dot_bracket_mapper, structure, sequence, modifications_counter

    modifications_counter[selected_loop] += 1
    selected_option = random.choice(['insert', 'delete'])

    # Ensure modifications stay within size limits
    if (loop_sizes[selected_loop] == min_size and selected_option == 'delete') or \
       (loop_sizes[selected_loop] == max_size and selected_option == 'insert'):
        return dot_bracket_mapper, structure, sequence, modifications_counter

    if selected_option == 'delete':
        if selected_side == 'A':
            first_index = dot_bracket_mapper.index(selected_loop_code)
            dot_bracket_mapper.pop(first_index)
            structure = structure[:first_index] + structure[first_index + 1:]
            sequence = sequence[:first_index] + sequence[first_index + 1:]
        elif selected_side == 'B':
            last_index = len(dot_bracket_mapper) - dot_bracket_mapper[::-1].index(selected_loop_code) - 1
            dot_bracket_mapper.pop(last_index)
            structure = structure[:last_index] + structure[last_index + 1:]
            sequence = sequence[:last_index] + sequence[last_index + 1:]
    elif selected_option == 'insert':
        selected_base = random.choice(['C', 'G', 'A', 'U'])
        if selected_side == 'A':
            first_index = dot_bracket_mapper.index(selected_loop_code)
            dot_bracket_mapper.insert(first_index, selected_loop_code)
            structure = structure[:first_index] + '.' + structure[first_index:]
            sequence = sequence[:first_index] + selected_base + sequence[first_index:]
        elif selected_side == 'B':
            last_index = len(dot_bracket_mapper) - dot_bracket_mapper[::-1].index(selected_loop_code) - 1
            dot_bracket_mapper.insert(last_index, selected_loop_code)
            structure = structure[:last_index] + '.' + structure[last_index:]
            sequence = sequence[:last_index] + selected_base + sequence[last_index:]

    validate_structure(structure, "internal loop indel", dot_bracket_mapper=dot_bracket_mapper, selected_option=selected_option)

    return dot_bracket_mapper, structure, sequence, modifications_counter


def bulge_indels(dot_bracket_mapper, structure, sequence, modifications_counter, bulges_list, min_size=1, max_size=8, max_modifications=np.inf):
    """
    Introduces insertions or deletions in the bulges of the RNA structure.

    Args:
        dot_bracket_mapper (list): Mapping of structural elements.
        structure (str): Current RNA structure.
        sequence (str): Current RNA sequence.
        modifications_counter (dict): Counter for modifications per bulge.
        bulges_list (list): List of bulges.
        min_size (int): Minimum bulge size.
        max_size (int): Maximum bulge size.
        max_modifications (int): Maximum modifications allowed per bulge.

    Returns:
        tuple: Updated mapper, structure, sequence, and modifications counter.
    """
    elements_sizes = {key: 0 for key in dot_bracket_mapper}
    for element in elements_sizes.keys():
        elements_sizes[element] = dot_bracket_mapper.count(element)

    bulges_sizes = {key: value for key, value in elements_sizes.items() if (key in bulges_list and min_size <= value <= max_size)}

    if not bulges_sizes:
        return dot_bracket_mapper, structure, sequence, modifications_counter

    selected_bulge = random.choice(list(bulges_sizes.keys()))

    if modifications_counter[selected_bulge] == max_modifications:
        return dot_bracket_mapper, structure, sequence, modifications_counter

    modifications_counter[selected_bulge] += 1
    selected_option = random.choice(['insert', 'delete'])

    # Ensure modifications stay within size limits
    if (bulges_sizes[selected_bulge] == min_size and selected_option == 'delete') or \
       (bulges_sizes[selected_bulge] == max_size and selected_option == 'insert'):
        return dot_bracket_mapper, structure, sequence, modifications_counter

    if selected_option == 'delete':
        first_index = dot_bracket_mapper.index(selected_bulge)
        dot_bracket_mapper.pop(first_index)
        structure = structure[:first_index] + structure[first_index + 1:]
        sequence = sequence[:first_index] + sequence[first_index + 1:]
    elif selected_option == 'insert':
        selected_base = random.choice(['C', 'G', 'A', 'U'])
        first_index = dot_bracket_mapper.index(selected_bulge)
        dot_bracket_mapper.insert(first_index, selected_bulge)
        structure = structure[:first_index] + '.' + structure[first_index:]
        sequence = sequence[:first_index] + selected_base + sequence[first_index:]

    validate_structure(structure, "bulge indel", dot_bracket_mapper=dot_bracket_mapper, selected_option=selected_option)

    return dot_bracket_mapper, structure, sequence, modifications_counter


def multi_loop_indels(dot_bracket_mapper, structure, sequence, modifications_counter, min_size=0, max_size=10, max_modifications=np.inf):
    """
    Introduces insertions or deletions in the multi-segment loops of the RNA structure.

    Args:
        dot_bracket_mapper (list): Mapping of structural elements.
        structure (str): Current RNA structure.
        sequence (str): Current RNA sequence.
        modifications_counter (dict): Counter for modifications per multi-loop.
        min_size (int): Minimum multi-loop size.
        max_size (int): Maximum multi-loop size.
        max_modifications (int): Maximum modifications allowed per multi-loop.

    Returns:
        tuple: Updated mapper, structure, sequence, and modifications counter.
    """
    elements_sizes = {key: 0 for key in dot_bracket_mapper}
    for element in elements_sizes.keys():
        elements_sizes[element] = dot_bracket_mapper.count(element)

    mloops_sizes = {key: value for key, value in elements_sizes.items() if (key.startswith('m') and min_size <= value <= max_size)}

    if not mloops_sizes:
        return dot_bracket_mapper, structure, sequence, modifications_counter

    selected_mloop = random.choice(list(mloops_sizes.keys()))

    if modifications_counter[selected_mloop] == max_modifications:
        return dot_bracket_mapper, structure, sequence, modifications_counter

    modifications_counter[selected_mloop] += 1
    selected_option = random.choice(['insert', 'delete'])

    # Ensure modifications stay within size limits
    if (mloops_sizes[selected_mloop] == min_size and selected_option == 'delete') or \
       (mloops_sizes[selected_mloop] == max_size and selected_option == 'insert'):
        return dot_bracket_mapper, structure, sequence, modifications_counter

    first_index = dot_bracket_mapper.index(selected_mloop)
    if structure[first_index] != '.':
        return dot_bracket_mapper, structure, sequence, modifications_counter

    if selected_option == 'delete':
        dot_bracket_mapper.pop(first_index)
        structure = structure[:first_index] + structure[first_index + 1:]
        sequence = sequence[:first_index] + sequence[first_index + 1:]
    elif selected_option == 'insert':
        selected_base = random.choice(['C', 'G', 'A', 'U'])
        dot_bracket_mapper.insert(first_index, selected_mloop)
        structure = structure[:first_index] + '.' + structure[first_index:]
        sequence = sequence[:first_index] + selected_base + sequence[first_index:]

    validate_structure(structure, "multi loop indel", dot_bracket_mapper=dot_bracket_mapper, selected_option=selected_option)

    return dot_bracket_mapper, structure, sequence, modifications_counter


### Functions for dinucleotide shuffling based on Altschul-Erickson algorithm

import random
from collections import Counter


def extract_dinucleotides(seq):
    """
    Extracts dinucleotides from a sequence and counts their occurrences.

    Args:
        seq (str): RNA sequence.

    Returns:
        Counter: Counts of each dinucleotide.
    """
    dinucleotides = [seq[i:i + 2] for i in range(len(seq) - 1)]
    return Counter(dinucleotides)


def generate_dinucleotide_list(dinucleotide_counts):
    """
    Generates a list of dinucleotides based on their counts.

    Args:
        dinucleotide_counts (Counter): Counts of dinucleotides.

    Returns:
        list: Flattened list of dinucleotides.
    """
    dinucleotides = list(dinucleotide_counts.keys())
    counts = list(dinucleotide_counts.values())
    flattened_list = [dinuc for dinuc, count in zip(dinucleotides, counts) for _ in range(count)]
    return flattened_list


def adjust_dinucleotide_list(dinucleotide_list, target_length):
    """
    Adjusts the dinucleotide list to match the target sequence length.

    Args:
        dinucleotide_list (list): List of dinucleotides.
        target_length (int): Desired sequence length.

    Returns:
        list: Adjusted dinucleotide list.
    """
    current_length = len(dinucleotide_list) + 1  # Original length includes one extra nucleotide
    adjustment = target_length - current_length

    if adjustment > 0:
        # Extend the list with random dinucleotides
        dinucleotides = list(set(dinucleotide_list))
        extension = random.choices(dinucleotides, k=adjustment)
        dinucleotide_list.extend(extension)
    elif adjustment < 0:
        # Truncate the list to reduce length
        dinucleotide_list = dinucleotide_list[:adjustment + len(dinucleotide_list)]

    return dinucleotide_list


def shuffle_and_reconstruct_sequence(dinucleotide_list):
    """
    Shuffles the dinucleotide list and reconstructs the RNA sequence.

    Args:
        dinucleotide_list (list): List of dinucleotides.

    Returns:
        str: Shuffled RNA sequence.
    """
    random.shuffle(dinucleotide_list)
    if not dinucleotide_list:
        return ""

    shuffled_seq = dinucleotide_list[0]
    for dinuc in dinucleotide_list[1:]:
        shuffled_seq += dinuc[1]

    return shuffled_seq


def generate_negative(anchor_sequence, neg_len_variation=0.1):
    """
    Generates a negative RNA structure by shuffling dinucleotides.

    Args:
        anchor_sequence (str): Anchor RNA sequence.
        neg_len_variation (float): Maximum length variation for negative structures.

    Returns:
        tuple: Negative sequence and its structure.
    """
    original_length = len(anchor_sequence)
    target_length = int(original_length * random.uniform(1 - neg_len_variation, 1 + neg_len_variation))

    dinucleotide_counts = extract_dinucleotides(anchor_sequence)
    dinucleotide_list = generate_dinucleotide_list(dinucleotide_counts)
    adjusted_dinucleotide_list = adjust_dinucleotide_list(dinucleotide_list, target_length)
    shuffled_seq = shuffle_and_reconstruct_sequence(adjusted_dinucleotide_list)

    negative_structure, _ = fold(shuffled_seq)

    return shuffled_seq, negative_structure


# The function that performs the triplet generation
def generate_triplet(seq_min_len, seq_max_len, seq_len_distribution, seq_len_mean, seq_len_sd,
                     variable_rearrangements, norm_nt, num_rearrangements,
                     n_stem_indels, n_hloop_indels, n_iloop_indels, n_bulge_indels, n_mloop_indels,
                     neg_len_variation, stem_min_size, stem_max_n_modifications, hloop_min_size,
                     hloop_max_size, iloop_min_size, iloop_max_size, bulge_min_size, bulge_max_size,
                     mloop_min_size, mloop_max_size, hloop_max_n_modifications, iloop_max_n_modifications,
                     bulge_max_n_modifications, mloop_max_n_modifications):
    """
    Generates RNA structure and sequence triplets for training.

    Args:
        seq_min_len (int): Minimum sequence length.
        seq_max_len (int): Maximum sequence length.
        seq_len_distribution (str): Distribution type for sequence lengths.
        seq_len_mean (int): Mean sequence length for normal distribution.
        seq_len_sd (int): Standard deviation for sequence length.
        variable_rearrangements (bool): Enable variable rearrangements.
        norm_nt (int): Nucleotide count for normalization.
        num_rearrangements (int): Number of rearrangement cycles.
        n_stem_indels (int): Number of stem indel cycles.
        n_hloop_indels (int): Number of hairpin loop indel cycles.
        n_iloop_indels (int): Number of internal loop indel cycles.
        n_bulge_indels (int): Number of bulge indel cycles.
        n_mloop_indels (int): Number of multi-loop indel cycles.
        neg_len_variation (float): Length variation for negative structures.
        stem_min_size (int): Minimum stem size.
        stem_max_n_modifications (int): Max modifications per stem.
        hloop_min_size (int): Minimum hairpin loop size.
        hloop_max_size (int): Maximum hairpin loop size.
        iloop_min_size (int): Minimum internal loop size.
        iloop_max_size (int): Maximum internal loop size.
        bulge_min_size (int): Minimum bulge size.
        bulge_max_size (int): Maximum bulge size.
        mloop_min_size (int): Minimum multi-loop size.
        mloop_max_size (int): Maximum multi-loop size.
        hloop_max_n_modifications (int): Max modifications per hairpin loop.
        iloop_max_n_modifications (int): Max modifications per internal loop.
        bulge_max_n_modifications (int): Max modifications per bulge.
        mloop_max_n_modifications (int): Max modifications per multi-loop.

    Returns:
        tuple: Generated structures and sequences triplets.
    """
    # Set sequence length based on distribution
    if seq_len_distribution == 'unif':
        seq_len = random.randint(seq_min_len, seq_max_len)
    elif seq_len_distribution == 'norm':
        while True:
            seq_len = int(np.random.normal(seq_len_mean, seq_len_sd))
            if seq_min_len <= seq_len <= seq_max_len:
                break

    # Generate random RNA sequence
    anchor_sequence = ''.join(random.choice("ACGU") for _ in range(seq_len))

    # Predict MFE secondary structure
    anchor_structure, _ = fold(anchor_sequence)

    # Validate structure
    if anchor_structure == '.' * len(anchor_structure) or anchor_structure.count(')') < 5:
        return None, None

    # Perform graph analysis on RNA structure
    cg = forgi.load_rna(anchor_structure, allow_many=False)
    graph_description = cg.to_bg_string()

    nodes_dict = {}
    edges_list = []

    lines = graph_description.splitlines()
    for line in lines:
        if line.startswith('define'):
            node_list = line.strip('define').split()
            if len(node_list) > 1:
                nodes_dict[node_list[0]] = list(map(int, node_list[1:]))
            else:
                # Handle single-node definitions
                pass

    try:
        neato_rna = cg.to_neato_string()
    except:
        return None, None

    lines = neato_rna.splitlines()
    for line in lines:
        if '--' in line:
            edge = line.strip(';').strip().split(' -- ')
            edges_list.append(edge)

    dot_bracket_mapper = [0 for _ in anchor_structure]
    for node_name, node_range in nodes_dict.items():
        if len(node_range) == 2:
            range_start, range_end = node_range[0] - 1, node_range[1] - 1
            for ix in range(range_start, range_end + 1):
                dot_bracket_mapper[ix] = node_name
        elif len(node_range) == 4:
            left_range_start, left_range_end = node_range[0] - 1, node_range[1] - 1
            for ix in range(left_range_start, left_range_end + 1):
                dot_bracket_mapper[ix] = node_name + '-A'
            right_range_start, right_range_end = node_range[2] - 1, node_range[3] - 1
            for ix in range(right_range_start, right_range_end + 1):
                dot_bracket_mapper[ix] = node_name + '-B'

    edges_dict = {edge[0]: [] for edge in edges_list}
    for edge in edges_list:
        edges_dict[edge[0]].append(edge[1])

    g = ig.Graph.ListDict(edges_dict, directed=False)
    pos_structure = anchor_structure
    g2 = g.copy()

    multiloops = [multiloop for multiloop in cg.junctions if all(element.startswith('m') for element in multiloop) and len(multiloop) > 1]

    rearrangements = math.ceil(seq_len / norm_nt) if variable_rearrangements else num_rearrangements
    shuffled = rearrangements > 0
    pos_sequence = anchor_sequence

    for shuffle_step in range(rearrangements):
        if not multiloops:
            break

        random_multiloop = random.choice(multiloops)
        random_loop = random.choice(random_multiloop)
        m_neighbors = g2.neighbors(random_loop)

        try:
            node_1 = m_neighbors[0]
            node_1_neighbors = g2.neighbors(node_1)
            node_1_loop_neighbors = [g2.vs['name'][ix] for ix in node_1_neighbors if g2.vs['name'][ix] in random_multiloop]

            node_2 = m_neighbors[1]
            node_2_neighbors = g2.neighbors(node_2)
            node_2_loop_neighbors = [g2.vs['name'][ix] for ix in node_2_neighbors if g2.vs['name'][ix] in random_multiloop]

            # Modify graph connections
            g2.delete_edges([(node_1, node_1_loop_neighbors[0]), (node_1, node_1_loop_neighbors[1])])
            g2.delete_edges([(node_2, node_2_loop_neighbors[0]), (node_2, node_2_loop_neighbors[1])])
            g2.add_edge(node_1, node_2_loop_neighbors[0])
            g2.add_edge(node_1, node_2_loop_neighbors[1])
            g2.add_edge(node_2, node_1_loop_neighbors[0])
            g2.add_edge(node_2, node_1_loop_neighbors[1])

            pos_structure, pos_sequence, dot_bracket_mapper = structure_shuffle(
                pos_structure, pos_sequence, node_1, node_2, g2, nodes_dict, dot_bracket_mapper
            )

            validate_structure(pos_structure, "structure shuffle", dot_bracket_mapper=dot_bracket_mapper)
            nodes_dict = update_nodes_dict(dot_bracket_mapper, nodes_dict)
            assert len(pos_structure) == len(anchor_structure)

        except IndexError:
            shuffled = False

    if shuffled:
        bulges_list = [node for node in nodes_dict if node.startswith('i') and len(nodes_dict[node]) == 2]
        internal_loops_list = [node for node in nodes_dict if node.startswith('i') and len(nodes_dict[node]) == 4]

        modifications_counter = {key: 0 for key in dot_bracket_mapper}

        for internal_loop in internal_loops_list:
            modifications_counter[internal_loop + '-A'] = 0
            modifications_counter[internal_loop + '-B'] = 0
            modifications_counter.pop(internal_loop, None)

        for _ in range(n_stem_indels):
            dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter = stem_indels(
                dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter,
                min_size=stem_min_size, max_modifications=stem_max_n_modifications
            )
            validate_structure(pos_structure, "stem indel", dot_bracket_mapper=dot_bracket_mapper)

        for _ in range(n_hloop_indels):
            dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter = hairpin_loop_indels(
                dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter,
                min_size=hloop_min_size, max_size=hloop_max_size,
                max_modifications=hloop_max_n_modifications
            )
            validate_structure(pos_structure, "hairpin loop indel", dot_bracket_mapper=dot_bracket_mapper)

        for _ in range(n_iloop_indels):
            dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter = internal_loop_indels(
                dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter, nodes_dict,
                internal_loops_list, min_size=iloop_min_size, max_size=iloop_max_size,
                max_modifications=iloop_max_n_modifications
            )
            validate_structure(pos_structure, "internal loop indel", dot_bracket_mapper=dot_bracket_mapper)

        for _ in range(n_bulge_indels):
            dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter = bulge_indels(
                dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter, bulges_list,
                min_size=bulge_min_size, max_size=bulge_max_size,
                max_modifications=bulge_max_n_modifications
            )
            validate_structure(pos_structure, "bulge indel", dot_bracket_mapper=dot_bracket_mapper)

        for _ in range(n_mloop_indels):
            dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter = multi_loop_indels(
                dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter,
                min_size=mloop_min_size, max_size=mloop_max_size,
                max_modifications=mloop_max_n_modifications
            )
            validate_structure(pos_structure, "multi loop indel", dot_bracket_mapper=dot_bracket_mapper)

    # Generate negative structure
    neg_sequence, neg_structure = generate_negative(anchor_sequence, neg_len_variation)

    return (anchor_structure, pos_structure, neg_structure), (anchor_sequence, pos_sequence, neg_sequence)


# Main function to run the process pool
def parallel_structure_generation(num_structures, num_workers, seq_min_len, seq_max_len, seq_len_distribution,
                                  seq_len_mean, seq_len_sd, variable_rearrangements, norm_nt, num_rearrangements,
                                  n_stem_indels, n_hloop_indels, n_iloop_indels, n_bulge_indels,
                                  n_mloop_indels, neg_len_variation, stem_min_size, stem_max_n_modifications,
                                  hloop_min_size, hloop_max_size, iloop_min_size, iloop_max_size,
                                  bulge_min_size, bulge_max_size, mloop_min_size, mloop_max_size,
                                  hloop_max_n_modifications, iloop_max_n_modifications,
                                  bulge_max_n_modifications, mloop_max_n_modifications):
    """
    Generates RNA structure triplets in parallel using multiple workers.

    Args:
        num_structures (int): Number of structures to generate.
        num_workers (int): Number of parallel workers.
        seq_min_len (int): Minimum sequence length.
        seq_max_len (int): Maximum sequence length.
        seq_len_distribution (str): Distribution type for sequence lengths.
        seq_len_mean (int): Mean sequence length for normal distribution.
        seq_len_sd (int): Standard deviation for sequence length.
        variable_rearrangements (bool): Enable variable rearrangements.
        norm_nt (int): Nucleotide count for normalization.
        num_rearrangements (int): Number of rearrangement cycles.
        n_stem_indels (int): Number of stem indel cycles.
        n_hloop_indels (int): Number of hairpin loop indel cycles.
        n_iloop_indels (int): Number of internal loop indel cycles.
        n_bulge_indels (int): Number of bulge indel cycles.
        n_mloop_indels (int): Number of multi-loop indel cycles.
        neg_len_variation (float): Length variation for negative structures.
        stem_min_size (int): Minimum stem size.
        stem_max_n_modifications (int): Max modifications per stem.
        hloop_min_size (int): Minimum hairpin loop size.
        hloop_max_size (int): Maximum hairpin loop size.
        iloop_min_size (int): Minimum internal loop size.
        iloop_max_size (int): Maximum internal loop size.
        bulge_min_size (int): Minimum bulge size.
        bulge_max_size (int): Maximum bulge size.
        mloop_min_size (int): Minimum multi-loop size.
        mloop_max_size (int): Maximum multi-loop size.
        hloop_max_n_modifications (int): Max modifications per hairpin loop.
        iloop_max_n_modifications (int): Max modifications per internal loop.
        bulge_max_n_modifications (int): Max modifications per bulge.
        mloop_max_n_modifications (int): Max modifications per multi-loop.

    Returns:
        tuple: Generated structure and sequence triplets.
    """
    structure_triplets = []
    sequence_triplets = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(generate_triplet, seq_min_len, seq_max_len, seq_len_distribution, seq_len_mean,
                            seq_len_sd, variable_rearrangements, norm_nt, num_rearrangements,
                            n_stem_indels, n_hloop_indels, n_iloop_indels,
                            n_bulge_indels, n_mloop_indels, neg_len_variation, stem_min_size,
                            stem_max_n_modifications, hloop_min_size, hloop_max_size, iloop_min_size,
                            iloop_max_size, bulge_min_size, bulge_max_size, mloop_min_size, mloop_max_size,
                            hloop_max_n_modifications, iloop_max_n_modifications,
                            bulge_max_n_modifications, mloop_max_n_modifications)
            for _ in range(num_structures)
        ]

        with tqdm(total=num_structures, desc="Processing") as pbar:
            for future in as_completed(futures):
                result_structures, result_sequences = future.result()
                if result_structures and result_sequences:
                    structure_triplets.append(result_structures)
                    sequence_triplets.append(result_sequences)
                pbar.update(1)

    print(f'\n{num_structures} structure pairs generated')
    return structure_triplets, sequence_triplets


def plot_rna_structure(ax, sequence, structure, structure_name):
    """
    Plots a single RNA structure on the given axis.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
        sequence (str): RNA sequence.
        structure (str): RNA structure in dot-bracket notation.
        structure_name (str): Identifier for the structure.

    Returns:
        bool: True if plotting was successful, False otherwise.
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp_file:
            temp_file.write('>' + structure_name + '\n')
            temp_file.write(sequence + '\n')
            temp_file.write(structure + '\n')

            cg = forgi.load_rna(temp_file.name, allow_many=False)

            fvm.plot_rna(cg, text_kwargs={"fontweight": "black"}, lighten=0.7,
                         backbone_kwargs={"linewidth": 3}, ax=ax)
            ax.set_title(structure_name)
            return True
    except Exception as e:
        print(f"Warning: Failed to plot structure {structure_name}: {str(e)}")
        return False


def plot_triplets(df, plot_dir, num_samples=5):
    """
    Generates plots for a specified number of RNA structure triplets.

    Args:
        df (DataFrame): DataFrame containing structure triplets.
        plot_dir (str): Directory to save plots.
        num_samples (int): Number of triplets to plot.

    Returns:
        None
    """
    successful_plots = 0
    attempted_indices = set()
    max_attempts = min(df.shape[0], num_samples * 3)

    with tqdm(total=num_samples, desc="Plotting triplets") as pbar:
        while successful_plots < num_samples and len(attempted_indices) < max_attempts:
            index = random.randint(0, df.shape[0] - 1)
            if index in attempted_indices:
                continue
            attempted_indices.add(index)
            triplet = df.iloc[index]
            success = plot_rna_structure(ax=None, sequence=triplet['sequence'], structure=triplet['structure'], structure_name=str(index))
            if success:
                successful_plots += 1
                pbar.update(1)

    if successful_plots < num_samples:
        print(f"Warning: Only managed to plot {successful_plots}/{num_samples} requested triplets")


def split_dataset(df, train_fraction, val_fraction):
    """
    Splits the dataset into training and validation sets.

    Args:
        df (DataFrame): Complete dataset.
        train_fraction (float): Fraction of data for training.
        val_fraction (float): Fraction of data for validation.

    Returns:
        tuple: Training and validation DataFrames.
    """
    if train_fraction + val_fraction != 1.0:
        raise ValueError("Train and validation fractions must sum to 1.0")

    train_size = int(len(df) * train_fraction)
    train_df = df.sample(train_size, random_state=42)
    val_df = df.drop(train_df.index)

    return train_df, val_df