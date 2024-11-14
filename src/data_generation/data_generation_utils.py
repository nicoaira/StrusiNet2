import numpy as np
import forgi
from ViennaRNA import fold
import math
import igraph as ig
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def update_nodes_dict(dot_bracket_mapper, nodes_dict):

    ''' This functions updates the nodes dict, which helps to map the range
    of each element of the strucutre, using the dot_bracket_mapper.
    There is a for loop that advances through the dot_bracket_mapper.
    In each step, it analyze whether
    '''

    updated_nodes_dict = {key : [] for key in nodes_dict.keys()}

    prev_element = 'START'

    for pos, element in enumerate(dot_bracket_mapper):

      if pos < len(dot_bracket_mapper) - 1:
        next_element = dot_bracket_mapper[pos+1]
      else:
        next_element = 'END'

      if element != prev_element:
        updated_nodes_dict[element].append(pos + 1)

      if element != next_element:
        updated_nodes_dict[element].append(pos + 1)

      prev_element = element


    for node, range in updated_nodes_dict.items():
      if range == []:
        updated_nodes_dict[node] = ''

    return updated_nodes_dict


def structure_transformation(structure,
                            case_found,
                            node_A_internals = None,
                            node_A_externals = None,
                            node_B_internals = None,
                            node_B_externals = None,
                            ):

    if case_found == 1 or case_found == 2:

        connector_1_range = ( node_A_internals[0] + 1 , node_B_externals[0] - 1 )
        connector_2_range = ( node_B_externals[1] + 1 , node_A_internals[1] - 1 )

        structure_temp  = structure[ :node_A_internals[0]]
        structure_temp += structure[connector_2_range[0] - 1 : connector_2_range[1]]
        structure_temp += structure[node_B_externals[0] - 1 : node_B_externals[1]]
        structure_temp += structure[connector_1_range[0] - 1 : connector_1_range[1]]
        structure_temp += structure[node_A_internals[1] - 1 : ]

    elif case_found == 3 or case_found == 4:

        connector_range = ( node_A_externals[1] + 1 , node_B_externals[0] - 1 )

        structure_temp  = structure[:node_A_externals[0] - 1]
        structure_temp += structure[node_B_externals[0] - 1 : node_B_externals[1]]
        structure_temp += structure[connector_range[0] - 1 : connector_range[1]]
        structure_temp += structure[node_A_externals[0] - 1 : node_A_externals[1]]
        structure_temp += structure[node_B_externals[1]:]

    return structure_temp


def structure_shuffle(pos_structure, pos_sequence, node1, node2, graph,
                      nodes_dict, dot_bracket_mapper):



    node_1_name = graph.vs['name'][node1]

    node_2_name = graph.vs['name'][node2]

    node_1_internals = ( nodes_dict[node_1_name][1] , nodes_dict[node_1_name][2] )
    node_1_externals = ( nodes_dict[node_1_name][0] , nodes_dict[node_1_name][3] )

    node_2_internals = ( nodes_dict[node_2_name][1] , nodes_dict[node_2_name][2] )
    node_2_externals = ( nodes_dict[node_2_name][0] , nodes_dict[node_2_name][3] )

    ## Two possible scenarios for the swapping:
    ## 1 -  Both structures to swap are independent, i.e., node_1 and node_2 ranges are not overlapping
    ## 2 -  One of the structures is inside the range of the other one. This is, the
    ##      structure_2_start


    if node_1_externals[0] < node_2_externals[0] and node_1_externals[1] > node_2_externals[1]:

        case_found = 1

        # node_1 = A
        # node_2 = B

        node_A_internals = node_1_internals
        node_A_externals = node_1_externals

        node_B_internals = node_2_internals
        node_B_externals = node_2_externals

    elif node_2_externals[0] < node_1_externals[0] and node_2_externals[1] > node_1_externals[1]:

        case_found = 2

        # node_2 = A
        # node_1 = B

        node_A_internals = node_2_internals
        node_A_externals = node_2_externals

        node_B_internals = node_1_internals
        node_B_externals = node_1_externals

    elif node_1_externals[1] < node_2_externals[0]:

        case_found = 3

        # node_1 = A
        # node_2 = B

        node_A_internals = node_1_internals
        node_A_externals = node_1_externals

        node_B_internals = node_2_internals
        node_B_externals = node_2_externals


    elif node_2_externals[1] < node_1_externals[0]:

        case_found = 4

        # node_2 = A
        # node_1 = B

        node_A_internals = node_2_internals
        node_A_externals = node_2_externals

        node_B_internals = node_1_internals
        node_B_externals = node_1_externals



    pos_structure = structure_transformation(pos_structure,
                                             case_found,
                                             node_A_internals,
                                             node_A_externals,
                                             node_B_internals,
                                             node_B_externals)




    dot_bracket_mapper = structure_transformation(dot_bracket_mapper,
                                                  case_found,
                                                  node_A_internals,
                                                  node_A_externals,
                                                  node_B_internals,
                                                  node_B_externals)


    pos_sequence = structure_transformation(pos_sequence,
                                            case_found,
                                            node_A_internals,
                                            node_A_externals,
                                            node_B_internals,
                                            node_B_externals)

    return pos_structure, pos_sequence, dot_bracket_mapper


def stem_indels(dot_bracket_mapper,
                structure,
                sequence,
                modifications_counter,
                min_size = 3,
                max_modifications=np.inf):

  ''' Takes a stucture and a dot_bracket_mapper and randomly makes an insertion
  or deletion in one the stems of the structure. The chance of selection of a
  stem is proportional of to its size. There is a possibility to set a min size
  of a stem to be modified.
  '''

  elements_sizes = {key : 0 for key in dot_bracket_mapper}

  for element in elements_sizes.keys():
      elements_sizes[element] = dot_bracket_mapper.count(element)

  stem_sizes = {key : value for key, value in elements_sizes.items() if (key.startswith('s') and min_size < value/2 )}

  # If there are no stems with size > min_size, then return the unaffected structure

  if not stem_sizes:
    return dot_bracket_mapper, structure, sequence, modifications_counter

  total_value = sum(stem_sizes.values())

  ## Randomly select an stem with a probability proportional to its size

  random_num = random.randint(0, total_value - 1)

  # Iterate through the dictionary to find the key corresponding to the random number
  for stem, size in stem_sizes.items():
      random_num -= size
      if random_num < 0:
          selected_stem = stem
          break

  # If the selected stem already suffered the max number of modifications, then
  # it returns the unaffected structure
  if modifications_counter[selected_stem] == max_modifications:
      return dot_bracket_mapper, structure, sequence, modifications_counter

  else:
      pass

  modifications_counter[selected_stem] += 1

  options = ['insert', 'delete']
  selected_option = random.choice(options)

  if selected_option == 'delete':
    # Find the index of the first occurrence of the element

    #####
    # ESTO DELECIONA/INTRODUCE LA BASE SIEMPRE AL PRINCIPIO DEL STEM (BIAS?)
    #####

    first_index = dot_bracket_mapper.index(selected_stem)

    # Find the index of the last occurrence of the element
    last_index = len(dot_bracket_mapper) - dot_bracket_mapper[::-1].index(selected_stem) - 1

    # Remove the first base of the stem
    dot_bracket_mapper.pop(first_index)
    structure = structure[:first_index] + structure[first_index+1:]
    sequence = sequence[:first_index] + sequence[first_index+1:]

    # we need to modify the last_index since the structure has been shorten in 1
    last_index -= 1

    # Remove the last base of the stem (the one basepaired with the first one)
    dot_bracket_mapper.pop(last_index)
    structure = structure[:last_index] + structure[last_index+1:]
    sequence = sequence[:last_index] + sequence[last_index+1:]


  if selected_option == 'insert':

    # Define bases to insert

    selected_base_1 = random.choice(['C', 'G', 'A', 'U'])

    ####
    # NO G:U ?
    ####

    if selected_base_1 == 'C':
        selected_base_2 = 'G'
    elif selected_base_1 == 'G':
        selected_base_2 = 'C'
    elif selected_base_1 == 'A':
        selected_base_2 = 'U'
    elif selected_base_1 == 'U':
        selected_base_2 = 'A'


    # Find the index of the first occurrence of the element
    first_index = dot_bracket_mapper.index(selected_stem)

    # Find the index of the last occurrence of the element
    last_index = len(dot_bracket_mapper) - dot_bracket_mapper[::-1].index(selected_stem) - 1

    dot_bracket_mapper.insert(first_index, selected_stem)
    structure = structure[:first_index] + '(' + structure[first_index:]
    sequence = sequence[:first_index] + selected_base_1 + sequence[first_index:]

    # we need to modify the last_index since the structure has been enlargen in 1
    last_index += 1
    dot_bracket_mapper.insert(last_index+1, selected_stem)
    structure = structure[:last_index] + ')' + structure[last_index:]
    sequence = sequence[:last_index] + selected_base_2 + sequence[last_index:]


  return dot_bracket_mapper, structure, sequence, modifications_counter


def hairpin_loop_indels(dot_bracket_mapper,
                        structure,
                        sequence,
                        modifications_counter,
                        min_size = 4,
                        max_size = 8,
                        max_modifications=np.inf):

  ''' Takes a stucture and a dot_bracket_mapper and randomly makes an insertion
  or deletion in one the hairpin loops of the structure.
  There is a possibility to set a min and max size of a hairpin to be modified.
  '''

  elements_sizes = {key : 0 for key in dot_bracket_mapper}

  for element in elements_sizes.keys():
      elements_sizes[element] = dot_bracket_mapper.count(element)

  loops_sizes = {key : value for key, value in elements_sizes.items() if (key.startswith('h') and min_size <= value <= max_size)}

  # If there are no loops with min_size < size < max_size, then return the
  # unaffected structure

  if not loops_sizes:
    return dot_bracket_mapper, structure, sequence, modifications_counter

  selected_loop = random.choice(list(loops_sizes.keys()))

  # If the selected loop already suffered the max number of modifications, then
  # it returns the unaffected structure
  if modifications_counter[selected_loop] == max_modifications:
      return dot_bracket_mapper, structure, sequence, modifications_counter
  else:
      pass

  options = ['insert', 'delete']
  selected_option = random.choice(options)

  modifications_counter[selected_loop] += 1

  # Make sure the modification does not make the loop to pass the limits
  if (loops_sizes[selected_loop] == min_size and selected_option == 'delete') \
   or (loops_sizes[selected_loop] == max_size and selected_option == 'insert'):
    return dot_bracket_mapper, structure, sequence, modifications_counter
  else:
    pass

  if selected_option == 'delete':
    # Find the index of the first occurrence of the element
    first_index = dot_bracket_mapper.index(selected_loop)

    dot_bracket_mapper.pop(first_index)
    structure = structure[:first_index] + structure[first_index+1:]
    sequence = sequence[:first_index] + sequence[first_index+1:]

  if selected_option == 'insert':

    # Select base to insert

    selected_base = random.choice(['C', 'G', 'A', 'U'])

    # Find the index of the first occurrence of the element
    first_index = dot_bracket_mapper.index(selected_loop)

    dot_bracket_mapper.insert(first_index, selected_loop)
    structure = structure[:first_index] + '.' + structure[first_index:]
    sequence = sequence[:first_index] + selected_base + sequence[first_index:]


  return dot_bracket_mapper, structure, sequence, modifications_counter


def internal_loop_indels(dot_bracket_mapper,
                         structure,
                         sequence,
                         modifications_counter,
                         nodes_dict,
                         internal_loops_list,
                         min_size = 1,
                         max_size = 8,
                         max_modifications=np.inf):

  ''' Takes a stucture and a dot_bracket_mapper and randomly makes an insertion
  or deletion in one the internal loops of the structure.
  There is a possibility to set a min and max size of a loop to be modified.
  '''

  loop_sizes = {}
  # this dictionary will store the size of all the loop that are within the
  # range for modifications. It will diffirenciate both sides of the loop
  for loop in internal_loops_list:

    loop_ranges = nodes_dict[loop]
    loop_A = loop_ranges[:2] # range of the loop at the 5' side
    loop_A_size = loop_A[1] - loop_A[0] + 1
    if min_size <= loop_A_size <= max_size:
      loop_sizes[loop+'-A'] = loop_A_size

    loop_B = loop_ranges[2:] # range of the loop at the 3' side
    loop_B_size = loop_B[1] - loop_B[0] + 1
    if min_size <= loop_B_size <= max_size:
      loop_sizes[loop+'-B'] = loop_B_size


  # If there are no loops with min_size < size < max_size, then return the
  # unaffected structure

  if not loop_sizes:
    return dot_bracket_mapper, structure, sequence, modifications_counter

  selected_loop = random.choice(list(loop_sizes.keys()))

  selected_side = selected_loop.split('-')[1] # A or B (5' or 3' side of the loop)

  selected_loop_code = selected_loop.split('-')[0] # the element code (e.g. i3)

  # If the selected loop already suffered the max number of modifications, then
  # it returns the unaffected structure

  if modifications_counter[selected_loop] == max_modifications:
      return dot_bracket_mapper, structure, sequence, modifications_counter
  else:
      pass

  options = ['insert', 'delete']
  selected_option = random.choice(options)

  modifications_counter[selected_loop] += 1

  # Make sure the modification does not make the loop to pass the limits
  if (loop_sizes[selected_loop] == min_size and selected_option == 'delete') \
   or (loop_sizes[selected_loop] == max_size and selected_option == 'insert'):
    return dot_bracket_mapper, structure, sequence, modifications_counter
  else:
    pass


  if selected_option == 'delete':
    if selected_side == 'A':
      # Find the index of the first occurrence of the element
      first_index = dot_bracket_mapper.index(selected_loop_code)

      dot_bracket_mapper.pop(first_index)
      structure = structure[:first_index] + structure[first_index+1:]
      sequence = sequence[:first_index] + sequence[first_index+1:]

    elif selected_side == 'B':
      # Find the index of the last occurrence of the element
      last_index = dot_bracket_mapper[::-1].index(selected_loop_code)  # Reverse the list and find the index
      last_index = len(dot_bracket_mapper) - last_index - 1  # Adjust the index to the original list

      dot_bracket_mapper.pop(last_index)
      structure = structure[:last_index] + structure[last_index+1:]
      sequence = sequence[:last_index] + sequence[last_index+1:]


  elif selected_option == 'insert':

    selected_base = random.choice(['C', 'G', 'A', 'U'])

    if selected_side == 'A':
      # Find the index of the first occurrence of the element
      first_index = dot_bracket_mapper.index(selected_loop_code)

      dot_bracket_mapper.insert(first_index, selected_loop_code)
      structure = structure[:first_index] + '.' + structure[first_index:]
      sequence = sequence[:first_index] + selected_base + sequence[first_index:]

    elif selected_side == 'B':
      # Find the index of the last occurrence of the element
      last_index = dot_bracket_mapper[::-1].index(selected_loop_code)  # Reverse the list and find the index
      last_index = len(dot_bracket_mapper) - last_index - 1  # Adjust the index to the original list

      dot_bracket_mapper.insert(last_index, selected_loop_code)

      structure = structure[:last_index] + '.' + structure[last_index:]
      sequence = sequence[:last_index] + selected_base + sequence[last_index:]

  return dot_bracket_mapper, structure, sequence, modifications_counter


def bulge_indels(dot_bracket_mapper,
                 structure,
                 sequence,
                 modifications_counter,
                 bulges_list,
                 min_size = 1,
                 max_size = 8,
                 max_modifications=np.inf):

  ''' Takes a stucture and a dot_bracket_mapper and randomly makes an insertion
  or deletion in one the bulges of the structure.
  There is a possibility to set a min and max size of a hairpin to be modified.
  '''

  elements_sizes = {key : 0 for key in dot_bracket_mapper}

  for element in elements_sizes.keys():
      elements_sizes[element] = dot_bracket_mapper.count(element)

  bulges_sizes = {key : value for key, value in elements_sizes.items() if (key in bulges_list and min_size <= value <= max_size)}

  # If there are no bulges with min_size < size < max_size, then return the
  # unaffected structure


  if not bulges_sizes:
    return dot_bracket_mapper, structure, sequence, modifications_counter

  selected_bulge = random.choice(list(bulges_sizes.keys()))

  # If the selected bluge already suffered the max number of modifications, then
  # it returns the unaffected structure

  if modifications_counter[selected_bulge] == max_modifications:
      return dot_bracket_mapper, structure, sequence, modifications_counter
  else:
      pass

  options = ['insert', 'delete']
  selected_option = random.choice(options)

  modifications_counter[selected_bulge] += 1

  # Make sure the modification does not make the loop to pass the limits
  if (bulges_sizes[selected_bulge] == min_size and selected_option == 'delete') \
   or (bulges_sizes[selected_bulge] == max_size and selected_option == 'insert'):
    return dot_bracket_mapper, structure, sequence, modifications_counter
  else:
    pass

  if selected_option == 'delete':
    # Find the index of the first occurrence of the element
    first_index = dot_bracket_mapper.index(selected_bulge)

    dot_bracket_mapper.pop(first_index)
    structure = structure[:first_index] + structure[first_index+1:]
    sequence = sequence[:first_index] + sequence[first_index+1:]

  if selected_option == 'insert':

    selected_base = random.choice(['C', 'G', 'A', 'U'])

    # Find the index of the first occurrence of the element
    first_index = dot_bracket_mapper.index(selected_bulge)

    dot_bracket_mapper.insert(first_index, selected_bulge)
    structure = structure[:first_index] + '.' + structure[first_index:]
    sequence = sequence[:first_index] + selected_base + sequence[first_index:]


  return dot_bracket_mapper, structure, sequence, modifications_counter


def multi_loop_indels(dot_bracket_mapper,
                      structure,
                      sequence,
                      modifications_counter,
                      min_size = 0,
                      max_size = 10,
                      max_modifications=np.inf):

  ''' Takes a stucture and a dot_bracket_mapper and randomly makes an insertion
  or deletion in one the multisegment loops of the structure.
  There is a possibility to set a min and max size of a  multisegment loops
  to be modified.
  '''

  elements_sizes = {key : 0 for key in dot_bracket_mapper}

  for element in elements_sizes.keys():
      elements_sizes[element] = dot_bracket_mapper.count(element)

  mloops_sizes = {key : value for key, value in elements_sizes.items() if (key.startswith('m') and min_size <= value <= max_size)}

  # If there are no mloops with min_size < size < max_size, then return the
  # unaffected structure

  if not mloops_sizes:
    return dot_bracket_mapper, structure, sequence, modifications_counter

  selected_mloop = random.choice(list(mloops_sizes.keys()))

  # If the selected loop already suffered the max number of modifications, then
  # it returns the unaffected structure
  if modifications_counter[selected_mloop] == max_modifications:
      return dot_bracket_mapper, structure, sequence,  modifications_counter
  else:
      pass

  options = ['insert', 'delete']
  selected_option = random.choice(options)

  modifications_counter[selected_mloop] += 1

  # Make sure the modification does not make the loop to pass the limits
  if (mloops_sizes[selected_mloop] == min_size and selected_option == 'delete') \
   or (mloops_sizes[selected_mloop] == max_size and selected_option == 'insert'):
    return dot_bracket_mapper, structure, sequence, modifications_counter
  else:
    pass

  first_index = dot_bracket_mapper.index(selected_mloop)
  if structure[first_index] != '.':
    return dot_bracket_mapper, structure, sequence, modifications_counter

  if selected_option == 'delete':
    # Find the index of the first occurrence of the element

    dot_bracket_mapper.pop(first_index)
    structure = structure[:first_index] + structure[first_index+1:]
    sequence = sequence[:first_index] + sequence[first_index+1:]

  if selected_option == 'insert':

    selected_base = random.choice(['C', 'G', 'A', 'U'])

    # Find the index of the first occurrence of the element

    dot_bracket_mapper.insert(first_index, selected_mloop)
    structure = structure[:first_index] + '.' + structure[first_index:]
    sequence = sequence[:first_index] + selected_base + sequence[first_index:]

  return dot_bracket_mapper, structure, sequence, modifications_counter


### Functions for dinucleotide shuffling based on Altschul-Erickson algorithm

import random
from collections import Counter

def extract_dinucleotides(seq):
    """Extract dinucleotides from the sequence and count their occurrences."""
    dinucleotides = [seq[i:i+2] for i in range(len(seq) - 1)]
    return Counter(dinucleotides)

def generate_dinucleotide_list(dinucleotide_counts):
    """Generate a list of dinucleotides based on their counts."""
    dinucleotides = list(dinucleotide_counts.keys())
    counts = list(dinucleotide_counts.values())
    flattened_list = [dinuc for dinuc, count in zip(dinucleotides, counts) for _ in range(count)]
    return flattened_list

def adjust_dinucleotide_list(dinucleotide_list, target_length):
    """Adjust the dinucleotide list to match the target length."""
    current_length = len(dinucleotide_list) + 1  # Original length includes one extra nucleotide
    adjustment = target_length - current_length

    if adjustment > 0:
        # Extend the list with random dinucleotides
        dinucleotides = list(set(dinucleotide_list))  # Unique dinucleotides
        extension = random.choices(dinucleotides, k=adjustment)
        dinucleotide_list.extend(extension)
    elif adjustment < 0:
        # Truncate the list
        dinucleotide_list = dinucleotide_list[:adjustment + len(dinucleotide_list)]

    return dinucleotide_list

def shuffle_and_reconstruct_sequence(dinucleotide_list):
    """Shuffle the dinucleotide list and reconstruct the sequence."""
    random.shuffle(dinucleotide_list)
    if not dinucleotide_list:
        return ""

    shuffled_seq = dinucleotide_list[0]
    for dinuc in dinucleotide_list[1:]:
        shuffled_seq += dinuc[1]

    return shuffled_seq

def generate_negative(anchor_sequence, neg_len_variation=.1):
    '''This function takes the random sequence to
    be used as anchor during triplets training and
    generate a random negative structure thas has
    the dinucleotide frequency as the anchor,
    to prevent bias.
    As the positive strucutre can have some variation in length,
    we allow some degree of length variation in the negative structure
    also.'''

    original_length = len(anchor_sequence)
    target_length = int(original_length * random.uniform(1-neg_len_variation,
                                                         1+neg_len_variation))  # Length within ±5%

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

    # Set the size distribution of anchor RNAs
    if seq_len_distribution == 'unif':
        seq_len = random.randint(seq_min_len, seq_max_len)
    elif seq_len_distribution == 'norm':
        while True:
            seq_len = int(np.random.normal(seq_len_mean, seq_len_sd))
            if seq_min_len <= seq_len <= seq_max_len:
                break

    # Generate a random RNA sequence
    anchor_sequence = ''.join(random.choice("ACGU") for _ in range(seq_len))

    # Predict the minimum free energy (MFE) secondary structure
    anchor_structure, _ = fold(anchor_sequence)

    # Check if the structure is valid
    if anchor_structure == '.' * len(anchor_structure) or anchor_structure.count(')') < 5:
        return None, None

    # Create the RNA object and perform graph analysis
    cg = forgi.load_rna(anchor_structure, allow_many=False)
    graph_description = cg.to_bg_string()

    nodes_dict = {}
    edges_list = []

    lines = graph_description.splitlines()
    for line in lines:
        if line.startswith('define'):
            node_list = line.strip('define').split()
            if len(node_list) > 1:
                nodes_dict[node_list[0]] = [int(str_base) for str_base in node_list[1:]]
            else:
                nodes_dict[node_list[0]] = ''

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
            range_start = node_range[0] - 1
            range_end = node_range[1] - 1
            for ix in range(range_start, range_end + 1):
                dot_bracket_mapper[ix] = node_name
        elif len(node_range) == 4:
            left_range_start = node_range[0] - 1
            left_range_end = node_range[1] - 1
            for ix in range(left_range_start, left_range_end + 1):
                dot_bracket_mapper[ix] = node_name
            right_range_start = node_range[2] - 1
            right_range_end = node_range[3] - 1
            for ix in range(right_range_start, right_range_end + 1):
                dot_bracket_mapper[ix] = node_name

    edges_dict = {edge[0]: [] for edge in edges_list}
    for edge in edges_list:
        edges_dict[edge[0]].append(edge[1])

    g = ig.Graph.ListDict(edges_dict, directed=False)
    pos_structure = anchor_structure
    g2 = g.copy()

    multiloops = [multiloop for multiloop in cg.junctions if all(element.startswith('m') for element in multiloop) and len(multiloop) > 1]


    rearrangements = math.ceil(seq_len/norm_nt) if variable_rearrangements else num_rearrangements

    shuffled = rearrangements > 0
    pos_sequence = anchor_sequence

    for shuffle_step in range(rearrangements):
        # print('multiloops',multiloops)
        if len(multiloops) < 1:
            # print('breaking')
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

            g2.delete_edges([(node_1, node_1_loop_neighbors[0]), (node_1, node_1_loop_neighbors[1])])
            g2.delete_edges([(node_2, node_2_loop_neighbors[0]), (node_2, node_2_loop_neighbors[1])])
            g2.add_edge(node_1, node_2_loop_neighbors[0])
            g2.add_edge(node_1, node_2_loop_neighbors[1])
            g2.add_edge(node_2, node_1_loop_neighbors[0])
            g2.add_edge(node_2, node_1_loop_neighbors[1])

            pos_structure, pos_sequence, dot_bracket_mapper = structure_shuffle(pos_structure, pos_sequence, node_1, node_2, g2, nodes_dict, dot_bracket_mapper)

            nodes_dict = update_nodes_dict(dot_bracket_mapper, nodes_dict)

            assert(len(pos_structure) == len(anchor_structure))

        except IndexError:
            shuffled = False


    if shuffled:
        bulges_list = [node for node in nodes_dict if node.startswith('i') and len(nodes_dict[node]) == 2]
        internal_loops_list = [node for node in nodes_dict if node.startswith('i') and len(nodes_dict[node]) == 4]

        modifications_counter = {key: 0 for key in dot_bracket_mapper}

        for internal_loop in internal_loops_list:
            modifications_counter[internal_loop+'-A'] = 0
            modifications_counter[internal_loop+'-B'] = 0
            modifications_counter.pop(internal_loop, None)

        for _ in range(n_stem_indels):
            dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter = stem_indels(
                dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter,
                min_size=stem_min_size, max_modifications=stem_max_n_modifications)

        for _ in range(n_hloop_indels):
            dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter = hairpin_loop_indels(
                dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter,
                min_size=hloop_min_size, max_size=hloop_max_size,
                max_modifications=hloop_max_n_modifications)

        for _ in range(n_iloop_indels):
            dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter = internal_loop_indels(
                dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter, nodes_dict,
                internal_loops_list, min_size=iloop_min_size, max_size=iloop_max_size,
                max_modifications=iloop_max_n_modifications)

        for _ in range(n_bulge_indels):
            dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter = bulge_indels(
                dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter, bulges_list,
                min_size=bulge_min_size, max_size=bulge_max_size,
                max_modifications=bulge_max_n_modifications)

        for _ in range(n_mloop_indels):
            dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter = multi_loop_indels(
                dot_bracket_mapper, pos_structure, pos_sequence, modifications_counter,
                min_size=mloop_min_size, max_size=mloop_max_size,
                max_modifications=mloop_max_n_modifications)

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
                triplet_result, sequence_result = future.result()

                if triplet_result is not None and sequence_result is not None:
                    structure_triplets.append(triplet_result)
                    sequence_triplets.append(sequence_result)

                pbar.update(1)

    print(f'\n{num_structures} structure pairs generated')
    return structure_triplets, sequence_triplets