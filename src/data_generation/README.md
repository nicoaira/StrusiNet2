# RNA Structure Data Generation Module Documentation

## Overview

This module generates synthetic RNA structure-sequence triplets for training machine learning models. Each triplet consists of:
- **Anchor (A)**: Original valid RNA structure and its sequence
- **Positive (P)**: Modified version of anchor that maintains structural similarity while introducing local variations
- **Negative (N)**: Significantly different structure with similar sequence composition

## Key Components and Workflow

### 1. Sequence and Structure Generation
1. **Anchor Generation**
   - Random RNA sequence generation with specified length distribution
   - Structure prediction using Vienna RNA package
   - Validation of structural complexity (minimum stem count)

2. **Positive Structure Generation**
   - Graph-based structure representation
   - Systematic modifications through:
     - Stem insertions/deletions
     - Loop modifications
     - Branch rearrangements
   - Maintains global structure while introducing local variations

3. **Negative Structure Generation**
   - Dinucleotide-shuffled sequence generation
   - Preserves sequence composition bias
   - Structure prediction on shuffled sequence
   - Ensures significant structural difference from anchor

### 2. Modification Types

#### Stem Modifications
- Insertions: Add base pairs to existing stems
- Deletions: Remove base pairs from stems
- Size constraints and maximum modification limits
- Probability of selection proportional to stem size

#### Loop Modifications
1. **Hairpin Loops**
   - Single-stranded regions at stem ends
   - Insertions/deletions of unpaired bases
   - Size constraints: 4-12 nucleotides (configurable)

2. **Internal Loops**
   - Unpaired regions between stems
   - Independent modification of 5' and 3' sides
   - Size constraints: 1-8 nucleotides per side

3. **Bulge Loops**
   - Single-stranded regions on one side of stem
   - Size constraints: 1-8 nucleotides

4. **Multi-branch Loops**
   - Junction regions connecting multiple stems
   - Conservative modifications to maintain structure

### 3. Branch Rearrangements
- Graph-based representation of structure
- Identification of valid swap points
- Maintains structural validity
- Number of rearrangements can be:
  - Fixed
  - Length-dependent (normalized by sequence length)

## Parameters Configuration

### Sequence Generation
```python
{
    'num_structures': 100000,  # Total triplets to generate
    'seq_min_len': 60,        # Minimum sequence length
    'seq_max_len': 600,       # Maximum sequence length
    'seq_len_distribution': 'unif',  # 'unif' or 'norm'
    'seq_len_mean': 24,       # Mean length for normal distribution
    'seq_len_sd': 7           # SD for normal distribution
}
```

### Structural Modifications
```python
{
    # Stem parameters
    'n_stem_indels': 4,       # Number of stem modification cycles
    'stem_min_size': 3,       # Minimum stem size for modification
    'stem_max_n_modifications': 10,  # Max modifications per stem

    # Loop parameters
    'n_hloop_indels': 4,      # Number of hairpin loop modifications
    'hloop_min_size': 4,      # Minimum hairpin loop size
    'hloop_max_size': 12,     # Maximum hairpin loop size
    'hloop_max_n_modifications': 5,  # Max modifications per hairpin
    
    # Similar parameters for internal, bulge, and multi-loops
}
```

### Performance Parameters
```python
{
    'num_workers': 16,        # Number of parallel processes
    'variable_rearrangements': True,  # Length-dependent rearrangements
    'norm_nt': 100,           # Nucleotides per rearrangement
    'split': False,           # Enable dataset splitting
    'train_fraction': 0.8,    # Fraction of data for training
    'val_fraction': 0.2       # Fraction of data for validation
}
```

## Output Format

### File Structure
Results directory contains:
- `results.csv`: Main data file with structures and sequences
- `results_metadata.json`: Run parameters and metadata
- `plots/`: Optional visualizations of structure triplets
- `train.csv`: Training data file (if split is enabled)
- `val.csv`: Validation data file (if split is enabled)

### CSV Format
```csv
structure_A,structure_P,structure_N,sequence_A,sequence_P,sequence_N
(((...))),.((....)),........,AUGCUAGC,AUGCUAGC,GCUAGCAU
```

## Usage Example

```python
from data_generation_utils import parallel_structure_generation

# Basic usage
structures, sequences = parallel_structure_generation(
    num_structures=1000,
    num_workers=16,
    seq_min_len=60,
    seq_max_len=600,
    results_dir='path/to/output'
)

# Using command line tool
python generate_data.py \
    --num_structures 1000 \
    --seq_min_len 60 \
    --seq_max_len 600 \
    --results_dir path/to/output \
    --plot_structures

# Using command line tool with dataset splitting
python generate_data.py \
    --num_structures 1000 \
    --seq_min_len 60 \
    --seq_max_len 600 \
    --results_dir path/to/output \
    --split \
    --train_fraction 0.8 \
    --val_fraction 0.2
```

## Visualization

The module can generate visualizations of structure triplets using forgi:
- 2D structure plots with stems and loops
- Color-coded structural elements
- Comparative visualization of A/P/N structures

## Dependencies

- ViennaRNA
- forgi
- igraph
- numpy
- matplotlib
- pandas

