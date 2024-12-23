# RNA Structure Data Generation Module Documentation

## Overview

This module generates synthetic RNA structure-sequence pairs for training machine learning models. It creates triplets of RNA structures consisting of:
- Anchor (A): Original valid RNA structure
- Positive (P): Modified version of anchor that maintains structural validity 
- Negative (N): Modified version that differs significantly from anchor

## Key Components

### 1. Data Generation Pipeline
- Generates random RNA sequences
- Predicts minimum free energy structures
- Performs structural modifications through:
  - Stem insertions/deletions
  - Loop modifications (hairpin, internal, bulge, multi-loop)
  - Rearrangements of structural elements

### 2. Core Functions
- `parallel_structure_generation`: Main function orchestrating parallel generation
- `stem_indels`: Handles stem modifications
- `hairpin_loop_indels`: Modifies hairpin loops
- `internal_loop_indels`: Handles internal loop changes
- `bulge_indels`: Manages bulge modifications
- `multi_loop_indels`: Controls multi-loop alterations

## Parameters Configuration

### Sequence Generation Parameters
- `num_structures`: Total number of triplets to generate (default: 100000)
- `seq_min_len`: Minimum sequence length (default: 60)
- `seq_max_len`: Maximum sequence length (default: 600)
- `seq_len_distribution`: Distribution type for lengths ("unif" or "norm")

### Structural Modification Parameters
1. **Stem Parameters**
   - `n_stem_indels`: Number of stem modification cycles (default: 4)
   - `stem_min_size`: Minimum stem size (default: 3)
   - `stem_max_n_modifications`: Maximum modifications per stem (default: 10)

2. **Loop Parameters**
   - Hairpin, Internal, Bulge, and Multi-loop modifications
   - Each has parameters for:
     - Number of modification cycles
     - Size constraints
     - Maximum modifications allowed

### Performance Parameters
- `num_workers`: Number of parallel threads (default: 16)
- `variable_rearrangements`: Enable length-dependent rearrangements
- `norm_nt`: Normalization factor for rearrangements (default: 100)

## Algorithm Workflow

1. **Sequence Generation**
   - Generate random RNA sequence within length constraints
   - Calculate minimum free energy structure

2. **Structure Validation**
   - Check for minimum stem count
   - Verify structure complexity

3. **Modification Process**
   - Apply stem modifications
   - Modify different types of loops
   - Ensure structural validity after each change

4. **Output Generation**
   - Create structure-sequence triplets
   - Save to CSV format with columns:
     - structure_A, structure_P, structure_N
     - sequence_A, sequence_P, sequence_N

## Usage

```python
from data_generation_utils import parallel_structure_generation
import params

# Generate structures
structures, sequences = parallel_structure_generation(
    num_structures=params.num_structures,
    num_workers=params.num_workers,
    seq_min_len=params.seq_min_len,
    seq_max_len=params.seq_max_len
)
```

