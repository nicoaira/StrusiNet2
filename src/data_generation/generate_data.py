import argparse
import pandas as pd
import json
from datetime import datetime
import uuid
import os
from data_generation_utils import parallel_structure_generation, plot_triplets, split_dataset

def generate_metadata(args):
    """Generate metadata dictionary with all parameters and run info"""
    run_id = str(uuid.uuid4())
    
    # Convert args namespace to dictionary
    params_dict = vars(args)
    
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "parameters": params_dict
    }
    
    return metadata

def save_with_metadata(df, metadata, results_dir):
    """Save CSV with metadata header and separate metadata file"""
    os.makedirs(results_dir, exist_ok=True)
    base_path = os.path.join(results_dir, 'results')
    metadata_file = f"{base_path}_metadata.json"
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    header_comment = f"# Run ID: {metadata['run_id']}\n# Generated: {metadata['timestamp']}\n"
    
    out_file = f"{base_path}.csv"
    with open(out_file, 'w') as f:
        f.write(header_comment)
        df.to_csv(f, index=False)

def main():
    parser = argparse.ArgumentParser(description='RNA Structure Generator')
    
    # Sequence generation parameters
    seq_group = parser.add_argument_group('Sequence Generation')
    seq_group.add_argument('--num_structures', type=int, default=100000, help='Number of structures to generate')
    seq_group.add_argument('--seq_min_len', type=int, default=60, help='Minimum sequence length')
    seq_group.add_argument('--seq_max_len', type=int, default=600, help='Maximum sequence length')
    seq_group.add_argument('--seq_len_distribution', choices=['norm', 'unif'], default='unif', help='Distribution of sequence lengths')
    seq_group.add_argument('--seq_len_mean', type=int, default=24, help='Mean sequence length (for normal distribution)')
    seq_group.add_argument('--seq_len_sd', type=int, default=7, help='Standard deviation of sequence length (for normal distribution)')

    # Rearrangement parameters
    rearr_group = parser.add_argument_group('Rearrangement')
    rearr_group.add_argument('--variable_rearrangements', type=bool, default=True, help='Enable length-dependent rearrangements')
    rearr_group.add_argument('--norm_nt', type=int, default=100, help='Nucleotide count for rearrangement normalization')
    rearr_group.add_argument('--num_rearrangements', type=int, default=1, help='Number of rearrangement cycles per norm_nt')
    rearr_group.add_argument('--neg_len_variation', type=float, default=0.07, help='Maximum length variation for negative structures')

    # Stem modification parameters
    stem_group = parser.add_argument_group('Stem Modifications')
    stem_group.add_argument('--n_stem_indels', type=int, default=4, help='Number of stem modification cycles')
    stem_group.add_argument('--stem_min_size', type=int, default=3, help='Minimum stem size')
    stem_group.add_argument('--stem_max_n_modifications', type=int, default=10, help='Maximum modifications per stem')

    # Loop modification parameters
    loop_group = parser.add_argument_group('Loop Modifications')
    loop_group.add_argument('--n_hloop_indels', type=int, default=4, help='Number of hairpin loop modification cycles')
    loop_group.add_argument('--n_iloop_indels', type=int, default=4, help='Number of internal loop modification cycles')
    loop_group.add_argument('--n_bulge_indels', type=int, default=4, help='Number of bulge loop modification cycles')
    loop_group.add_argument('--n_mloop_indels', type=int, default=4, help='Number of multi loop modification cycles')
    
    # Loop size parameters
    loop_size_group = parser.add_argument_group('Loop Size Constraints')
    loop_size_group.add_argument('--hloop_min_size', type=int, default=4, help='Minimum hairpin loop size')
    loop_size_group.add_argument('--hloop_max_size', type=int, default=12, help='Maximum hairpin loop size')
    loop_size_group.add_argument('--iloop_min_size', type=int, default=3, help='Minimum internal loop size')
    loop_size_group.add_argument('--iloop_max_size', type=int, default=10, help='Maximum internal loop size')
    loop_size_group.add_argument('--bulge_min_size', type=int, default=1, help='Minimum bulge loop size')
    loop_size_group.add_argument('--bulge_max_size', type=int, default=12, help='Maximum bulge loop size')
    loop_size_group.add_argument('--mloop_min_size', type=int, default=1, help='Minimum multi loop size')
    loop_size_group.add_argument('--mloop_max_size', type=int, default=10, help='Maximum multi loop size')

    # Loop modification limits
    loop_mod_group = parser.add_argument_group('Loop Modification Limits')
    loop_mod_group.add_argument('--hloop_max_n_modifications', type=int, default=5, help='Maximum modifications per hairpin loop')
    loop_mod_group.add_argument('--iloop_max_n_modifications', type=int, default=5, help='Maximum modifications per internal loop')
    loop_mod_group.add_argument('--bulge_max_n_modifications', type=int, default=8, help='Maximum modifications per bulge loop')
    loop_mod_group.add_argument('--mloop_max_n_modifications', type=int, default=7, help='Maximum modifications per multi loop')

    # Performance parameters
    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers')
    perf_group.add_argument('--results_dir', type=str, required=True, help='Directory to save output files')

    # Add visualization parameters
    vis_group = parser.add_argument_group('Visualization')
    vis_group.add_argument('--plot_structures', action='store_true', default=False,
                          help='Generate structure plots')
    vis_group.add_argument('--num_plots', type=int, default=5,
                          help='Number of structure triplets to plot')

    # Add dataset splitting parameters
    split_group = parser.add_argument_group('Dataset Splitting')
    split_group.add_argument('--split', action='store_true', default=False, help='Enable dataset splitting')
    split_group.add_argument('--train_fraction', type=float, default=0.8, help='Fraction of data for training')
    split_group.add_argument('--val_fraction', type=float, default=0.2, help='Fraction of data for validation')
    
    args = parser.parse_args()
    
    # Handle train/val fraction calculations
    if args.split:
        if args.train_fraction is None and args.val_fraction is None:
            args.train_fraction = 0.8
            args.val_fraction = 0.2
        elif args.train_fraction is None:
            args.train_fraction = 1.0 - args.val_fraction
        elif args.val_fraction is None:
            args.val_fraction = 1.0 - args.train_fraction
            
        if abs(args.train_fraction + args.val_fraction - 1.0) > 1e-6:
            raise ValueError("Train and validation fractions must sum to 1.0")

    # Generate metadata
    metadata = generate_metadata(args)
    # Remove the redundant data_split field - it's already in parameters

    structure_triplets, sequence_triplets = parallel_structure_generation(
        num_structures=args.num_structures,
        num_workers=args.num_workers,
        seq_min_len=args.seq_min_len,
        seq_max_len=args.seq_max_len,
        seq_len_distribution=args.seq_len_distribution,
        seq_len_mean=args.seq_len_mean,
        seq_len_sd=args.seq_len_sd,
        variable_rearrangements=args.variable_rearrangements,
        norm_nt=args.norm_nt,
        num_rearrangements=args.num_rearrangements,
        n_stem_indels=args.n_stem_indels,
        n_hloop_indels=args.n_hloop_indels,
        n_iloop_indels=args.n_iloop_indels,
        n_bulge_indels=args.n_bulge_indels,
        n_mloop_indels=args.n_mloop_indels,
        neg_len_variation=args.neg_len_variation,
        stem_min_size=args.stem_min_size,
        stem_max_n_modifications=args.stem_max_n_modifications,
        hloop_min_size=args.hloop_min_size,
        hloop_max_size=args.hloop_max_size,
        iloop_min_size=args.iloop_min_size,
        iloop_max_size=args.iloop_max_size,
        bulge_min_size=args.bulge_min_size,
        bulge_max_size=args.bulge_max_size,
        mloop_min_size=args.mloop_min_size,
        mloop_max_size=args.mloop_max_size,
        hloop_max_n_modifications=args.hloop_max_n_modifications,
        iloop_max_n_modifications=args.iloop_max_n_modifications,
        bulge_max_n_modifications=args.bulge_max_n_modifications,
        mloop_max_n_modifications=args.mloop_max_n_modifications
    )

    df = pd.DataFrame(columns=['structure_A', 'structure_P', 'structure_N', 'sequence_A', 'sequence_P', 'sequence_N'])
    df[['structure_A', 'structure_P', 'structure_N']] = pd.DataFrame(structure_triplets)
    df[['sequence_A', 'sequence_P', 'sequence_N']] = pd.DataFrame(sequence_triplets)

    # Add unique ID to each row
    df.insert(0, 'id', range(1, len(df) + 1))

    os.makedirs(args.results_dir, exist_ok=True)
    base_path = os.path.join(args.results_dir, 'triplets_dataset')
    metadata_file = f"{base_path}_metadata.json"
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create header comment for CSV files
    header_comment = f"# Run ID: {metadata['run_id']}\n# Generated: {metadata['timestamp']}\n"

    if args.split:
        train_df, val_df = split_dataset(df, args.train_fraction, args.val_fraction)
        
        # Save train file with header comment
        with open(f"{base_path}_train.csv", 'w') as f:
            f.write(header_comment)
            train_df.to_csv(f, index=False)
            
        # Save validation file with header comment
        with open(f"{base_path}_val.csv", 'w') as f:
            f.write(header_comment)
            val_df.to_csv(f, index=False)
        
        # Plot structures if requested
        if args.plot_structures:
            train_plots_dir = os.path.join(args.results_dir, 'plots', 'train')
            val_plots_dir = os.path.join(args.results_dir, 'plots', 'val')
            os.makedirs(train_plots_dir, exist_ok=True)
            os.makedirs(val_plots_dir, exist_ok=True)
            plot_triplets(train_df, train_plots_dir, num_samples=args.num_plots)
            plot_triplets(val_df, val_plots_dir, num_samples=args.num_plots)
    else:
        # Save single file with header comment
        with open(f"{base_path}.csv", 'w') as f:
            f.write(header_comment)
            df.to_csv(f, index=False)
        
        # Plot structures if requested
        if args.plot_structures:
            plots_dir = os.path.join(args.results_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            plot_triplets(df, plots_dir, num_samples=args.num_plots)

    print(f"Results saved in {args.results_dir}")

if __name__ == "__main__":
    main()