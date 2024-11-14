import params
from data_generation.data_generation_utils import parallel_structure_generation
import argparse
import pandas as pd


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', type=str, default='dot_bracket_dummy_RNPs_triplets_240913.csv', help='Output path of the dataset.')
    args = parser.parse_args()


    structure_triplets, sequence_triplets  = parallel_structure_generation(
        num_structures = params.num_structures,
        num_workers = params.num_workers,
        seq_min_len = params.seq_min_len,
        seq_max_len = params.seq_max_len,
        seq_len_distribution = params.seq_len_distribution,
        seq_len_mean = params.seq_len_mean,
        seq_len_sd = params.seq_len_sd,
        variable_rearrangements = params.variable_rearrangements,
        norm_nt = params.norm_nt,
        num_rearrangements = params.num_rearrangements,
        n_stem_indels = params.n_stem_indels,
        n_hloop_indels = params.n_hloop_indels,
        n_iloop_indels = params.n_iloop_indels,
        n_bulge_indels = params.n_bulge_indels,
        n_mloop_indels = params.n_mloop_indels,
        neg_len_variation = params.neg_len_variation,
        stem_min_size = params.stem_min_size,
        stem_max_n_modifications = params.stem_max_n_modifications,
        hloop_min_size = params.hloop_min_size, 
        hloop_max_size = params.hloop_max_size,
        iloop_min_size = params.iloop_min_size,
        iloop_max_size = params.iloop_max_size,
        bulge_min_size = params.bulge_min_size,
        bulge_max_size = params.bulge_max_size,
        mloop_min_size = params.mloop_min_size,
        mloop_max_size = params.mloop_max_size,
        hloop_max_n_modifications = params.hloop_max_n_modifications, 
        iloop_max_n_modifications = params.iloop_max_n_modifications,
        bulge_max_n_modifications = params.bulge_max_n_modifications,
        mloop_max_n_modifications = params.mloop_max_n_modifications
    )

    df = pd.DataFrame(columns=['structure_A', 'structure_P', 'structure_N', 'sequence_A', 'sequence_P', 'sequence_N'])

    # Assign values to the DataFrame
    df[['structure_A', 'structure_P', 'structure_N']] = pd.DataFrame(structure_triplets)

    df[['sequence_A', 'sequence_P', 'sequence_N']] = pd.DataFrame(sequence_triplets)

    # Write the DataFrame to a CSV file
    print(args.out_file, 'created!')
    df.to_csv(args.out_file, index=False)



if __name__ == "__main__":
    main()