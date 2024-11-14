# Number of structures to be generated:
num_structures = 100000 # integer

# Size range for generated sequences:
seq_min_len = 60 # integer
seq_max_len = 600 # integer

# Distribution of generated sequence lenghts (normal/uniform):
seq_len_distribution = "unif" # ["norm", "unif"]

# Sequence lenght mean (only for normal distribution):
seq_len_mean = 24 # integer

# Sequence lenght std (only for normal distribution):
seq_len_sd = 7 # integer

# Number of rearregements cycles depends on sequence length:
variable_rearrangements = True # boolean

# Number of nt to normalize the number of rearrangement cycles (ignored if `variable_rearrangements` = False):
norm_nt = 100 # integer

# Number of rearrangement cycles per `norm_nt` nt. If `variable_rearrangements` = False, then this will be a fixed number of rearrangements:
num_rearrangements = 1 # integer

# max % of length variation (respect to anchor) of negative structures:
neg_len_variation = 7 # integer
neg_len_variation /= 100


# Number of threads:
num_workers = 16 # integer

# Filename for the output csv:
out_file = 'dot_bracket_dummy_RNPs_triplets_240913.csv' # string

if seq_len_distribution == 'norm':
    assert(seq_min_len < seq_len_mean < seq_max_len)



# Stem indels parameters

# Number of stems modification cycles:
n_stem_indels = 4 # integer

# Minimum possible size of stems before/after modification:
stem_min_size = 3 # integer

# Maximum amount of modifications in one stem:
stem_max_n_modifications = 10 # integer


# Bulges indels parameters

# Number of bulge loops modification cycles:
n_bulge_indels = 4 # integer

# Minimum possible size of bulge loops before/after modification:
bulge_min_size = 1 # integer

# Maximum possible size of bulge loops before/after modification:
bulge_max_size = 12 # integer

# Maximum amount of modifications in one bulge loop:
bulge_max_n_modifications = 8 # integer



# Hairpin loops indels parameters

# Number of hairpin loops modification cycles:
n_hloop_indels = 4 # integer

# Minimum possible size of hairpin loops before/after modification:
hloop_min_size = 4 # integer

# Maximum possible size of hairpin loops before/after modification:
hloop_max_size = 12 # integer

# Maximum amount of modifications in one hairpin loop:
hloop_max_n_modifications = 5 # integer




# Internal loops indels parameters

# Number of internal loops modification cycles:
n_iloop_indels = 4 # integer

# Minimum possible size of internal loops (for each side) before/after modification:
iloop_min_size = 3 # integer

# Maximum possible size of internal loops (for each side) before/after modification:
iloop_max_size = 10 # integer

# Maximum amount of modifications (counting both sides) in one internal loop:
iloop_max_n_modifications = 5 # integer




# Multiloops indels parameters

# Number of multi loops modification cycles:
n_mloop_indels = 4 # integer

# Minimum possible size of multi loops before/after modification:
mloop_min_size = 1 # integer

# Maximum possible size of multi loops before/after modification:
mloop_max_size = 10 # integer

# Maximum amount of modifications in one multi loop:
mloop_max_n_modifications = 7 # integer