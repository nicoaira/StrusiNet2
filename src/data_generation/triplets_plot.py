import tempfile
import random
import matplotlib as plt
import forgi
import forgi.visual.mplotlib as fvm

def plot_rna_structure(df, structure_name, random_sample, triplet_letter):
    sequence = df.iloc[random_sample]['sequence_' + triplet_letter]
    structure = df.iloc[random_sample]['structure_' + triplet_letter]

    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp_file:
        # Write the strings to the temporary file
        temp_file.write('>' + structure_name + '\n')
        temp_file.write(sequence + '\n')
        temp_file.write(structure + '\n')

        # Make sure to flush to ensure all data is written
        temp_file.flush()

        # Use the temporary file as input to another function
        temp_file.seek(0)  # Move to the beginning of the file before reading

        with open(temp_file.name, 'r') as file:

            cg = forgi.load_rna(temp_file.name, allow_many=False)

            fvm.plot_rna(cg, text_kwargs={"fontweight":"black"}, lighten=0.7,
                        backbone_kwargs={"linewidth":3})
            plt.title('Structure #' +str(random_sample) + ' ' + structure_name)

            plt.show()

def plot_triplets(df, num_samples = 5):
    for i in range(num_samples):

        random_sample = random.randint(0, df.shape[0]-1)
        plt.rcParams['figure.figsize'] = [16.0, 12.0]

        plot_rna_structure("Anchor", random_sample, 'A')
        plot_rna_structure("Positive", random_sample, 'P')
        plot_rna_structure("Negative", random_sample, 'N')


        if i < num_samples -1:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')