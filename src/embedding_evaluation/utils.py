from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, f1_score
import seaborn as sns
from minineedle import needle
from scipy.stats import ttest_ind

def cos_similarity(emb_1, emb_2):

  cos_similarity = F.cosine_similarity(emb_1, emb_2, axis=1)

  return cos_similarity.item()


def square_dist(emb1, emb2):
    """
    Compute the squared distance between two embeddings using PyTorch.

    Args:
    - emb1 (torch.Tensor): The first embedding tensor.
    - emb2 (torch.Tensor): The second embedding tensor.

    Returns:
    - distance (float): The squared distance between the two embeddings.
    """
    return torch.sum((emb1 - emb2) ** 2, dim=1)


def euclidean_dist(emb1, emb2):
    """
    Compute the Euclidean distance between two embedding vectors in a row of the dataframe.

    Args:
    - emb1 (torch.Tensor): The first embedding tensor.
    - emb2 (torch.Tensor): The second embedding tensor.

    Returns:
    - distance (float): The Euclidean distance between the two embeddings.
    """

    # Compute the Euclidean distance using torch.norm
    return torch.norm(emb1 - emb2).item()


def get_corr_matrix(df, metrics=['cosine_similarity',
                                 'square_distance',
                                 'euclidean_distance']):

  return df[metrics].corr()


def length_average(A, B):
    len_A = len(A)
    len_B = len(B)
    average = (len_A + len_B) / 2

    return average

def relative_difference(A, B):
    len_A = len(A)
    len_B = len(B)
    average = (len_A + len_B) / 2
    relative_difference = abs(len_A - len_B) / average * 100

    return relative_difference



def plot_ROC(binary_pair_type, score):
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(binary_pair_type, score)
    roc_auc = auc(fpr, tpr)

    print('AUC =', str(round(roc_auc, 2)), '\n')



    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def compute_f1_score(binary_pair_type, score, threshold):
  y_pred = (score > threshold).astype(int)
  f1 = f1_score(binary_pair_type, y_pred)

  return round(f1, 4)


def plot_distribution(x, df, pairing_type='pair_type2'):

    plt.figure(figsize=(12, 6))
    xlim = df[x].max()
    binwidth = xlim/100
    plt.xlim(0, xlim)
    title = 'Frequency Distribution of ' + x.capitalize()
    sns.set(style="whitegrid")

    # Create a histogram with colored bins based on pair_type
    g = sns.histplot(data=df, x=x, hue=pairing_type, element='step', stat='density', common_norm=False, binwidth=binwidth)

    # Set plot labels and title
    g.set(xlabel=x.capitalize(), ylabel='Frequency', title=title)

    # Move the legend outside the plot
    sns.move_legend( g, "upper left", bbox_to_anchor=(1, 1), title='Pair type')

    # Display the plot
    plt.show()


def annotate_pairs_rfam(row):
  if row['rfam_1'] == row['rfam_2']:
    return row['rfam_1']
  else:
    return 'Different Rfam'

def annotate_pairs_rfam_bin(row):
  if row['rfam_1'] == row['rfam_2']:
    return 'Same Rfam'
  else:
    return 'Different Rfam'

def annotate_pairs_rna_type(row):
  if row['rna_type_1'] == row['rna_type_2']:
    return row['rna_type_1']
  else:
    return 'Different RNA type'

def annotate_pairs_rna_type_bin(row):
  if row['rna_type_1'] == row['rna_type_2']:
    return 'Same RNA type'
  else:
    return 'Different RNA type'

def get_alignment_score(seq1, seq2):
  seq_pair = needle.NeedlemanWunsch(seq1, seq2)
  seq_pair.align()

  return seq_pair.get_score()


def model_stadistics(df, metric, metric_type, RNA_type, pairing_type='pair_type2', f1_threshold=0.5):

  target_pair = RNA_type+'-'+RNA_type

  target_metric_serie = df.loc[
      df.pair_type ==  target_pair, metric]

  random_metric_serie = df.loc[
      df.pair_type ==  'random', metric]

  target_metric_mean = target_metric_serie.mean()
  random_metric_mean = random_metric_serie.mean()

  print(target_pair + ' pairs mean ' + metric + ' =', round(target_metric_mean, 3))
  print('Random pairs mean'  + metric + ' =', round(random_metric_mean, 3))

  # Perform the t-test
  t_statistic, p_value = ttest_ind(target_metric_serie,
                                  random_metric_serie,
                                  equal_var=False)

  if p_value == 0:
    print(f"P-value < 1e-324")
  else:
    print(f"P-value = {p_value}")


  if metric_type == 'similarity':
    score = df[metric]
    use_f1_threshold = f1_threshold
  elif metric_type == 'distance':
    score = 1/df[metric]
    use_f1_threshold = 1/f1_threshold

  else:
    raise ValueError("Invalid metric_type")

  f1 = compute_f1_score(df['binary_pair_type'],
                      score,
                      threshold = use_f1_threshold)

  print('F1 Score =', f1, '(threshold = '+str(f1_threshold)+')')

  plot_ROC(df['binary_pair_type'], score)

  print('\n')

  plot_distribution(x=metric, df=df, pairing_type=pairing_type)

