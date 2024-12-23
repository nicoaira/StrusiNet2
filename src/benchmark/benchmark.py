import pandas as pd
import subprocess
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import argparse
import os
from datetime import datetime
import pytz
import shutil
import json
import re
import platform
import time
import sys

def check_device(verbose=False):
    """
    Check if CUDA is available. If it is, return "cuda". Otherwise, return "cpu".
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to print out the type of device found. Defaults to False.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_model = torch.cuda.get_device_name(0)
        if verbose:
            print(f"GPU is available. Model: {gpu_model}")
    else:
        device = "cpu"
        if verbose:
            print(f"No GPU found, using CPU...")

    return device

def get_time(timezone='Europe/Madrid'):
    """
    Get the current time in the specified timezone and format it as '-YYMMDD-HHMMSS'.

    Parameters
    ----------
    timezone : str, optional
        Timezone to use. Default is 'Europe/Madrid'.

    Returns
    -------
    str
        The current time as a string in '-YYMMDD-HHMMSS' format.
    """
    geo_tz = pytz.timezone(timezone)
    geo_time = datetime.now(geo_tz)
    formatted_time = geo_time.strftime('-%y%m%d-%H%M%SS')
    return formatted_time

def get_embeddings(model_script,
                   sampled_rnas_path,
                   emb_output_path,
                   model_weights_path,
                   model_type,
                   gin_layers,
                   graph_encoding,
                   structure_column_name,
                   structure_column_num,
                   header):
    """
    Generate embeddings by running the model script as a subprocess.

    Parameters
    ----------
    model_script : str
        Path to the script that generates embeddings.
    sampled_rnas_path : str
        Path to the input file containing RNA sequences and structures.
    emb_output_path : str
        Path where the embeddings output file will be saved.
    model_weights_path : str
        Path to the model weights.
    model_type: str
        Model that's being evaluated: siamese or gin
    gin_layers: int
        Optional number of gin layers.
    graph_encoding: str
        Type of gin encoding, forgi or allocator
    structure_column_name : str
        Name of the column containing RNA secondary structures, if provided.
    structure_column_num : int
        Column index of the RNA secondary structures if name is not provided.
    header : bool
        Whether the input file has a header.

    Raises
    ------
    FileNotFoundError
        If the model script is not found.
    subprocess.CalledProcessError
        If the subprocess call to generate embeddings fails.
    """
    if not os.path.isfile(model_script):
        raise FileNotFoundError(f"Model script '{model_script}' not found.")

    device = check_device(verbose=True)

    command = [
        "python", model_script,
        "--input", sampled_rnas_path,
        "--output", emb_output_path,
        "--model_path", model_weights_path,
        "--model_type", model_type,
        "--device", device,
        "--header", str(header)
    ]

    if gin_layers is not None:
        command.extend(["--gin_layers", str(gin_layers)])
    
    if graph_encoding is not None:
        command.extend(["--graph_encoding", graph_encoding])

    if structure_column_name:
        command.extend(["--structure_column_name", structure_column_name])
    elif structure_column_num is not None:
        command.extend(["--structure_column_num", str(structure_column_num)])
    else:
        command.extend(["--structure_column_name", "secondary_structure"])

    try:
        print("Command to execute:", ' '.join(command))
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred while running the model script!")
        print(e)

def calculate_square_distance(id1, id2, embedding_dict):
    """
    Calculate the squared Euclidean distance between two embeddings.

    Parameters
    ----------
    id1 : str
        First RNAcentral ID.
    id2 : str
        Second RNAcentral ID.
    embedding_dict : dict
        Dictionary mapping RNAcentral IDs to their embedding vectors (numpy arrays).

    Returns
    -------
    float
        The squared distance, or NaN if any embedding is missing.
    """
    vector1 = embedding_dict.get(id1)
    vector2 = embedding_dict.get(id2)
    if vector1 is not None and vector2 is not None:
        return np.sum((vector1 - vector2) ** 2)
    else:
        return np.nan

def count_initial_comment_lines(file_path):
    """
    Count how many initial lines in the given file start with '#' so we can skip them
    when reading the file with pandas without using comment='#'.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    int
        Number of initial lines that start with '#'.
    """
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                count += 1
            else:
                break
    return count

def load_benchmark_dataset(benchmark_path, expected_id):
    """
    Load the benchmark dataset, skipping commented lines at the top, and verify the Benchmark ID
    against the expected ID from the metadata.

    Parameters
    ----------
    benchmark_path : str
        Path to the benchmark dataset file.
    expected_id : str
        The expected benchmark ID (from the JSON metadata) to verify.

    Returns
    -------
    pd.DataFrame
        The loaded benchmark dataset.

    Raises
    ------
    ValueError
        If the benchmark ID in the file does not match the expected ID or is not found.
    """
    benchmark_id_pattern = re.compile(r'^#\s*Benchmark ID:\s*([A-Za-z0-9]+)')
    file_id = None
    skiprows = 0
    with open(benchmark_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                skiprows += 1
                match = benchmark_id_pattern.search(line)
                if match:
                    file_id = match.group(1)
            else:
                break

    if file_id is None:
        raise ValueError(f"No Benchmark ID found in {benchmark_path}.")
    if file_id != expected_id:
        raise ValueError(f"Benchmark ID mismatch in {benchmark_path}! Expected: {expected_id}, Found: {file_id}")

    benchmark_df = pd.read_csv(benchmark_path, sep='\t', skiprows=skiprows)
    return benchmark_df

def get_distances(embedding_dict,
                  benchmark_path,
                  benchmark_name,
                  benchmark_version,
                  benchmarking_results_path,
                  expected_id,
                  save_distances=False,
                  no_save=False):
    """
    Calculate the square distances between embeddings for all pairs in the benchmark dataset,
    and optionally save the results.

    Parameters
    ----------
    embedding_dict : dict
        RNAcentral ID -> embedding vector.
    benchmark_path : str
        Path to the benchmark dataset.
    benchmark_name : str
        Name of the benchmark.
    benchmark_version : str
        Version of the benchmark.
    benchmarking_results_path : str
        Directory for results.
    expected_id : str
        Expected benchmark ID.
    save_distances : bool
        Whether to save distances.
    no_save : bool
        If True, do not save any outputs.

    Returns
    -------
    pd.DataFrame
        Benchmark dataframe with 'square_distance' column.
    """
    benchmark_df = load_benchmark_dataset(benchmark_path, expected_id)

    square_distances = []
    for _, row in tqdm(benchmark_df.iterrows(), total=benchmark_df.shape[0],
                       desc=f"Calculating distances for {benchmark_name} (v{benchmark_version})"):
        distance = calculate_square_distance(row['rnacentral_id_1'], row['rnacentral_id_2'], embedding_dict)
        square_distances.append(distance)

    benchmark_df['square_distance'] = square_distances

    if save_distances and (not no_save):
        benchmark_version_f = 'v'+'_'.join(str(benchmark_version).split('.'))
        benchmark_w_dist_file = benchmark_name + '_' + benchmark_version_f + '_w_dist.tsv'
        benchmark_w_dist_path = os.path.join(benchmarking_results_path, benchmark_w_dist_file)
        benchmark_df.to_csv(benchmark_w_dist_path, sep='\t', index=False)

    #return benchmark_df
    return benchmark_df[benchmark_df.square_distance.notna()]

def get_roc_auc(benchmark_name, benchmark_version,
                benchmark_df, target,
                benchmarking_results_path,
                skip_barplot=False, skip_auc_curve=False,
                no_save=False):
    """
    Calculate ROC AUC scores and optionally generate plots.

    Parameters
    ----------
    benchmark_name : str
    benchmark_version : str
    benchmark_df : pd.DataFrame
    target : str
        Target column.
    benchmarking_results_path : str
    skip_barplot : bool
    skip_auc_curve : bool
    no_save : bool

    Returns
    -------
    dict
        A dictionary with AUC results and average AUC.
    """
    unique_rna_types = benchmark_df['rna_type_1'].unique()
    auc_results = {}

    for rna_type in unique_rna_types:
        rna_type_df = benchmark_df[benchmark_df['rna_type_1'] == rna_type]
        y_true = rna_type_df[target]
        y_scores = -rna_type_df['square_distance']

        if len(y_true.unique()) > 1:
            auc_val = roc_auc_score(y_true, y_scores)
        else:
            auc_val = float('nan')
        auc_results[rna_type] = auc_val

    average_auc = np.nanmean(list(auc_results.values()))

    print("Benchmark: " + benchmark_name)
    for rna_type, auc_val in auc_results.items():
        print(f"RNA Type: {rna_type}, AUC: {auc_val:.4f}")
    print(f"Average AUC across all RNA types: {average_auc:.4f}")

    if no_save:
        return {"auc_results": auc_results, "average_auc": average_auc}

    plot_dir = os.path.join(benchmarking_results_path, "plots")
    if not os.path.exists(plot_dir) and (not skip_barplot or not skip_auc_curve):
        os.makedirs(plot_dir)

    benchmark_version_f = 'v'+'_'.join(str(benchmark_version).split('.'))

    if not skip_barplot:
        plt.figure(figsize=(10, 6))
        plt.bar(auc_results.keys(), auc_results.values())
        plt.xlabel('RNA Type')
        plt.ylabel('AUC')
        plt.title('AUC by RNA Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        benchmark_plot_filename = 'AUC_benchmark_' + benchmark_name + '-' + benchmark_version_f + '.png'
        plt.savefig(os.path.join(plot_dir, benchmark_plot_filename), dpi=300)
        plt.close()

    if not skip_auc_curve:
        n_types = len(unique_rna_types)
        n_cols = 3
        n_rows = int(np.ceil(n_types / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_types == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, rna_type in enumerate(unique_rna_types):
            ax = axes[idx]
            rna_type_df = benchmark_df[benchmark_df['rna_type_1'] == rna_type]
            y_true = rna_type_df[target]
            y_scores = -rna_type_df['square_distance']

            if len(y_true.unique()) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc_val = roc_auc_score(y_true, y_scores)

                ax.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}")
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax.set_title(f'RNA Type: {rna_type}')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(loc="lower right")
            else:
                ax.set_title(f'RNA Type: {rna_type}')
                ax.text(0.5, 0.5, 'Only one class present', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])

        for idx in range(n_types, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        benchmark_plot_filename = 'ROC_curves_benchmark_' + benchmark_name + '-' + benchmark_version_f + '.png'
        plt.savefig(os.path.join(plot_dir, benchmark_plot_filename), dpi=300)
        plt.close()

    return {"auc_results": auc_results, "average_auc": average_auc}

def cleanup(files=[], directories=[]):
    """
    Remove specified temporary files and directories.

    Parameters
    ----------
    files : list of str, optional
        List of file paths to remove.
    directories : list of str, optional
        List of directory paths to remove.
    """
    if len(files) == 0 and len(directories) == 0:
        return
    else:
        print('Removing temporary directories and files...')
        for d in directories:
            if os.path.exists(d):
                shutil.rmtree(d)
        for f in files:
            if os.path.exists(f):
                os.remove(f)

def parse_benchmarks(benchmark_args, benchmark_metadata):
    """
    Parse the benchmark dataset arguments and select appropriate entries.
    If only a name is provided, select latest version.
    If name-vX is provided, select that version.

    Parameters
    ----------
    benchmark_args : list of str
    benchmark_metadata : dict

    Returns
    -------
    list of dict
    """
    name_dict = {}
    for entry in benchmark_metadata["benchmark_datasets"]:
        name = entry["name"]
        version = entry["version"]
        if name not in name_dict:
            name_dict[name] = []
        name_dict[name].append(entry)

    for k in name_dict:
        name_dict[k].sort(key=lambda x: x['version'], reverse=False)

    selected_benchmarks = []

    for arg in benchmark_args:
        if '-v' in arg:
            match = re.match(r"^(.*)-v(\d+)$", arg)
            if not match:
                raise ValueError(f"Invalid benchmark argument format: {arg}")
            b_name = match.group(1)
            b_version = int(match.group(2))

            if b_name not in name_dict:
                raise ValueError(f"No benchmark named '{b_name}' found in metadata.")

            found = False
            for entry in name_dict[b_name]:
                if entry["version"] == b_version:
                    selected_benchmarks.append(entry)
                    found = True
                    break
            if not found:
                raise ValueError(f"No version '{b_version}' found for benchmark '{b_name}'.")
        else:
            if arg not in name_dict:
                raise ValueError(f"No benchmark named '{arg}' found in metadata.")
            latest_entry = name_dict[arg][-1]
            selected_benchmarks.append(latest_entry)

    return selected_benchmarks

def check_required_files(model_script,
                         model_weights_path,
                         benchmark_metadata_path,
                         datasets_dir,
                         selected_benchmarks,
                         benchmark_metadata):
    """
    Check all required files exist.

    Parameters
    ----------
    model_script, model_weights_path, benchmark_metadata_path : str
    datasets_dir : str
    selected_benchmarks : list of dict
    benchmark_metadata : dict

    Raises
    ------
    FileNotFoundError, ValueError
    """
    if not os.path.isfile(model_script):
        raise FileNotFoundError(f"Model script not found: {model_script}")

    if not os.path.isfile(model_weights_path):
        raise FileNotFoundError(f"Model weights file not found: {model_weights_path}")

    if not os.path.isfile(benchmark_metadata_path):
        raise FileNotFoundError(f"Benchmark metadata JSON not found: {benchmark_metadata_path}")

    needed_primary_ids = set([bm["primary_sampled_dataset_id"] for bm in selected_benchmarks])
    primary_map = {p["unique_id"]: p for p in benchmark_metadata["primary_sampled_datasets"]}

    for pid in needed_primary_ids:
        if pid not in primary_map:
            raise ValueError(f"No primary sampled dataset found for ID: {pid}")
        primary_filename = primary_map[pid]["filename"]
        primary_path = os.path.join(datasets_dir, primary_filename)
        if not os.path.isfile(primary_path):
            raise FileNotFoundError(f"Primary sampled dataset not found: {primary_path}")

    for bm in selected_benchmarks:
        bm_filename = bm["filename"]
        bm_path = os.path.join(datasets_dir, bm_filename)
        if not os.path.isfile(bm_path):
            raise FileNotFoundError(f"Benchmark dataset not found: {bm_path}")

def log_information(log_path, info_dict):
    """
    Log information to a specified log file.

    Parameters
    ----------
    log_path : str
        Path to the log file.
    info_dict : dict
        Dictionary with information to log.
    """
    with open(log_path, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        for key, value in info_dict.items():
            f.write(f"{key}: {value}\n")

def run_benchmark(model_script,
                  benchmark_datasets,
                  benchmark_metadata,
                  benchmark_metadata_path,
                  datasets_dir,
                  save_embeddings,
                  emb_output_path,
                  model_weights_path,
                  model_type,
                  gin_layers,
                  graph_encoding,
                  structure_column_name,
                  structure_column_num,
                  header,
                  skip_barplot,
                  skip_auc_curve,
                  results_path,
                  save_distances,
                  no_save,
                  only_needed_embeddings,
                  no_log):
    """
    Main function to run the benchmark with logging support.

    Logging includes:
    - Date and time
    - Command run
    - Platform characteristics (OS, Python version, GPU)
    - Datasets used
    - Timing info for embeddings and distances
    - Benchmark AUC results

    Parameters
    ----------
    model_script, benchmark_datasets, benchmark_metadata, benchmark_metadata_path : str / list
    datasets_dir : str
    save_embeddings, skip_barplot, skip_auc_curve, save_distances, no_save, only_needed_embeddings, no_log : bool
    emb_output_path, model_weights_path, structure_column_name, structure_column_num : various
    header : bool
    results_path : str
    """
    start_time = time.time()
    bm_start_time = get_time(timezone='Europe/Madrid')

    selected_benchmarks = parse_benchmarks(benchmark_datasets, benchmark_metadata)

    primary_map = {p["unique_id"]: p for p in benchmark_metadata["primary_sampled_datasets"]}
    needed_primary_ids = set([bm["primary_sampled_dataset_id"] for bm in selected_benchmarks])

    check_required_files(
        model_script=model_script,
        model_weights_path=model_weights_path,
        benchmark_metadata_path=benchmark_metadata_path,
        datasets_dir=datasets_dir,
        selected_benchmarks=selected_benchmarks,
        benchmark_metadata=benchmark_metadata
    )

    # Collect needed IDs if only_needed_embeddings
    primary_datasets_needed_ids = {pid: set() for pid in needed_primary_ids}
    if only_needed_embeddings:
        for bm in selected_benchmarks:
            benchmark_filename = bm["filename"]
            benchmark_id = bm["id"]
            benchmark_path = os.path.join(datasets_dir, benchmark_filename)
            bm_df = load_benchmark_dataset(benchmark_path, benchmark_id)
            pid = bm["primary_sampled_dataset_id"]
            primary_datasets_needed_ids[pid].update(bm_df['rnacentral_id_1'].unique())
            primary_datasets_needed_ids[pid].update(bm_df['rnacentral_id_2'].unique())

    if not no_save:
        benchmarking_results_path = results_path + bm_start_time
        if not os.path.exists(benchmarking_results_path):
            os.makedirs(benchmarking_results_path)
    else:
        benchmarking_results_path = results_path

    # Set up logging
    log_path = None
    if not no_log and not no_save:
        log_path = os.path.join(benchmarking_results_path, 'benchmark.log')

    # Log basic info
    if log_path:
        log_info = {
            "Date and Time": str(datetime.now()),
            "Command Run": " ".join(sys.argv),
            "Platform": platform.platform(),
            "Python Version": platform.python_version(),
            "CUDA Available": str(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            log_info["GPU"] = torch.cuda.get_device_name(0)
        else:
            log_info["GPU"] = "No GPU"
        log_information(log_path, log_info)

    primary_embeddings_map = {}

    # Embedding generation timing
    embedding_start = time.time()

    for pid in needed_primary_ids:
        primary_info = primary_map[pid]
        primary_filename = primary_info["filename"]
        primary_path = os.path.join(datasets_dir, primary_filename)
        skiprows = count_initial_comment_lines(primary_path)

        if not no_save:
            if save_embeddings:
                emb_output_dir = os.path.join(benchmarking_results_path, "embeddings_" + pid)
                if not os.path.exists(emb_output_dir):
                    os.makedirs(emb_output_dir)
            else:
                emb_output_dir = "tmp"
                if not os.path.exists(emb_output_dir):
                    os.makedirs(emb_output_dir)
        else:
            emb_output_dir = "tmp"
            if not os.path.exists(emb_output_dir):
                os.makedirs(emb_output_dir)

        if only_needed_embeddings:
            needed_ids = primary_datasets_needed_ids[pid]
            primary_df = pd.read_csv(primary_path, sep='\t', header=0 if header else None, skiprows=skiprows)
            if 'rnacentral_id' not in primary_df.columns:
                raise ValueError("Primary dataset does not have 'rnacentral_id' column.")
            filtered_df = primary_df[primary_df['rnacentral_id'].isin(needed_ids)]
            temp_filtered_path = os.path.join(emb_output_dir, primary_info["name"] + '_filtered' + bm_start_time + '.tsv')
            filtered_df.to_csv(temp_filtered_path, sep='\t', index=False)
            embedding_input_path = temp_filtered_path
            expected_ids = set(filtered_df['rnacentral_id'].unique())
            n_rows = filtered_df.shape[0]
        else:
            full_df = pd.read_csv(primary_path, sep='\t', header=0 if header else None, skiprows=skiprows)
            if 'rnacentral_id' not in full_df.columns:
                raise ValueError("Primary dataset does not have 'rnacentral_id' column.")
            embedding_input_path = primary_path
            expected_ids = set(full_df['rnacentral_id'].unique())
            n_rows = full_df.shape[0]

        emb_filename = '/' + primary_info["name"] + '_w_emb' + bm_start_time + '.tsv'
        curr_emb_output_path = emb_output_dir + emb_filename

        emb_gen_start = time.time()
        get_embeddings(
            model_script=model_script,
            sampled_rnas_path=embedding_input_path,
            emb_output_path=curr_emb_output_path,
            model_weights_path=model_weights_path,
            model_type=model_type,
            gin_layers=gin_layers,
            graph_encoding=graph_encoding,
            structure_column_name=structure_column_name,
            structure_column_num=structure_column_num,
            header=header
        )
        emb_gen_end = time.time()

        embeddings_df = pd.read_csv(curr_emb_output_path, sep='\t')
        embeddings_df['embedding_vector'] = embeddings_df['embedding_vector'].apply(
            lambda x: np.array(list(map(float, x.split(',')))) if isinstance(x, str) else np.nan
        )
        embedding_dict = dict(zip(embeddings_df['rnacentral_id'], embeddings_df['embedding_vector']))

        if save_embeddings:
            for rid in expected_ids:
                if rid not in embedding_dict:
                    embedding_dict[rid] = np.nan

        primary_embeddings_map[pid] = embedding_dict

        if log_path:
            emb_time = emb_gen_end - emb_gen_start
            emb_log_info = {
                f"Embedding Generation for PID {pid}": f"{emb_time:.4f} seconds total, {emb_time/n_rows if n_rows>0 else 'N/A'} per row"
            }
            log_information(log_path, emb_log_info)

        if not save_embeddings:
            cleanup(directories=[emb_output_dir])

    embedding_end = time.time()

    # Benchmarking (distances and AUC)
    for bm in selected_benchmarks:
        benchmark_name = bm["name"]
        benchmark_filename = bm["filename"]
        benchmark_path = os.path.join(datasets_dir, benchmark_filename)
        benchmark_target = bm["target"]
        benchmark_version = bm["version"]
        benchmark_id = bm["id"]

        pid = bm["primary_sampled_dataset_id"]
        embedding_dict = primary_embeddings_map[pid]

        # Distance calculation timing
        dist_start = time.time()
        benchmark_df = get_distances(
            embedding_dict=embedding_dict,
            benchmark_path=benchmark_path,
            benchmark_name=benchmark_name,
            benchmark_version=benchmark_version,
            benchmarking_results_path=benchmarking_results_path,
            expected_id=benchmark_id,
            save_distances=save_distances,
            no_save=no_save
        )
        dist_end = time.time()

        n_pairs = benchmark_df.shape[0]

        # AUC calculation timing
        auc_start = time.time()
        auc_info = get_roc_auc(
            benchmark_name=benchmark_name,
            benchmark_version=benchmark_version,
            benchmark_df=benchmark_df,
            target=benchmark_target,
            benchmarking_results_path=benchmarking_results_path,
            skip_barplot=skip_barplot,
            skip_auc_curve=skip_auc_curve,
            no_save=no_save
        )
        auc_end = time.time()

        if log_path:
            dist_time = dist_end - dist_start
            auc_log_info = {
                f"Distance Calculation for {benchmark_name}": f"{dist_time:.4f} seconds total, {dist_time/n_pairs if n_pairs>0 else 'N/A'} per pair"
            }
            if auc_info is not None:
                auc_log_info[f"AUC Results for {benchmark_name}"] = f"AUC by RNA Type: {auc_info['auc_results']}, Average AUC: {auc_info['average_auc']:.4f}"

            log_information(log_path, auc_log_info)

    end_time = time.time()

    if log_path:
        # General info about datasets
        dataset_info = {
            "Primary Datasets Used": str([primary_map[pid]["filename"] for pid in needed_primary_ids]),
            "Benchmark Datasets Used": str([bm["filename"] for bm in selected_benchmarks])
        }

        total_time = end_time - start_time
        dataset_info["Total Execution Time"] = f"{total_time:.4f} seconds"
        log_information(log_path, dataset_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from RNA secondary structures using a trained model and benchmark them.")

    parser.add_argument('--model-script', dest='model_script', type=str, 
                        default="./predict_embedding.py",
                        help='Path to the model script. Default: "./predict_embedding.py".')

    parser.add_argument('--benchmark-metadata', dest='benchmark_metadata_path', type=str,
                        default='benchmark_datasets.json',
                        help="Name of the JSON file containing benchmark dataset information (in --datasets-dir). Default: 'benchmarking_datasets.json'")

    parser.add_argument('--datasets-dir', dest='datasets_dir', type=str,
                        default='./datasets',
                        help="Directory containing the benchmark metadata JSON, primary sampled datasets, and benchmark datasets. Default: './datasets'")

    parser.add_argument('--benchmark-datasets', dest='benchmark_datasets',
                        nargs='+',
                        help="Specify one or more benchmark datasets by name or name-version. If not provided, uses all latest versions.")

    parser.add_argument('--save-embeddings', dest='save_embeddings',
                        action='store_true',
                        help="If set, embeddings will be saved. Ignored if --no-save is given.")

    parser.add_argument('--emb-output-path', dest='emb_output_path', type=str,
                        default='benchmarking_results/embeddings',
                        help='Output path for embeddings if save_embeddings is set.')

    parser.add_argument('--structure-column-name', dest='structure_column_name', type=str,
                        help='Name of the column with RNA secondary structures. Default: "secondary_structure"')

    parser.add_argument('--structure-column-num', dest='structure_column_num', type=int,
                        help='Column number of the RNA secondary structures (0-indexed). If both name and num provided, name takes precedence.')

    parser.add_argument('--model-path', dest='model_path', type=str, 
                        default='saved_model/ResNet-Secondary.pth',
                        help='Path to the trained model file. Default: "saved_model/ResNet-Secondary.pth".')

    parser.add_argument('--header', type=str, default='True',
                        help='Specify whether input CSV files have a header (True/False). Default: True.')

    parser.add_argument('--skip-barplot', action='store_true', dest='skip_barplot', 
                        help='Skip generating the AUC barplot.')

    parser.add_argument('--skip-auc-curve', action='store_true', dest='skip_auc_curve', 
                        help='Skip generating the ROC curve.')

    parser.add_argument('--results-path', dest='results_path', type=str,
                        default='./benchmarking_results',
                        help='Path to save results. Time-stamp appended unless --no-save is specified. Default: "./benchmarking_results".')

    parser.add_argument('--save-distances', action='store_true', dest='save_distances',
                        help='Save the benchmark dataframes with the distance column.')

    parser.add_argument('--no-save', action='store_true', dest='no_save',
                        help='Do not save any output files.')

    parser.add_argument('--only-needed-embeddings', dest='only_needed_embeddings', action='store_true',
                        help='If set, only generate embeddings for RNAcentral IDs required by the benchmarks.')

    parser.add_argument('--no-log', dest='no_log', action='store_true',
                        help='If set, no log file will be created.')
    
    parser.add_argument('--model_type', type=str, default='siamese', required=True, choices=['siamese', 'gin_1','gin_2','gin_3', 'gin'], help="Type of model to use: 'siamese' or 'gin'.")
    
    parser.add_argument('--gin_layers', type=int, help='Number of gin layers.')

    parser.add_argument('--graph_encoding', type=str, choices=['allocator', 'forgi'], default='allocator', help='Encoding to use for the transformation to graph. Only used in case of gin modeling')

    args = parser.parse_args()

    if args.header.lower() not in ['true', 'false']:
        raise ValueError("Invalid value for --header. Use 'True' or 'False'.")
    args.header = (args.header.lower() == 'true')

    benchmark_metadata_fullpath = os.path.join(args.datasets_dir, args.benchmark_metadata_path)

    with open(benchmark_metadata_fullpath, 'r') as file:
        benchmark_metadata = json.load(file)

    if not args.benchmark_datasets:
        all_names = {}
        for entry in benchmark_metadata["benchmark_datasets"]:
            name = entry["name"]
            version = entry["version"]
            if name not in all_names:
                all_names[name] = []
            all_names[name].append(version)
        benchmark_datasets = []
        for name, versions in all_names.items():
            max_version = max(versions)
            benchmark_datasets.append(f"{name}-v{max_version}")
    else:
        benchmark_datasets = args.benchmark_datasets

    run_benchmark(
        model_script=args.model_script,
        benchmark_datasets=benchmark_datasets,
        benchmark_metadata=benchmark_metadata,
        benchmark_metadata_path=benchmark_metadata_fullpath,
        datasets_dir=args.datasets_dir,
        save_embeddings=args.save_embeddings,
        emb_output_path=args.emb_output_path,
        model_weights_path=args.model_path,
        model_type=args.model_type,
        gin_layers=args.gin_layers,
        graph_encoding=args.graph_encoding,
        structure_column_name=args.structure_column_name, 
        structure_column_num=args.structure_column_num,
        header=args.header,
        skip_barplot=args.skip_barplot,
        skip_auc_curve=args.skip_auc_curve,
        results_path=args.results_path,
        save_distances=args.save_distances,
        no_save=args.no_save,
        only_needed_embeddings=args.only_needed_embeddings,
        no_log=args.no_log
    )
