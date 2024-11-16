import logging
import os
import re
import sys

def set_logger(log_filename_base, save_dir):
    # Set up logging
    log_filename = f"{log_filename_base}.log"
    log_filepath = os.path.join(save_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
    )

    return log_filepath

def parse_experiment_path(file_path):
    # Identify the target pattern "experiments/yyyy-mm-dd"
    match = re.search(r"experiments/\d{4}-\d{2}-\d{2}", file_path)
    
    if not match:
        raise ValueError("The specified path does not contain 'experiments/yyyy-mm-dd' format.")
    
    # Get the folder structure after the "experiments/yyyy-mm-dd" segment
    sub_path = file_path[match.end() + 1:]  # Skip "experiments/yyyy-mm-dd/" part
    folder_names = os.path.splitext(sub_path)[0].split(os.sep)  # Remove extension and split by folder
    
    # Join folder names with underscores
    return "_".join(folder_names)

def search_log_files(
    log_dir: str, keywords: list[str], logic: str = "and"
) -> list[str]:
    if logic not in ["or", "and"]:
        raise ValueError("The logic parameter must be 'or' or 'and'.")

    res_files = sorted(os.listdir(log_dir))

    if logic == "and":
        res_files_filtered = [
            f for f in res_files if all(keyword in f for keyword in keywords)
        ]
    elif logic == "or":
        res_files_filtered = [
            f for f in res_files if any(keyword in f for keyword in keywords)
        ]

    return res_files_filtered


