import torch
import numpy as np
import random
import os

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # if you are using multi-GPU.
        # Potentially make operations deterministic, but can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def create_results_subfolder(base_results_dir, subfolder_name):
    """Creates a subfolder within the base results directory."""
    subfolder_path = os.path.join(base_results_dir, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    return subfolder_path

if __name__ == '__main__':
    set_seed(123)
    print("Seed set to 123.")
    
    # Example of creating a subfolder
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # results_folder = os.path.join(base_dir, 'results')
    # prediction_csv_folder = create_results_subfolder(results_folder, 'prediction_csvs_test')
    # print(f"Created folder: {prediction_csv_folder}")