import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Define mapping for fault types to numerical labels
LABEL_MAPPING = {
    "health": 0,
    "ball": 1,
    "comb": 2,
    "inner": 3,
    "outer": 4
}

NUM_CLASSES = len(LABEL_MAPPING)

# Expected sequence length (number of data points per sample)
SEQUENCE_LENGTH = 1024

def get_label_from_filename(filename):
    parts = filename.split('_')
    fault_type = parts[0]
    # Handle cases like 'ball20' or 'health30'
    if fault_type.startswith("ball"):
        return LABEL_MAPPING["ball"]
    if fault_type.startswith("comb"):
        return LABEL_MAPPING["comb"]
    if fault_type.startswith("health"):
        return LABEL_MAPPING["health"]
    if fault_type.startswith("inner"):
        return LABEL_MAPPING["inner"]
    if fault_type.startswith("outer"):
        return LABEL_MAPPING["outer"]
    raise ValueError(f"Unknown fault type in filename: {filename}")

class BearingDataset(Dataset):
    def __init__(self, data_dir, sequence_length=SEQUENCE_LENGTH, transform=None, fit_scaler_on_all_data=None):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []
        self.labels = []
        self.filenames = [] # To store original filenames for saving results
        self.sample_indices_in_file = [] # To store column index for saving results

        all_data_for_scaler = []

        if fit_scaler_on_all_data is None:
            # If no scaler is passed, collect all data from this dataset to fit a new one
            print(f"Collecting data from {data_dir} to fit scaler...")
            for filename in os.listdir(data_dir):
                if filename.endswith(".csv"):
                    file_path = os.path.join(data_dir, filename)
                    try:
                        df = pd.read_csv(file_path, header=None)
                        if df.shape[0] != self.sequence_length:
                            print(f"Warning: File {filename} has {df.shape[0]} rows, expected {self.sequence_length}. Skipping.")
                            continue
                        # Each column is a sample
                        for i in range(df.shape[1]):
                            sample_data = df.iloc[:, i].values.astype(np.float32)
                            all_data_for_scaler.append(sample_data.reshape(-1, 1))
                    except pd.errors.EmptyDataError:
                        print(f"Warning: File {filename} is empty. Skipping.")
                    except Exception as e:
                        print(f"Error reading {filename}: {e}. Skipping.")
            
            if not all_data_for_scaler:
                print(f"No data found in {data_dir} to fit scaler.")
                self.scaler = None
            else:
                all_data_concatenated = np.concatenate(all_data_for_scaler, axis=0)
                self.scaler = StandardScaler()
                self.scaler.fit(all_data_concatenated)
                print(f"Scaler fitted on data from {data_dir}.")
        else:
            # Use the provided scaler (e.g., fitted on training data)
            self.scaler = fit_scaler_on_all_data
            if self.scaler is None:
                 print(f"Warning: Provided scaler for {data_dir} is None. Data will not be scaled.")

        # Now, load and scale the data
        for filename in os.listdir(data_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(data_dir, filename)
                try:
                    label = get_label_from_filename(filename)
                    df = pd.read_csv(file_path, header=None)
                    
                    if df.shape[0] != self.sequence_length:
                        # Already warned during scaler fitting if it was this dataset
                        continue

                    for i in range(df.shape[1]): # Each column is a sample
                        sample_data = df.iloc[:, i].values.astype(np.float32).reshape(-1, 1)
                        
                        if self.scaler:
                            scaled_sample_data = self.scaler.transform(sample_data)
                        else:
                            scaled_sample_data = sample_data # No scaling if scaler is not available
                            
                        self.samples.append(scaled_sample_data.flatten()) # Store as 1D array
                        self.labels.append(label)
                        self.filenames.append(filename) # Store filename
                        self.sample_indices_in_file.append(i) # Store column index

                except pd.errors.EmptyDataError:
                    # Already warned
                    pass
                except ValueError as ve:
                    print(f"Skipping file {filename} due to error: {ve}")
                except Exception as e:
                    # Already warned
                    pass
        
        if not self.samples:
            print(f"Warning: No samples loaded from {data_dir}. Check data and file formats.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        filename = self.filenames[idx]
        sample_idx_in_file = self.sample_indices_in_file[idx]

        # Reshape sample to [sequence_length, 1] for Transformer (feature_dim=1)
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(1)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample_tensor = self.transform(sample_tensor)
        
        return sample_tensor, label_tensor, filename, sample_idx_in_file

    def get_scaler(self):
        return self.scaler

if __name__ == '__main__':
    # Example usage:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_dir = os.path.join(base_dir, 'bearingset', 'train_set')
    test_data_dir = os.path.join(base_dir, 'bearingset', 'test_set')

    print(f"Attempting to load training data from: {train_data_dir}")
    # Fit scaler on training data only
    train_dataset = BearingDataset(data_dir=train_data_dir)
    train_scaler = train_dataset.get_scaler()

    if train_scaler:
        print("Scaler fitted on training data.")
        # Use the scaler fitted on training data for the test set
        print(f"Attempting to load test data from: {test_data_dir}")
        test_dataset = BearingDataset(data_dir=test_data_dir, fit_scaler_on_all_data=train_scaler)
    else:
        print("Could not fit scaler on training data. Test data will not be scaled or loaded if dependent on scaler.")
        test_dataset = BearingDataset(data_dir=test_data_dir) # Will try to fit its own or run unscaled

    if len(train_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        print(f"Number of training samples: {len(train_dataset)}")
        sample_batch, label_batch, _, _ = next(iter(train_loader))
        print(f"Sample batch shape: {sample_batch.shape}") # Expected: [batch_size, sequence_length, 1]
        print(f"Label batch shape: {label_batch.shape}")
        print(f"Example label: {label_batch[0]}")
    else:
        print("Training dataset is empty.")

    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print(f"Number of test samples: {len(test_dataset)}")
        sample_batch_test, _, _, _ = next(iter(test_loader))
        print(f"Test sample batch shape: {sample_batch_test.shape}")
    else:
        print("Test dataset is empty.")

    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Label mapping: {LABEL_MAPPING}")