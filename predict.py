import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from data_loader import BearingDataset, SEQUENCE_LENGTH, NUM_CLASSES, LABEL_MAPPING
from model import BearingTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(BASE_DIR, 'bearingset', 'test_set')
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'bearingset', 'train_set') # Needed to get the scaler
MODEL_LOAD_PATH = os.path.join(BASE_DIR, 'models') # Directory where models are saved
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PREDICTIONS_SAVE_DIR = os.path.join(RESULTS_DIR, 'predictions_output') # Subfolder for CSV predictions
PLOTS_SAVE_DIR = os.path.join(RESULTS_DIR, 'prediction_plots') # Subfolder for plots

# Hyperparameters (should match the trained model's architecture)
INPUT_DIM = 1
SEQ_LEN = SEQUENCE_LENGTH
D_MODEL = 64 # Example, ensure this matches your saved model
NHEAD = 4    # Example
NUM_ENCODER_LAYERS = 3 # Example
DIM_FEEDFORWARD = 256 # Example
DROPOUT = 0.1 # Example
NUM_CLASSES_MODEL = NUM_CLASSES
BATCH_SIZE = 64 # Can be different from training, but often similar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories if they don't exist
os.makedirs(PREDICTIONS_SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)

# Reverse mapping for labels to names
REV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

def find_latest_model(model_dir):
    """Finds the latest .pth model file in a directory based on filename timestamp or modification time."""
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return None
    # Try to sort by timestamp in filename (e.g., _YYYYMMDD_HHMMSS_ or _best_epoch_...)
    # This is a heuristic, might need adjustment based on your exact naming convention
    try:
        # Prefer 'best' models if they exist and have a clear metric
        best_models = sorted([mf for mf in model_files if 'best' in mf and 'valacc' in mf],
                             key=lambda x: float(x.split('valacc_')[-1].split('.pth')[0]), reverse=True)
        if best_models:
            print(f"Found best model: {best_models[0]}")
            return os.path.join(model_dir, best_models[0])
        
        # Fallback to simple latest by name if no 'best' or if parsing fails
        model_files.sort(reverse=True) # Sorts alphabetically, often YYYYMMDD sorts correctly
        print(f"Found model by name sorting: {model_files[0]}")
        return os.path.join(model_dir, model_files[0])
    except Exception as e:
        print(f"Could not sort models by name/metric, falling back to modification time: {e}")
        # Fallback to modification time if filename parsing is complex or fails
        latest_model = max([os.path.join(model_dir, f) for f in model_files], key=os.path.getmtime)
        print(f"Found model by modification time: {os.path.basename(latest_model)}")
        return latest_model

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def predict_on_test_set(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Scaler from training data
    print("Loading training data to get scaler...")
    train_dataset_for_scaler = BearingDataset(data_dir=TRAIN_DATA_DIR)
    if len(train_dataset_for_scaler) == 0:
        print("Training dataset (for scaler) is empty. Cannot proceed without scaler.")
        return
    scaler = train_dataset_for_scaler.get_scaler()
    if scaler is None:
        print("Failed to get scaler from training data. Cannot proceed.")
        return
    print("Scaler obtained from training data.")

    # Load Test Data
    print("Loading test data...")
    test_dataset = BearingDataset(data_dir=TEST_DATA_DIR, fit_scaler_on_all_data=scaler)
    if len(test_dataset) == 0:
        print("Test dataset is empty. Exiting.")
        return
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test dataset loaded with {len(test_dataset)} samples.")

    # Load Model
    model = BearingTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES_MODEL
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        print("Ensure model architecture in predict.py matches the saved model.")
        return
        
    model.eval()
    print(f"Model loaded from {model_path}")

    all_predictions = []
    all_true_labels = []
    all_filenames = []
    all_sample_indices = []

    with torch.no_grad():
        for inputs, labels, filenames_batch, sample_indices_batch in test_loader:
            inputs = inputs.to(device)
            # labels are not used for prediction itself but for storing true labels
            
            outputs = model(inputs)
            _, predicted_classes = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames_batch)
            all_sample_indices.extend(sample_indices_batch.cpu().numpy())

    # Performance Evaluation
    accuracy = accuracy_score(all_true_labels, all_predictions)
    class_names = [REV_LABEL_MAPPING[i] for i in range(NUM_CLASSES_MODEL)]
    report = classification_report(all_true_labels, all_predictions, target_names=class_names, digits=4)
    
    print(f"\nTest Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # Save performance metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(PLOTS_SAVE_DIR, f"classification_report_{timestamp}.txt")
    with open(report_filename, 'w') as f:
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Test Set Performance:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Classification report saved to {report_filename}")

    # Plot and save confusion matrix
    cm_filename = os.path.join(PLOTS_SAVE_DIR, f"confusion_matrix_{timestamp}.png")
    plot_confusion_matrix(all_true_labels, all_predictions, class_names, cm_filename)

    # Save predictions for each sample
    # Group predictions by original CSV file
    results_by_file = {}
    for i in range(len(all_filenames)):
        fname = all_filenames[i]
        sample_idx = all_sample_indices[i]
        pred_label_num = all_predictions[i]
        true_label_num = all_true_labels[i]
        
        pred_label_name = REV_LABEL_MAPPING.get(pred_label_num, "Unknown")
        true_label_name = REV_LABEL_MAPPING.get(true_label_num, "Unknown") # Should always be known from filename

        if fname not in results_by_file:
            results_by_file[fname] = []
        results_by_file[fname].append({
            'original_filename': fname,
            'sample_index_in_file': sample_idx, # This is the column index
            'true_label_numerical': true_label_num,
            'true_label_name': true_label_name,
            'predicted_label_numerical': pred_label_num,
            'predicted_label_name': pred_label_name
        })

    # Save each original file's predictions to a new CSV
    for original_csv_filename, predictions_list in results_by_file.items():
        # Sort by sample_index_in_file to maintain original order if needed
        predictions_list.sort(key=lambda x: x['sample_index_in_file'])
        df_predictions = pd.DataFrame(predictions_list)
        
        # Construct new filename for the prediction results
        base_name, ext = os.path.splitext(original_csv_filename)
        output_csv_filename = os.path.join(PREDICTIONS_SAVE_DIR, f"{base_name}_predictions{ext}")
        
        df_predictions.to_csv(output_csv_filename, index=False)
        print(f"Predictions for {original_csv_filename} saved to {output_csv_filename}")

    print(f"\nAll prediction results and plots saved in {RESULTS_DIR}")

if __name__ == '__main__':
    # Find the latest model to use for prediction
    chosen_model_path = find_latest_model(MODEL_LOAD_PATH)
    
    if chosen_model_path:
        print(f"\nStarting prediction using model: {chosen_model_path}")
        predict_on_test_set(chosen_model_path)
    else:
        print(f"No model found in {MODEL_LOAD_PATH}. Please train a model first.")