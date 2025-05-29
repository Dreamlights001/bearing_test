import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import BearingDataset, SEQUENCE_LENGTH, NUM_CLASSES
from model import BearingTransformer
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'bearingset', 'train_set')
TEST_DATA_DIR = os.path.join(BASE_DIR, 'bearingset', 'test_set') # For validation during training
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'training_plots')

# Hyperparameters
INPUT_DIM = 1  # Number of features (amplitude)
SEQ_LEN = SEQUENCE_LENGTH
D_MODEL = 64
NHEAD = 4
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
NUM_CLASSES_MODEL = NUM_CLASSES

LEARNING_RATE = 0.0005 # Adjusted learning rate
BATCH_SIZE = 64 # Increased batch size
NUM_EPOCHS = 50 # Increased epochs for potentially better convergence
WEIGHT_DECAY = 1e-5 # Added weight decay for regularization

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epoch_count, save_path_base):
    epochs = range(1, epoch_count + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_filename = f"{save_path_base}_metrics_epoch_{epoch_count}.png"
    plt.savefig(plot_filename)
    print(f"Metrics plot saved to {plot_filename}")
    plt.close()

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print("Loading training data...")
    train_dataset = BearingDataset(data_dir=TRAIN_DATA_DIR)
    if len(train_dataset) == 0:
        print("Training dataset is empty. Exiting.")
        return
    train_scaler = train_dataset.get_scaler()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    print("Loading validation data...")
    # Use the scaler fitted on training data for the validation set
    val_dataset = BearingDataset(data_dir=TEST_DATA_DIR, fit_scaler_on_all_data=train_scaler)
    if len(val_dataset) == 0:
        print("Validation dataset is empty. Validation steps will be skipped.")
        val_loader = None
    else:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Model, Loss, Optimizer
    model = BearingTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES_MODEL
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # Training Loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path_base = os.path.join(MODEL_SAVE_DIR, f"bearing_transformer_model_{timestamp}")
    plot_save_path_base = os.path.join(RESULTS_DIR, f"training_plot_{timestamp}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (inputs, labels, _, _) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if (i + 1) % 20 == 0: # Print every 20 mini-batches
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        # Validation Step
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs_val, labels_val, _, _ in val_loader:
                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, labels_val)
                    val_running_loss += loss_val.item() * inputs_val.size(0)
                    _, predicted_val = torch.max(outputs_val.data, 1)
                    total_val += labels_val.size(0)
                    correct_val += (predicted_val == labels_val).sum().item()
            
            epoch_val_loss = val_running_loss / len(val_dataset)
            epoch_val_acc = correct_val / total_val
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

            scheduler.step(epoch_val_loss) # LR scheduler step

            # Save best model based on validation accuracy
            if epoch_val_acc > best_val_accuracy:
                best_val_accuracy = epoch_val_acc
                best_model_path = f"{model_save_path_base}_best_epoch_{epoch+1}_valacc_{best_val_accuracy:.4f}.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}")
        else:
            # If no validation loader, save based on training loss (less ideal)
            val_losses.append(0) # Placeholder
            val_accuracies.append(0) # Placeholder
            # Save the model at the end of each epoch if no validation
            current_model_path = f"{model_save_path_base}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), current_model_path)
            print(f"Model saved to {current_model_path} (no validation set used for best model selection)")

        # Plot metrics periodically (e.g., every 5 epochs or at the end)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
            plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epoch + 1, plot_save_path_base)

    print("Training finished.")
    final_model_path = f"{model_save_path_base}_final_epoch_{NUM_EPOCHS}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Final plot
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, NUM_EPOCHS, plot_save_path_base + "_final")

if __name__ == '__main__':
    train_model()