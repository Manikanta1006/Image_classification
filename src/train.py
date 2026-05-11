import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import json
import os
import sys

# Add the 'src' directory to the Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SimpleCNN
from dataset import get_data_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train the image classifier.")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    parser.add_argument("--model-dir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def train():
    # --- 1. Configuration and Hyperparameters ---
    args = parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    # Setup device (Use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if data directory is properly set up
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"Error: Could not find training data folder at '{train_dir}'.")
        print("Please read the README.md to understand how to structure your image data.")
        return

    # --- 2. Load Data ---
    print("Loading data...")
    train_loader, val_loader, classes = get_data_loaders(data_dir, batch_size)
    num_classes = len(classes)
    print(f"Classes found ({num_classes}): {classes}")
    if num_classes < 2:
        print("Error: Add at least two class folders inside data/train before training.")
        return

    # --- 3. Initialize Model, Loss Function, and Optimizer ---
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 4. Training Loop ---
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients from the previous step
            optimizer.zero_grad()
            
            # Forward pass: compute predicted outputs
            outputs = model(images)
            
            # Calculate the loss
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradient of the loss
            loss.backward()
            
            # Update weights
            optimizer.step()

            running_loss += loss.item()

        # Print statistics for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # --- 5. Save the trained model ---
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'simple_cnn.pth')
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "image_size": 224,
        },
        save_path,
    )
    with open(os.path.join(model_dir, "classes.json"), "w", encoding="utf-8") as file:
        json.dump(classes, file, indent=2)
    print(f"\nTraining complete! Model saved to {save_path}")

if __name__ == "__main__":
    train()
