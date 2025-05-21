import torch
import torch.nn as nn
import torch.optim as optim
from model.vehicle_model import get_model
from utils.prepare_data import load_data
import matplotlib.pyplot as plt
import time
import sys
import os

if os.getenv("GITHUB_ACTIONS") == "true": # Check if running in GitHub Actions and suppress output
    sys.stdout = open(os.devnull, 'w')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available and set the device accordingly else use CPU

model = get_model().to(device) # Load the model and move it to the appropriate device
print(f"Model loaded on {device}") # Print model and device information
train_loader, val_loader = load_data("data/Vehicle") # Load the data using the custom dataset loader

criterion = nn.BCEWithLogitsLoss() # Define the loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) # Define the optimizer

best_val_loss = float('inf') # Initialize best validation loss to infinity
patience = 3 # Set patience for early stopping
counter = 0 # Initialize counter for early stopping
min_delta = 1e-4 # Minimum change to qualify as an improvement

train_losses = [] # List to store training losses
val_losses = [] # List to store validation losses

train_accuracy = [] # List to store training accuracies
val_accuracy = [] # List to store validation accuracies

epochs = 50 # Number of epochs to train the model

print("Starting training...") # Print training start message
for epoch in range(epochs): # Loop over epochs
    start = time.time() # Start time for epoch
    model.train() # Set the model to training mode
    train_loss, correct, total = 0, 0, 0 # Initialize training loss and accuracy variables
    for images, labels in train_loader: # Loop over training data
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1) # Move data to device and reshape labels (batch_size, 1)
        optimizer.zero_grad() # Zero the gradients
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        train_loss += loss.item() # Accumulate training loss
        preds = (torch.sigmoid(outputs) > 0.5).float() # Apply sigmoid and threshold to get predictions
        correct += (preds == labels).sum().item() # Count correct predictions
        total += labels.size(0) # Count total samples

    train_accuracy = 100*(correct / total) # Compute training accuracy
    train_loss /= len(train_loader) # Average training loss

    print(f"Starting validation for epoch {epoch+1}...") # Print validation start message
    model.eval() # Set the model to evaluation mode for validation
    val_loss, correct, total = 0, 0, 0 # Initialize validation loss and accuracy variables
    with torch.no_grad(): # Disable gradient calculation for validation
        for images, labels in val_loader: # Loop over validation data
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1) # Move data to device and reshape labels
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            val_loss += loss.item() # Accumulate validation loss
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item() # Count correct predictions
            total += labels.size(0) # Count total samples

    val_accuracy = 100 * (correct / total) # Compute validation accuracy
    val_loss /= len(val_loader) # Average validation loss

    train_losses.append(train_loss) # Append training loss to list
    val_losses.append(val_loss) # Append validation loss to list
    train_accuracy.append(train_accuracy) # Append training accuracy to list
    val_accuracy.append(val_accuracy) # Append validation accuracy to list
    end = time.time() # End time for epoch
    print(f"- Epoch {epoch+1}\n\tTime elapsed: {end - start:.4f} seconds,\n\tTrain Loss: {train_loss:.4f},\n\tTrain Accuracy: {train_accuracy:.2f}%,\n\tVal Loss: {val_loss:.4f},\n\tVal Accuracy: {val_accuracy:.2f}%") # Print training and validation loss and accuracy


    # Early stopping
    # Check if validation loss improved

    if best_val_loss - val_loss > min_delta: # Check if validation loss improved
        best_val_loss = val_loss # Update best validation loss
        torch.save(model.state_dict(), "out/vehicle_model.pth") # Save the model
        print("Validation loss improved, saving model...")
        counter = 0 # Reset counter
    else:
        counter += 1 # Increment counter if validation loss did not improve
        print(f"No improvement in validation loss, early stopping counter: {counter}/{patience}")
        if counter >= patience: # Check if patience exceeded
            print("Early stopping!") # Print early stopping message
            break


# Plot training and validation loss
plt.plot(train_losses, label='Train Loss') # Plot training loss
plt.plot(val_losses, label='Validation Loss') # Plot validation loss
plt.title('Training and Validation Loss') # Set title
plt.xlabel('Epochs') # Set x-axis label
plt.ylabel('Loss') # Set y-axis label
plt.legend() # Show legend
plt.grid(True) # Show grid
plt.savefig("out/graphs/loss_plot.png") # Save the plot
plt.show() # Show the plot

# Plot training and validation accuracy
plt.plot(train_accuracy, label='Train Accuracy') # Plot training accuracy
plt.plot(val_accuracy, label='Validation Accuracy') # Plot validation accuracy
plt.title('Training and Validation Accuracy') # Set title
plt.xlabel('Epochs') # Set x-axis label
plt.ylabel('Accuracy (%)') # Set y-axis label
plt.legend() # Show legend
plt.grid(True) # Show grid
plt.savefig("out/graphs/accuracy_plot.png") # Save the plot
plt.show() # Show the plot

print("Training complete!") # Print training complete message
