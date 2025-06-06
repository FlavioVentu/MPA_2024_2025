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
unfreeze_done = False # Flag to indicate if the model is unfrozen
reduce_done = False # Flag to indicate if the learning rate has been reduced

best_val_loss = float('inf') # Initialize best validation loss to infinity
patience = 4 # Set patience for early stopping
unfreeze_patience = 3 # Set patience for unfreezing
red_patience = 2 # Set patience for learning rate reduction
ofit_patience = 2 # Set patience for overfitting
counter = 0 # Initialize counter for early stopping
min_delta = 1e-4 # Minimum change to qualify as an improvement
ofit_counter = 0 # Initialize counter for overfitting

train_losses = [] # List to store training losses
val_losses = [] # List to store validation losses

train_acc = [] # List to store training accuracies
val_acc = [] # List to store validation accuracies

epochs = 100 # Number of epochs to train the model

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
    train_acc.append(train_accuracy) # Append training accuracy to list
    val_acc.append(val_accuracy) # Append validation accuracy to list
    end = time.time() # End time for epoch
    print(f"- Epoch {epoch+1}\n\tTime elapsed: {end - start:.4f} seconds,\n\tTrain Loss: {train_loss:.4f},\n\tTrain Accuracy: {train_accuracy:.2f}%,\n\tVal Loss: {val_loss:.4f},\n\tVal Accuracy: {val_accuracy:.2f}%") # Print training and validation loss and accuracy


    # Early stopping
    # Check if validation loss improved

    if val_loss > train_loss: # Check if validation loss is greater than training loss
        ofit_counter += 1 # Increment overfitting counter
        if ofit_counter >= ofit_patience:
            print("Overfitting detected, stopping training!")
            break
        print(f"Possible overfitting detected, incrementing overfitting counter: {ofit_counter}/{ofit_patience}")


    elif best_val_loss - val_loss > min_delta: # Check if validation loss improved
        best_val_loss = val_loss # Update best validation loss
        torch.save(model.state_dict(), "out/vehicle_model.pth") # Save the model
        print("Validation loss improved, saving model...")
        counter = 0 # Reset counter
    else:
        counter += 1 # Increment counter if validation loss did not improve
        print(f"No improvement in validation loss, early stopping counter: {counter}/{patience}")

        if counter >= red_patience and not reduce_done:
            print("Reducing learning rate...")
            for param_group in optimizer.param_groups: # Loop over optimizer parameter groups
                param_group['lr'] *= 0.5 # Reduce learning rate by half (5-e5)
            reduce_done = True # Set reduce_done to True if unfreeze_done is True
            counter = 0 # Reset counter

        if counter >= unfreeze_patience and not unfreeze_done: # Check if patience for unfreezing exceeded
            print("Unfreezing last layer of the model...")
            for name, param in model.named_parameters():
                if "layer4" in name:
                    param.requires_grad = True # Unfreeze the last layer
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-5) # Reinitialize optimizer with a lower learning rate
            unfreeze_done = True
            counter = 0  # Reset counter

        # Early stopping
        if counter >= patience: # Check if patience exceeded
            print("Early stopping!") # Print early stopping message
            break

os.makedirs("out/graphs", exist_ok=True) # Create output directory for graphs if it doesn't exist

# Plot training and validation loss
plt.plot(train_losses, label='Train Loss') # Plot training loss
plt.plot(val_losses, label='Validation Loss') # Plot validation loss
plt.title('Training and Validation Loss') # Set title
plt.xlabel('Epochs') # Set x-axis label
plt.ylabel('Loss') # Set y-axis label
plt.legend() # Show legend
plt.grid(True) # Show grid
plt.savefig("out/graphs/loss_plot.png") # Save the plot
plt.clf() # Clear the plot for the next graph

# Plot training and validation accuracy
plt.plot(train_acc, label='Train Accuracy') # Plot training accuracy
plt.plot(val_acc, label='Validation Accuracy') # Plot validation accuracy
plt.title('Training and Validation Accuracy') # Set title
plt.xlabel('Epochs') # Set x-axis label
plt.ylabel('Accuracy (%)') # Set y-axis label
plt.legend() # Show legend
plt.grid(True) # Show grid
plt.ylim(60, 100) # Set y-axis limits
plt.yticks(range(60, 101, 2)) # Set y-axis ticks
plt.savefig("out/graphs/accuracy_plot.png") # Save the plot

print("Training complete!") # Print training complete message
