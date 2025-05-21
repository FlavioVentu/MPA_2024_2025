from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .transforms import get_train_transforms, get_val_test_transforms
from torch.utils.data import Subset
from random import shuffle
import json

def load_data(data_dir, batch_size=16): # Function to load data
    dataset = ImageFolder(root=data_dir) # Load dataset using ImageFolder and custom transforms

    dataset_size = len(dataset) # Get number of samples in the dataset
    train_size = int(0.7 * dataset_size) # 70% for training
    val_size = int(0.2 * dataset_size) # 20% for validation

    indices = list(range(len(dataset))) # Get indices of the dataset
    shuffle(indices) # Shuffle the indices
    train_indices = indices[:train_size] # Get training indices
    val_indices = indices[train_size:train_size + val_size] # Get validation indices
    test_indices = indices[train_size + val_size:] # Get test indices

    train_set = Subset(dataset, train_indices) # Create a subset for training
    val_set = Subset(dataset, val_indices) # Create a subset for validation

    with open("out/test_set.json", "w") as f: # Save test set indices to a JSON file
        json.dump(test_indices, f)

    print("Applying transforms to training and validation sets...")
    train_set.dataset.transform = get_train_transforms() # Apply training transforms
    val_set.dataset.transform = get_val_test_transforms() # Apply validation transforms

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0) # Create DataLoader for training set
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0) # Create DataLoader for validation set

    return train_loader, val_loader # Return DataLoaders for train, val, and test sets