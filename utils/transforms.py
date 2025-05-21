from torchvision import transforms

def get_base_transforms(): # Base transforms for all images in the dataset for matching the input size and type of the model
    return [
        transforms.Grayscale(num_output_channels=3), # Convert grayscale images to 3 channels
        transforms.Resize((224, 224)) # Resize images to 224x224
    ]

def get_train_transforms(): # Transformations for training images
    base = get_base_transforms()
    augmentations = [
        transforms.RandomHorizontalFlip(), # Randomly flip images horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Randomly change brightness and contrast
        transforms.RandomRotation(10) # Randomly rotate images by up to 10 degrees
    ]
    # Compose augmentation + base + tensor conversion + normalization
    return transforms.Compose(augmentations + base + [
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5]*3, std=[0.5]*3)
    ])

def get_val_test_transforms(): # Transformations for validation and test set images
    base = get_base_transforms() 
    return transforms.Compose(base + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
