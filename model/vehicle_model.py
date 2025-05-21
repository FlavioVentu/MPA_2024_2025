from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

def get_model():

    model = resnet18(weights=ResNet18_Weights.DEFAULT) # Load the pre-trained ResNet-18 model and its weights

    for param in model.parameters(): # Freeze all layers
        param.requires_grad = False

    # Replace the last fully connected layer with a new one for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
