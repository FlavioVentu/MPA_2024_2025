import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from model.vehicle_model import get_model
import matplotlib.pyplot as plt
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available and set the device accordingly
model = get_model().to(device) # Load the model and move it to the appropriate device
model.load_state_dict(torch.load("out/vehicle_model.pth")) # Load the best model
print(f"Model loaded on {device}") # Print model and device information
model.eval() # Set the model to evaluation mode

with open("out/test_set.json", "r") as f: # Load test set indices from JSON file
    test_indices = json.load(f) # Read the indices from the file

dataset = ImageFolder(root="data/Vehicle") # Load the dataset using ImageFolder
test_set = Subset(dataset, test_indices) # Create a subset of the dataset for testing

from utils.transforms import get_val_test_transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
transform = get_val_test_transforms()

# Funzione per mostrare 4 immagini a schermo
def show_images(images, labels):
    _, axes = plt.subplots(1, 4, figsize=(12,3))
    for i, (img, label) in enumerate(zip(images, labels)):
        img_np = np.array(img)
        if img_np.shape[0] == 3:  # Canale primo
            img_np = np.transpose(img_np, (1,2,0))
        axes[i].imshow(img_np)
        axes[i].axis('off')
        axes[i].set_title(f"[{i+1}] Label: {label}")
    plt.show()

# Traccia accuratezza
correct = 0
total = 0

y_true = []
y_pred = []

# --- Modalità manuale o automatica ---
mode = input("Scegli modalità: [m]anuale o [a]utomatica? (m/a): ").strip().lower()
if mode == "a":
    # --- Modalità automatica: testa tutte le immagini e mostra la matrice di confusione ---
    print("Esecuzione in modalità automatica...")
    for idx in range(len(test_set)):
        img, label = test_set[idx]
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_t)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0
        y_true.append(label)
        y_pred.append(pred)
        if pred == label:
            correct += 1
        total += 1
    print(f"Accuracy: {correct}/{total} = {correct/total:.2f}")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(dataset.classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice di Confusione")
    plt.savefig("out/graphs/confusion_matrix.png")
    plt.show()
else:
    # --- Modalità manuale (come prima) ---
    while True:
        # Scegli casualmente 4 immagini dal test set
        indices = np.random.choice(len(test_set), 4, replace=False)
        images = []
        labels = []
        for idx in indices:
            img, label = test_set[idx]
            # Dataset senza trasformazioni, quindi le applichiamo ora per la rete
            img_t = transform(img)
            images.append(img)
            labels.append(dataset.classes[label])  # nome della classe
        
        # Mostra immagini a schermo
        show_images(images, labels)
        
        # Chiedi all’utente quale immagine selezionare (1-4)
        choice = input("Seleziona un'immagine da 1 a 4 (o q per uscire): ")
        if choice.lower() == 'q':
            # Calcola e mostra la matrice di confusione

            if total > 0:
                cm = confusion_matrix(y_true, y_pred, labels=list(range(len(dataset.classes))))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
                disp.plot(cmap=plt.cm.Blues)
                plt.title("Matrice di Confusione")
                plt.savefig("out/graphs/confusion_matrix.png") # Save the plot
                plt.show()
            else:
                print("Nessuna predizione effettuata, impossibile mostrare la matrice di confusione.")
            break
        try:
            choice = int(choice)
            assert 1 <= choice <= 4
        except:
            print("Input non valido, riprova.")
            continue

        # Prendi l’immagine scelta e prepara per modello
        selected_idx = indices[choice-1]
        img, label = test_set[selected_idx]
        img_t = transform(img).unsqueeze(0).to(device)  # aggiungi batch dim
        
        # Predict
        with torch.no_grad():
            output = model(img_t)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0
        
        print(f"Predizione modello: {dataset.classes[pred]}, Etichetta vera: {dataset.classes[label]}")

        # Aggiorna accuratezza
        if pred == label:
            print("Corretto!")
            correct += 1
        else:
            print("Sbagliato!")
        total += 1
        print(f"Accuracy attuale: {correct}/{total} = {correct/total:.2f}\n")

        y_true.append(label)
        y_pred.append(pred)