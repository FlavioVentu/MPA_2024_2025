# Vehicle Classifier (Car / Not Car)

This project trains a binary image classifier to detect if a vehicle is centered in grayscale images using a ResNet18 CNN and transfer learning.

## Structure

- `train.py`: Training loop with early stopping and TensorBoard.
- `test.py`: Evaluate model on test set.
- `models/`: Contains ResNet model definition.
- `utils/`: Dataloader, transforms, and TensorBoard logger.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train.py
python test.py
tensorboard --logdir=runs
```
