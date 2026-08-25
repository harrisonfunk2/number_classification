# MNIST Handwritten Digit Classification

## Goal

Build and train a convolutional neural network (CNN) to classify  
`28 × 28` grayscale images of handwritten digits from `0` to `9`.

## Dataset

The MNIST dataset contains 70,000 labeled handwritten digit images:

- 60,000 training images
- 10,000 testing images

The original test set was kept separate and used only after the final model configuration was selected.

## Model Architecture

The CNN contains:

- Three convolutional layers with ReLU activation
- Two max-pooling layers
- A fully connected classifier
- Two dropout layers with `p=0.25`
- A final output layer containing 10 classes

Dropout was added to reduce overfitting and improve generalization.

## Data Split

The original 60,000-image training set was divided into training and validation subsets using the `validation_set()` function.

- Training set: 85%
- Validation set: 15%
- Test set: Original 10,000 MNIST test images

## Baseline Model

The file `baseline_model.ipynb` contains the original baseline CNN.

The baseline model achieved approximately 99.2% validation accuracy, but its training curves showed signs of overfitting.

## Model Experiments

Several model configurations and hyperparameters were tested individually and in combination.

Experiments included:

- Dropout with different probabilities
- Batch normalization
- Different learning rates
- Dropout combined with batch normalization

The final model was selected using validation loss and validation accuracy.

## Best Model Checkpoint

During training, the model weights from the epoch with the lowest validation loss were saved and restored before final evaluation.

This prevents the final model from automatically using the weights from the last training epoch when an earlier epoch performed better.

## Results

On the untouched MNIST test set, the final model achieved:

- Accuracy: approximately **99.41%**
- Incorrect predictions: **59 out of 10,000 images**
- Macro precision, recall, and F1 score: approximately **99.4%**

The project also includes:

- Training and validation loss curves
- Training and validation accuracy curves
- A confusion matrix
- Inspection of high-confidence incorrect predictions

## Project Files

- `cnn.py` — complete training and evaluation pipeline
- `cnn_analysis.ipynb` — interactive analysis with plots and outputs
- `baseline_model.ipynb` — original baseline CNN
- `best_cnn_model.pth` — saved parameters from the best model checkpoint
