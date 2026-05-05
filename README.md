# Simple Neural Network
Neural network built from scratch using NumPy to recognize simple shapes (horizontal and vertical lines) from 3x3 input images.

## Requirements

- Python 3.7+
- NumPy

Install NumPy if not already available:
```bash
pip install numpy
```

## How to Run

1. Clone the repository:
```bash
git clone <repository-url>
cd simple-neural-network
```

2. Run the neural network:
```bash
python neural_network.py
```

## What It Does

The script will:
1. Generate training data (100 samples per class) and test data (20 samples per class)
   - Training uses full 3-pixel lines (all positions in a row/column)
   - Network generalizes to recognize 2-pixel lines as well
2. Train a neural network with:
   - 9 input neurons (for 3x3 grid)
   - 12 hidden neurons (ReLU activation)
   - 2 output neurons (sigmoid activation)
3. Print training progress every 100 epochs
4. Display training and test accuracy
5. Test specific examples of horizontal and vertical lines (including 2-pixel versions)

## Example Output

```
============================================================
Neural Network for Shape Recognition
============================================================

Generating training data...
Training data shape: (200, 9)
Training labels shape: (200, 2)

Initializing neural network...
Training network...
Epoch 100/1000, Loss: 0.xxxx
...
Epoch 1000/1000, Loss: 0.xxxx

Training Accuracy: 1.0000 (100%)
Test Accuracy: 1.0000 (100%)

============================================================
Testing with specific shapes
============================================================

Horizontal line test (2-pixel):
  Input: [[1. 1. 0.]
          [0. 0. 0.]
          [0. 0. 0.]]
  Prediction (Horizontal, Vertical): [0.95 0.05]
  Predicted class: Horizontal

Vertical line test (2-pixel):
  Input: [[0. 1. 0.]
          [0. 1. 0.]
          [0. 0. 0.]]
  Prediction (Horizontal, Vertical): [0.05 0.95]
  Predicted class: Vertical

Note: The network was trained on full 3-pixel lines but generalizes well to 2-pixel lines.
```

## Implementation Details

- **Framework**: Pure NumPy (no TensorFlow/Keras)
- **Architecture**: Feedforward neural network with one hidden layer
- **Training**: Gradient descent with backpropagation
- **Data**: Binary 3x3 matrices representing horizontal (row) and vertical (column) lines
- **Generalization**: Can recognize both 3-pixel (full) and 2-pixel (partial) lines
- **Data shuffling**: Enabled to improve training convergence

## Limitations

**This network cannot recognize slanted lines.**

The neural network is specifically trained to classify only two classes:
- Horizontal lines (pixels in a row set to 1)
- Vertical lines (pixels in a column set to 1)

The network generalizes well to **2-pixel lines** even though it was trained on **3-pixel lines**.

Slanted lines (e.g., `[[1, 0, 0], [0, 1, 0], [0, 0, 1]]` or `[[0, 0, 1], [0, 1, 0], [1, 0, 0]]`) are **not recognized** because:

1. The network was never trained on slanted line examples
2. The output layer has only 2 neurons (one for horizontal, one for vertical)
3. Slanted patterns would be misclassified as either horizontal or vertical with low confidence

To recognize slanted lines, you would need to:
- Add slanted line samples to the training data
- Add a third output neuron for slanted classification
- Retrain the network with the expanded dataset
