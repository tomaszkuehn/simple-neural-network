 I want to build neural network to recognize 2 different shapes (horizontal line, vertical line) from 3x3 input matrix. For this I need 9 input neurons and two output neurons, about 12 internal neurons. Using tensorFlow, input matrix is binary, shapes are 2-3 pixels.

Plan: Build Neural Network for Shape Recognition
Implement a feedforward neural network in TensorFlow/Keras to classify 3x3 binary matrices into two categories: horizontal lines and vertical lines. The network will have 9 input neurons, one hidden layer with 12 neurons, and 2 output neurons.

Steps

1. Set up the Python environment and dependencies: Create a virtual environment, install TensorFlow, NumPy, and Matplotlib.
2. Generate training and test datasets: Create binary 3x3 matrices representing horizontal and vertical lines. For horizontal lines, set 2-3 pixels in a single row to 1. For vertical lines, set 2-3 pixels in a single column to 1. Generate 1000 samples per class, with labels (0 for horizontal, 1 for vertical).
3. Build the neural network model: Use Keras Sequential API with Dense layers: Input (9) -> Dense(12, ReLU) -> Dense(2, Softmax).
4. Compile the model: Use Adam optimizer, categorical cross-entropy loss, accuracy metric.
5. Train the model: Fit on training data for 50 epochs, with validation split.
6. Evaluate the model: Test on unseen data, print accuracy and confusion matrix.
7. Visualize results: Plot training history and some predictions.

Relevant files

requirements.txt — List dependencies: tensorflow, numpy, matplotlib
data_generator.py — Script to generate the 3x3 matrices and labels
model.py — Define and train the neural network
evaluate.py — Test the model and visualize results

Verification

1. Run python data_generator.py to ensure data generation works and outputs correct shapes.
2. Run python model.py to train the model; check that loss decreases and accuracy increases.
3. Run python evaluate.py to test; verify test accuracy >90% and correct classifications.

Decisions

Framework: TensorFlow/Keras as specified.
Data: Binary 3x3 matrices; horizontal lines have 1s concentrated in rows (2-3 pixels per row), vertical in columns.
Model: Simple feedforward NN with one hidden layer of 12 neurons.
Scope: Includes data generation, training, evaluation; excludes advanced features like CNNs or data augmentation beyond basic generation.
Further Considerations

If the model doesn't converge well, consider adjusting the number of hidden neurons or adding more layers.