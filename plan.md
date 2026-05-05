 I want to build neural network to recognize 2 different shapes (horizontal line, vertical line) from 3x3 input matrix. For this I need 9 input neurons and two output neurons, about 12 internal neurons. Input matrix is binary, shapes are 3 pixels (full lines across the grid).

Plan: Build Neural Network for Shape Recognition
Implement a feedforward neural network from scratch using NumPy to classify 3x3 binary matrices into two categories: horizontal lines and vertical lines. The network will have 9 input neurons, one hidden layer with 12 neurons (ReLU activation), and 2 output neurons (sigmoid activation for binary classification).

Steps

1. Set up the Python environment: Install NumPy (no TensorFlow needed - implementation is from scratch).
2. Generate training and test datasets: Create binary 3x3 matrices representing horizontal and vertical lines. For horizontal lines, set all 3 pixels in a single row to 1. For vertical lines, set all 3 pixels in a single column to 1. Generate 100 samples per class for training, 20 for testing. Data is shuffled before training.
3. Build the neural network: Implement a custom NeuralNetwork class with:
   - Weights initialized with small random values (randn * 0.01)
   - Hidden layer: Dense(9 -> 12) with ReLU activation
   - Output layer: Dense(12 -> 2) with sigmoid activation
   - Backpropagation with gradient descent
4. Train the model: Use binary cross-entropy loss, learning rate 0.5, train for 1000 epochs.
5. Evaluate the model: Test on unseen data, print accuracy for both training and test sets.
6. Test with specific examples: Verify predictions on sample horizontal and vertical lines.

Relevant files

neural_network.py — Single file containing the NeuralNetwork class, data generation, training, and evaluation code
README.md — Project description

Verification

1. Run python neural_network.py to generate data, train the model, and see results.
2. Check that loss decreases during training (printed every 100 epochs).
3. Verify test accuracy >90% and correct classifications on test examples.

Decisions

Framework: Pure NumPy implementation (no TensorFlow/Keras) - built from scratch.
Data: Binary 3x3 matrices; horizontal lines have all 3 pixels in a row set to 1, vertical lines have all 3 pixels in a column set to 1. Data is shuffled before training.
Model: Simple feedforward NN with one hidden layer of 12 neurons (ReLU), output layer with 2 neurons (sigmoid).
Loss: Binary cross-entropy.
Optimizer: Gradient descent (no Adam - simple implementation).
Scope: Single file implementation including data generation, training, evaluation, and testing.
Further Considerations

- The implementation works well for this simple task but may need adjustments for more complex patterns.
- Consider lowering learning rate if training becomes unstable.
- Data is perfectly separable (no noise), so high accuracy is expected.