import numpy as np
from typing import Tuple, List


class NeuralNetwork:
    """Neural network for recognizing horizontal and vertical lines in a 3x3 grid."""
    
    def __init__(self, input_size: int = 9, hidden_size: int = 12, output_size: int = 2, 
                 learning_rate: float = 0.1):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input neurons (9 for 3x3 grid)
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons (2 for binary classification)
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        # Store activations for backpropagation
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return (x > 0).astype(float)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Output predictions of shape (batch_size, output_size)
        """
        # Hidden layer with ReLU activation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer with sigmoid activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """
        Backward pass (backpropagation).
        
        Args:
            X: Input data of shape (batch_size, input_size)
            y: True labels of shape (batch_size, output_size)
            output: Network output from forward pass
        """
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = True) -> List[float]:
        """
        Train the neural network.
        
        Args:
            X: Training input data of shape (num_samples, input_size)
            y: Training labels of shape (num_samples, output_size)
            epochs: Number of training epochs
            verbose: Whether to print loss during training
            
        Returns:
            List of loss values for each epoch
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss (binary cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data of shape (num_samples, input_size)
            
        Returns:
            Predictions of shape (num_samples, output_size)
        """
        return self.forward(X)
    
    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """
        Make class predictions (argmax) on new data.
        
        Args:
            X: Input data of shape (num_samples, input_size)
            
        Returns:
            Class predictions of shape (num_samples,)
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)


def generate_training_data(num_samples: int = 100, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for horizontal and vertical line detection.

    Args:
        num_samples: Number of samples per class
        verbose: If True, print each generated sample (for debugging)

    Returns:
        Tuple of (X, y) where X is the input data and y is the labels
    """
    X_horizontal = []
    X_vertical = []

    samples_per_position = num_samples // 3
    remainder = num_samples % 3

    def _print_sample(sample: np.ndarray, label: str, index: int) -> None:
        if not verbose:
            return
        grid = sample.reshape(3, 3)
        print(f"{label} sample #{index}:")
        for row in grid:
            print(" ".join(str(int(x)) for x in row))
        print()

    # Helper to create a pure line (no noise) in a row
    def _row_line(row: int) -> np.ndarray:
        arr = np.zeros(9)
        arr[row * 3 : (row + 1) * 3] = 1
        return arr

    # Helper to create a pure line (no noise) in a column
    def _col_line(col: int) -> np.ndarray:
        arr = np.zeros(9)
        arr[[col, col + 3, col + 6]] = 1
        return arr

    # Generate horizontal lines for each row
    idx = 0
    for row in range(3):
        for _ in range(samples_per_position + (1 if row < remainder else 0)):
            sample = _row_line(row)
            X_horizontal.append(sample)
            _print_sample(sample, "Horizontal", idx)
            idx += 1

    # Generate vertical lines for each column
    idx = 0
    for col in range(3):
        for _ in range(samples_per_position + (1 if col < remainder else 0)):
            sample = _col_line(col)
            X_vertical.append(sample)
            _print_sample(sample, "Vertical", idx)
            idx += 1

    X = np.vstack([X_horizontal, X_vertical])
    y = np.vstack([np.zeros((len(X_horizontal), 1)), np.ones((len(X_vertical), 1))])

    # One-hot encoding for binary classification
    y_onehot = np.hstack([1 - y, y])

    return X, y_onehot


def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate accuracy of predictions."""
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    return np.mean(predicted_classes == true_classes)


if __name__ == "__main__":
    print("=" * 60)
    print("Neural Network for Shape Recognition")
    print("=" * 60)
    
    # Generate training and test data
    print("\nGenerating training data...")
    X_train, y_train = generate_training_data(num_samples=100, verbose=True)
    X_test, y_test = generate_training_data(num_samples=20, verbose=True)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Create and train the network
    print("\nInitializing neural network...")
    nn = NeuralNetwork(input_size=9, hidden_size=12, output_size=2, learning_rate=0.5)
    
    print("Training network...")
    losses = nn.train(X_train, y_train, epochs=1000, verbose=True)
    
    # Evaluate on training data
    train_predictions = nn.predict(X_train)
    train_accuracy = calculate_accuracy(train_predictions, y_train)
    print(f"\nTraining Accuracy: {train_accuracy:.4f} ({int(train_accuracy * 100)}%)")
    
    # Evaluate on test data
    test_predictions = nn.predict(X_test)
    test_accuracy = calculate_accuracy(test_predictions, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f} ({int(test_accuracy * 100)}%)")
    
    # Test with specific examples
    print("\n" + "=" * 60)
    print("Testing with specific shapes")
    print("=" * 60)
    
    # Horizontal line (middle row filled)
    horizontal_line = np.zeros(9)
    horizontal_line[0:2] = 1
    pred_h = nn.predict(horizontal_line.reshape(1, -1))
    print(f"\nHorizontal line test:")
    print(f"  Input: {horizontal_line.reshape(3, 3)}")
    print(f"  Prediction (Horizontal, Vertical): {pred_h[0]}")
    print(f"  Predicted class: {'Horizontal' if pred_h[0, 0] > pred_h[0, 1] else 'Vertical'}")
    
    # Vertical line (middle column filled)
    vertical_line = np.zeros(9)
    vertical_line[[4, 7]] = 1
    pred_v = nn.predict(vertical_line.reshape(1, -1))
    print(f"\nVertical line test:")
    print(f"  Input: {vertical_line.reshape(3, 3)}")
    print(f"  Prediction (Horizontal, Vertical): {pred_v[0]}")
    print(f"  Predicted class: {'Horizontal' if pred_v[0, 0] > pred_v[0, 1] else 'Vertical'}")
