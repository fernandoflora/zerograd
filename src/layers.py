from typing import Callable
import numpy as np

class Layer:
    """Base class for all neural network layers."""
    def __init__(self) -> None:
        self.input: np.ndarray | None = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, output_gradient:np.ndarray, learning_rate: float) -> np.ndarray:
        raise NotImplementedError

class Dense(Layer):
    """Fully connected layer: computes Y = X · W + b."""
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        # Initialize wights randomly (scaled to keep gradients stable)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Calculate how much to adjust weights and bias
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # Calculate the error to pass back to the previous layer
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Update parameters (turn the knobs)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient


class Activation(Layer):
    """Applies a non-linear activation function element-wise."""

    def __init__(
        self,
        activation: Callable[[np.ndarray], np.ndarray],
        activation_prime: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        return self.activation(self.input)

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return output_gradient * self.activation_prime(self.input)

