from typing import Callable
import numpy as np
from src.layers import Layer


class Network:
    """Sequential neural network composed of a list of layers."""

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Runs a forward pass through all layers."""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        loss_prime: Callable[[np.ndarray, np.ndarray], np.ndarray],
        epochs: int,
        learning_rate: float,
        verbose: bool = True,
    ) -> list[float]:
        """Trains the network using gradient descent and returns the loss history."""
        history: list[float] = []

        for epoch in range(1, epochs + 1):
            # 1. Forward pass
            output = self.predict(x_train)

            # 2. Compute loss
            error = loss_fn(y_train, output)
            history.append(error)

            # 3. Backward pass (in reverse order)
            grad = loss_prime(y_train, output)
            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)

            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {error:.6f}")

        return history