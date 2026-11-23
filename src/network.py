from typing import Callable
import numpy as np
from src.layers import Dense, Layer


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
        batch_size: int | None = None,
        verbose: bool = True,
    ) -> list[float]:
        """Trains the network using (mini-batch) gradient descent."""
        history: list[float] = []
        n_samples = x_train.shape[0]
        effective_batch = n_samples if batch_size is None else batch_size

        for epoch in range(1, epochs + 1):
            # Shuffle dataset at the start of each epoch
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n_samples, effective_batch):
                x_batch = x_shuffled[start : start + effective_batch]
                y_batch = y_shuffled[start : start + effective_batch]

                # 1. Forward pass
                output = self.predict(x_batch)

                # 2. Compute loss
                epoch_loss += loss_fn(y_batch, output)
                num_batches += 1

                # 3. Backward pass
                grad = loss_prime(y_batch, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            avg_loss = epoch_loss / num_batches
            history.append(avg_loss)

            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.6f}")

        return history

    def save(self, filepath: str) -> None:
        """Saves weights and biases of all Dense layers to a .npz file."""
        params: dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                params[f"w_{i}"] = layer.weights
                params[f"b_{i}"] = layer.bias
        np.savez(filepath, **params)

    def load(self, filepath: str) -> None:
        """Loads weights and biases from a .npz file into the network."""
        data = np.load(filepath)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                layer.weights = data[f"w_{i}"]
                layer.bias = data[f"b_{i}"]