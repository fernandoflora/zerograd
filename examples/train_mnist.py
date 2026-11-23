import os
import time
import urllib.request
import numpy as np

from src.activations import relu, relu_prime
from src.layers import Activation, Dense, Softmax
from src.losses import cross_entropy, cross_entropy_prime
from src.network import Network

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_PATH = "data/mnist.npz"
MODEL_PATH = "models/mnist_model.npz"


def download_mnist() -> None:
    """Downloads MNIST dataset as a NumPy .npz file if not already present."""
    if not os.path.exists(DATA_PATH):
        print(f"Downloading MNIST dataset to {DATA_PATH}...")
        urllib.request.urlretrieve(MNIST_URL, DATA_PATH)
        print("Download complete.")


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Converts integer labels (e.g. 3) into one-hot vectors ([0,0,0,1,0,...])."""
    encoded = np.zeros((labels.shape[0], num_classes))
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def load_data():
    download_mnist()
    with np.load(DATA_PATH) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    # 1. Flatten 28x28 images into 784-element vectors and normalize to [0, 1]
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    # 2. One-hot encode labels for Cross-Entropy
    y_train_oh = one_hot_encode(y_train)

    return x_train, y_train_oh, x_test, y_test


def train() -> None:
    np.random.seed(42)
    x_train, y_train, x_test, y_test = load_data()

    # Architecture: 784 (pixels) -> 128 -> 64 -> 10 (digits 0-9)
    net = Network(
        [
            Dense(784, 128),
            Activation(relu, relu_prime),
            Dense(128, 64),
            Activation(relu, relu_prime),
            Dense(64, 10),
            Softmax(),
        ]
    )

    print("Training on 60,000 MNIST images...")
    start_time = time.time()

    net.fit(
        x_train,
        y_train,
        loss_fn=cross_entropy,
        loss_prime=cross_entropy_prime,
        epochs=15,
        learning_rate=0.1,
        batch_size=64,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.2f}s")

    # Evaluate on 10,000 unseen test images
    predictions = net.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == y_test) * 100.0
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save trained weights
    net.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()