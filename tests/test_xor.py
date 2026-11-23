import numpy as np
from src.activations import tanh, tanh_prime
from src.layers import Activation, Dense
from src.losses import mse, mse_prime
from src.network import Network


def test_xor() -> None:
    # Fix the random seed so the test is reproducible every time
    np.random.seed(42)

    # XOR dataset: 4 samples, 2 inputs -> 1 output
    x_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y_train = np.array([
        [0],
        [1],
        [1],
        [0],
    ])

    # Build a 2-layer network: 2 inputs -> 3 hidden neurons -> 1 output
    net = Network([
        Dense(2, 3),
        Activation(tanh, tanh_prime),
        Dense(3, 1),
        Activation(tanh, tanh_prime),
    ])

    history = net.fit(
        x_train,
        y_train,
        loss_fn=mse,
        loss_prime=mse_prime,
        epochs=1000,
        learning_rate=0.1,
        verbose=True,
    )

    predictions = net.predict(x_train)
    rounded = np.round(predictions)

    print("\nPredictions:")
    for x, pred, expected in zip(x_train, predictions, y_train):
        print(f"Input: {x} | Predicted: {pred[0]:.4f} | Expected: {expected[0]}")

    # Assertions: the test fails if the network didn't learn XOR properly
    assert history[-1] < 0.01, f"Final loss too high: {history[-1]}"
    assert np.array_equal(rounded, y_train), f"Wrong predictions:\n{rounded}"
    print("\nXOR test passed!")


if __name__ == "__main__":
    test_xor()