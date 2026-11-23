import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Mean Squared Error: measures how far predictions are from true values """
    return float(np.mean(np.power(y_true - y_pred, 2)))

def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """ Derivative of MSE: tells the network which direction to adjust"""
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cross-Entropy loss for multi-class classification (expects probabilities)"""
    # Clip predictions to avoid log(0 -> -inf
    clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(np.sum(y_true * np.log(clipped), axis = 1)))

def cross_entropy_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Combined derivative of softmax + cross-entropy."""
    return (y_pred - y_true) / y_true.shape[0]