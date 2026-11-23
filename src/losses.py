import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Mean Squared Error: measures how far predictions are from true values """
    return float(np.mean(np.power(y_true - y_pred, 2)))

def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """ Derivative of MSE: tells the network which direction to adjust"""
    return 2 * (y_pred - y_true) / y_true.size
