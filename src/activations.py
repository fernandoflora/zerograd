import numpy as np

def tanh(x: np.ndarray) -> np.ndarray:
    """Squashes values between -1 and 1."""
    return np.tanh(x)

def tanh_prime(x: np.ndarray) -> np.ndarray:
    """Derivate of tahn for backpropagation"""
    return 1.0 - np.power(np.tanh (x), 2)

def relu(x: np.ndarray) -> np.ndarray:
    """Replaces negative values with 0, keeps positive values as they are."""
    return np.maximum(0, x)

def relu_prime(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if x > 0, else 0."""
    return (x > 0).astype(float)

def softmax(x:np.ndarray) -> np.ndarray:
    """Converts raw scores (logits) into probabilities that sum to 1."""
    # Subtract max for nummerical stbility (prevents np.exp from overflowing)
    shifted = x - np.max(x, axis = 1, keepdims = True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis = 1, keepdims = 1)