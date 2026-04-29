import numpy as np

def initialize_params():
    """
    Sets initial values for the model. 
    Starting at 0.0 provides a neutral baseline before training begins.
    """
    weight = 0.0
    bias = 0.0
    return weight, bias

def predict(X, weight, bias):
    # Calculates the model's output using the linear formula y = wx + c (bias)
    # Note: Initial output will be 0.0 until weights are optimized
    return weight * X + bias


def compute_loss(y_pred, y_true):
    # Calculates the Mean Squared Error (MSE)
    # Measures the average squared distance between predictions and actual values
    return np.mean((y_pred - y_true) ** 2)