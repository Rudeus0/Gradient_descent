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

def compute_gradients(X, y_pred, y_true):
    # n: Number of samples
    n = len(y_true)
    
    # dw: Derivative of loss with respect to weight (slope direction) 
    # dw uses X because the weight's influence on the error scales with the input size
    dw = (2/n) * np.sum((y_pred - y_true) * X)
    
    # db: Derivative of loss with respect to bias (intercept direction)
    # db doesn't use X because bias shifts the prediction independently of the input
    db = (2/n) * np.sum(y_pred - y_true)
    
    return dw, db

def update_params(weight, bias, dw, db, learning_rate):
    # Adjusts parameters by moving in the opposite direction of the gradient
    # learning_rate: Controls the size of the step taken toward the minimum loss
    # (Multiplication happens before subtraction)
    weight = weight - learning_rate * dw
    bias = bias - (learning_rate * 1000) * db # bias needs larger step
    
    return weight, bias