import numpy as np

def generate_data(n=100, seed=41):
    np.random.seed(seed)
    X = np.random.uniform(50, 250, n)
    noise = np.random.normal(0, 20, n)
    y = 1.5 * X + 50 + noise 
    return X, y
    
