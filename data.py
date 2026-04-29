import numpy as np

def generate_data(n=100, seed=41):
    # n: The number of data points (samples) to generate
    # seed: Ensures reproducibility so the "random" results stay the same every run
    np.random.seed(seed)
    
    # X: Input feature; picks 100 values evenly spread between 50 and 250
    X = np.random.uniform(50, 250, n)
    
    # noise: Random "errors" from a Bell Curve (Mean=0, StdDev=20) to make data realistic
    noise = np.random.normal(0, 20, n)
    
    # y: Linear Regression formula (y = mx + c + error(noise)) where m=1.5 and c=50 c is intercept
    y = 1.5 * X + 50 + noise 
    return X, y
    
