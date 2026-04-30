from data import generate_data
from model import initialize_params, predict, compute_loss, compute_gradients, update_params


def train(X, y, learning_rate=0.0001, epochs=1000):
    # Set starting weights and bais to baseline (0.0)
    weight, bais = initialize_params
    
    #Track performance over time for visualizatiion, "error score" of each round
    loss_history = []
    
