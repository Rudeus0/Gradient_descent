from data import generate_data
from model import initialize_params, predict, compute_loss, compute_gradients, update_params


def train(X, y, learning_rate=0.0001, epochs=1000):
    # Set starting weights and bais to baseline (0.0)
    weight, bias = initialize_params
    
    #Track performance over time for visualizatiion, "error score" of each round
    loss_history = []
    
for epoch in range(epochs):
    # 1. Forward Pass: Make a prediction
    y_pred = predict(X, weight, bias)
    
    # 2. Calculate current error
    loss = compute_loss(y_pred, y_true)
    
    # 3. Backward Pass: Calculate directions for improvement
    dw, db = compute_gradients(X, y_pred, y_true)
    
    # 4. Optimizer: Update parameters to reduce future error
    weight, bias = update_params(weight, bias, dw, db, learning_rate)
    
    # 5. Logging: Save loss for performance tracking
    loss_history.append(loss)
    
        