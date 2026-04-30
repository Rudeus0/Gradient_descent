from data import generate_data
from model import initialize_params, predict, compute_loss, compute_gradients, update_params


def train(X, y, learning_rate=0.0001, epochs=1000):
    # Set starting weights and bais to baseline (0.0)
    weight, bias = initialize_params()
    
    #Track performance over time for visualizatiion, "error score" of each round
    loss_history = []
    
    for epoch in range(epochs):
        # 1. Forward Pass: Make a prediction
        y_pred = predict(X, weight, bias)
        
        # 2. Calculate current error
        loss = compute_loss(y_pred, y)
        
        # 3. Backward Pass: Calculate directions for improvement
        dw, db = compute_gradients(X, y_pred, y)
        
        # 4. Optimizer: Update parameters to reduce future error
        weight, bias = update_params(weight, bias, dw, db, learning_rate)
        
        # 5. Logging: Save loss for performance tracking
        loss_history.append(loss)
        
        # Logging: Print progress every 100 epochs to monitor convergence
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.2f} | Weight: {weight:.4f} | Bias: {bias:.4f}")

    return weight, bias, loss_history



if __name__ == "__main__":
    X, y = generate_data()
    weight, bias, loss_history = train(X, y)
    print(f"\n final Weight: {weight: .4f} (should be ~ 1.5)")
    print(f"Final bias: {bias:.4f} (should be ~50)")