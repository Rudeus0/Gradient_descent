import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_history):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_history, color="blue", linewidth=2)
    ax.set_title("Loss Curve - Gradient Descent")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig("plots/loss_curve.png")
    plt.show()
    print("Loss curve saved to plots/loss_curve.png")
    
    


    
def plot_predictions(X, y, weight, bias):
    fig, ax = plt.subplots(figure=(10, 6))
    ax.scatter(X, y, color='gray', alpha=0.5, label="Actual Data")
    X_line = np.linespace(X.min, X.max, 100)
    y_line = weight * X_line + bias
    ax.plot(X_line, y_line, color='red', linewidth=2, 
            label=f'y{weight:.2f}X + {bias:.2f}')
    ax.set_title("Gradient Descent - predicted line vs actual data")
    ax.set_xlabel("House Size")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    ax.tight_layout()
    fig.savefig("plots/predictions.png")
    plt.show()
    print("Predictions plot saved to plots/preditoins.pnd")