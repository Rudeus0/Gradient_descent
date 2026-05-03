# Gradient Descent from Scratch

Linear regression implemented manually using only NumPy — no scikit-learn, no TensorFlow, no shortcuts. Built to understand what every ML library does internally.

---

## What This Project Proves

Most people use `sklearn.LinearRegression()` without knowing what happens inside. This project implements every step from scratch:

- Forward pass — make a prediction
- Loss calculation — measure how wrong you are
- Backward pass — calculate which direction to improve
- Parameter update — take a step toward the minimum

**If you can explain gradient descent, you can explain every ML algorithm.**

---

## The Problem

Given house sizes (50–250 sq ft), predict house prices. The hidden relationship in the data is:

```
price = 1.5 × size + 50 + noise
```

Gradient descent must find `1.5` and `50` on its own — without being told.

---

## Results

```
Final Weight: 1.4978  (target: 1.5)
Final Bias:   51.21   (target: 50)
Loss:         326     (converged from 77,333)
```

---

## Project Structure

```
gradient_descent/
├── data.py          # generates synthetic house price data
├── model.py         # gradient descent — predict, loss, gradients, update
├── visualize.py     # loss curve + predictions plot
├── main.py          # training loop + entry point
├── plots/
│   ├── loss_curve.png
│   └── predictions.png
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# Clone
git clone https://github.com/Rudeus0/gradient-descent-from-scratch.git
cd gradient-descent-from-scratch

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

---

## Results — Charts

### Loss Curve (Log Scale)
![Loss Curve](plots/loss_curve.png)

The loss drops from 77,333 to ~326 in the first 100 epochs, then smoothly converges. Log scale shows the full descent — linear scale makes everything after epoch 100 look flat because the initial drop is so large.

### Predicted Line vs Actual Data
![Predictions](plots/predictions.png)

Red line = model's learned equation: `y = 1.50x + 51.21`
Gray dots = actual noisy data points

The line passes through the center of the scatter — the model found the hidden pattern.

---

## The Math

### Linear Regression
```
prediction = weight × X + bias
```

### Mean Squared Error (Loss)
```
loss = mean((prediction - actual)²)
```

### Gradients (Derivatives of loss)
```
dw = (2/n) × Σ (prediction - actual) × X
db = (2/n) × Σ (prediction - actual)
```

`dw` uses X because the weight's influence on error scales with input size.
`db` does not use X because bias shifts the prediction independently of input.

### Parameter Update
```
weight = weight - learning_rate × dw
bias   = bias   - learning_rate × db
```

---

## Questions I Worked Through While Building This

**Q: Why is the first prediction zero?**

Starting with weight=0 and bias=0 creates a blank slate.
0 × X + 0 = 0. The model knows nothing at epoch 0 — that's intentional.
Every parameter starts neutral and gradient descent moves them toward truth.

**Q: Why is y lowercase but X uppercase?**

Convention from linear algebra. X is a matrix (multiple features possible), y is a vector (single output). Every ML library uses this same convention.

**Q: What does compute_loss actually measure?**

The gap between y_pred and y_true is the error.
MSE squares that gap to penalize large mistakes more than small ones.
A prediction off by 10 contributes 100 to the loss — not 10.
Squaring also removes negative signs so errors don't cancel each other out.

**Q: Why does dw multiply by X but db does not?**

Weight controls how much X influences the prediction.
A large X means the weight has a bigger impact on the error — so the gradient scales with X.
Bias shifts every prediction by the same amount regardless of X.
Its gradient is just the average error — no X involved.

**Q: Why weight - learning_rate * dw and not (weight - learning_rate) * dw?**
Operator precedence — multiplication happens before subtraction.
learning_rate * dw = the step size
weight - step = new position
The goal is to move weight in the opposite direction of the gradient (downhill).


**Q: Why does bias not converge to 50 even after 5000 epochs?**

The bias gradient `db` is much smaller than `dw` because it doesn't scale with X values. The optimizer treats both equally by default. Fixed by using a separate learning rate multiplier for bias (`learning_rate × 1000`). This is exactly the problem that motivated the Adam optimizer — it automatically adjusts learning rates per parameter.

**Q: How do you choose the learning rate?**

Start with 0.001 or 0.0001 and read the loss curve:
- Loss explodes to infinity → learning rate too high, lower it
- Loss barely moves → learning rate too low, raise it
- Loss drops smoothly then flattens → correct range
This project went through 0.0001 (exploded) → 0.00001 (too slow) → 0.000003 (correct).

**Q: Why does bias need a 1000x multiplier on the learning rate?**
Weight gradient dw scales with X values (50-250 range).
Bias gradient db does not scale with X — it's just the mean error.
Result: dw is ~1000x larger than db so weight converges fast and bias crawls.
Fix: apply a 1000x multiplier to db's learning rate step.
This is exactly the problem Adam optimizer solves automatically.



**Q: Why is the final weight 1.4978 and not exactly 1.5?**

The data has noise: `y = 1.5 * X + 50 + random_noise`
The model finds the best fit line through noisy points — not the perfect line.
1.4978 IS the mathematically correct answer for this dataset.
In real ML you never know the true weights. The model found what it could.

**Q: What is converging?**  

Getting closer to the target and slowing down as it approaches.
Like a ball rolling into a bowl — fast at the edges, slow near the bottom.
When loss stops decreasing significantly the model has converged.
Your loss went from 77,333 → 326 and flattened. That is convergence.

**Q: What is bias in the model?**

In `y = weight × X + bias`, bias is the base prediction before seeing any input.
It shifts the entire line up or down.
Without bias the line always passes through zero — useless for real data.
House price at size 0 = 51.21 (the bias). Every prediction starts from there.

**Q: How do you know if the model found good values?**

Three ways:
1. Compare to true values (only possible with synthetic data like this)
2. Loss is low and converged (stopped decreasing)
3. Predicted line passes through the center of the scatter plot visually
All three confirmed here: weight ≈ 1.5, bias ≈ 50, loss converged, line fits data.


**Q: Why does exploding gradient happen?**

Learning rate too high → gradients too large → weights overshoot the minimum → loss goes to infinity. Fixed by lowering the learning rate. Normalizing input X also helps by keeping gradients small.

---

## Debugging Log — What Went Wrong

| Problem | Cause | Fix |
|---------|-------|-----|
| Loss exploded to infinity | Learning rate 0.0001 too high for X range 50-250 | Reduced to 0.000003 |
| Bias stuck at 0.01 after 5000 epochs | Bias gradient too small relative to weight gradient | Applied 1000x multiplier to bias learning rate |
| Weight converged but bias didn't | Different gradient scales for weight vs bias | Separate learning rate per parameter |

---

## Key Learnings

- Gradient descent is just: measure error → calculate direction → take a step → repeat
- Learning rate is the most important hyperparameter — too high = explode, too low = never converge
- Weight and bias have different gradient scales — real optimizers (Adam, RMSProp) handle this automatically
- Noisy data means the model will never find the exact true values — and that's correct behaviour
- Log scale on loss curves reveals the full descent that linear scale hides

---

## Tech Stack

- Python 3.12
- NumPy — all math operations
- Matplotlib — visualizations
- No ML libraries used
