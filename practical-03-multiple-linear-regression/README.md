# Practical 3 — Multiple Linear Regression

**Dataset:** Diabetes (442 samples, 10 features, continuous target) · **File:** [`practical_03_multiple_linear_regression.py`](practical_03_multiple_linear_regression.py)

## 1. Aim
Predict a continuous target (disease progression one year after baseline) from several features and evaluate with **MSE, RMSE and R²**.

## 2. Theory

### Model
ŷ = β₀ + β₁x₁ + β₂x₂ + … + βₚxₚ  (matrix form: ŷ = Xβ)

### Ordinary Least Squares (OLS)
Chooses β to minimize the residual sum of squares Σ(yᵢ − ŷᵢ)². Closed form: **β = (XᵀX)⁻¹ Xᵀy**. `intercept_` is β₀; `coef_` holds β₁…βₚ.

### Assumptions
Linearity · independence of errors · homoscedasticity (constant error variance) · normally-distributed residuals · no severe multicollinearity.

### Evaluation metrics
| Metric | Formula | Meaning |
|---|---|---|
| **MSE** | (1/n) Σ(y − ŷ)² | average squared error; penalizes large errors |
| **RMSE** | √MSE | same units as target |
| **R²** | 1 − SS_res / SS_tot | fraction of variance explained (1 = perfect, 0 = predicts the mean) |

## 3. Code walkthrough
| Step | Explanation |
|---|---|
| Load | `load_diabetes(return_X_y=True)` → feature matrix `X`, target `y`. (Features are already mean-centred & scaled.) |
| Split | 80 % train / 20 % test, `random_state=42`. |
| Train | `LinearRegression().fit(X_train, y_train)`. |
| Predict | `model.predict(X_test)`. |
| Metrics | `mean_squared_error`, `np.sqrt`, `r2_score`. |

### Full code
```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")
```

### Output
```text
Intercept: 151.34560453985995
Coefficients: [  37.90402135 -241.96436231  542.42875852  347.70384391 -931.48884588
  518.06227698  163.41998299  275.31790158  736.1988589    48.67065743]
MSE: 2900.194
RMSE: 53.853
R2 Score: 0.453
```

## 4. Interpretation
* **R² ≈ 0.45** → the model explains about 45 % of the variance in disease progression on unseen data — a moderate fit (the relationship is only partly linear and the data is noisy).
* **RMSE ≈ 53.9** → predictions are off by ~54 target units on average.
* Feature order is `age, sex, bmi, bp, s1 … s6`. Large positive coefficients (e.g. `bmi` ≈ 542, `s5` ≈ 736) mean strong positive influence; a large negative one (`s1` ≈ −931) means strong negative influence. Several coefficients are large and of opposite signs, a sign of **multicollinearity** among the blood-serum features.

## 5. Viva questions
* **What does R² = 0.45 mean?** 45 % of variance is explained by the model.
* **Why RMSE as well as MSE?** It is in the original units, easier to interpret.
* **Why split into train/test?** To estimate performance on unseen data.

## 6. Notes from debugging
Runs without errors; results identical to the notebook.
