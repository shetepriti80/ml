# Practical 5 — Regularized Regression: Ridge and Lasso

**Dataset:** Diabetes (10 features) · **File:** [`practical_05_ridge_lasso.py`](practical_05_ridge_lasso.py)

## 1. Aim
Compare Linear Regression, Ridge (L2) and Lasso (L1); study how regularization shrinks coefficients and reduces overfitting.

## 2. Theory

### The problem
Plain OLS can overfit when features are many or correlated: coefficients become large and unstable. Regularization adds a **penalty on coefficient size** to the loss.

### Ridge (L2)
Loss = Σ(y − ŷ)² + **α Σ βⱼ²**
* Shrinks all coefficients toward 0 but **rarely makes them exactly 0**.
* Handles multicollinearity well.

### Lasso (L1)
Loss = Σ(y − ŷ)² + **α Σ |βⱼ|**
* Can drive some coefficients **exactly to 0** → built-in **feature selection**.

### Role of α (alpha)
α = 0 → ordinary least squares · small α → little shrinkage · large α → strong shrinkage (risk of underfitting). Tune α with cross-validation (`RidgeCV`, `LassoCV`).

| | Ridge | Lasso |
|---|---|---|
| Penalty | L2 (β²) | L1 (|β|) |
| Zero coefficients? | No | Yes |
| Best when | many correlated features | few important features |

**Scaling is essential** — the penalty treats all coefficients equally, so features must be on the same scale (hence `StandardScaler`).

## 3. Code walkthrough
| Step | Explanation |
|---|---|
| Split | 80/20, `random_state=42`. |
| Models dict | Each model is a `make_pipeline(StandardScaler(), estimator)` so scaling is fitted only on training data. Ridge `alpha=1.0`, Lasso `alpha=0.1, max_iter=10000`. |
| Loop | Fit → predict → store MSE and R². |
| Coefficients | `models[name].named_steps[list(models[name].named_steps)[-1]]` retrieves the last step (the estimator) of each pipeline, then prints `coef_` rounded to 3 decimals (on the standardized scale). |

### Full code
```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000))
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append([name, mean_squared_error(y_test, pred),
                 r2_score(y_test, pred)])

print(pd.DataFrame(rows, columns=["Model", "MSE", "R2"]))

# Compare coefficients on the standardized feature scale
for name in ["Linear Regression", "Ridge", "Lasso"]:
    estimator = models[name].named_steps[list(models[name].named_steps)[-1]]
    print(name, np.round(estimator.coef_, 3) if hasattr(estimator, "coef_") else "")
```

### Output
```text
               Model          MSE        R2
0  Linear Regression  2900.193628  0.452603
1              Ridge  2892.014566  0.454147
2              Lasso  2884.624289  0.455541
Linear Regression [  1.754 -11.512  25.607  16.829 -44.449  24.641   7.677  13.139  35.161
   2.351]
Ridge [  1.807 -11.448  25.733  16.734 -34.672  17.053   3.37   11.764  31.378
   2.458]
Lasso [  1.73  -11.316  25.825  16.644 -29.358  13.276   0.548  10.236  29.633
   2.393]
```

## 4. Interpretation
* All three models perform almost the same (R² ≈ 0.45); Lasso is marginally best (MSE 2884.6).
* **Shrinkage:** the largest coefficient (`s1`) goes −44.4 (OLS) → −34.7 (Ridge) → −29.4 (Lasso); `s3` goes 7.68 → 3.37 → 0.55.
* With Lasso α = 0.1 no coefficient becomes exactly 0, but `s3` is nearly eliminated; a larger α would zero out more features.
* Slight regularization improves generalization without hurting accuracy.

## 5. Viva questions
* **Ridge vs Lasso?** L2 shrinks; L1 shrinks *and* selects features.
* **Why scale before regularizing?** Penalty depends on coefficient size, which depends on feature scale.
* **What happens if α is too large?** Underfitting (high bias).

## 6. Notes from debugging
Runs without errors; output identical to the notebook. The `hasattr(estimator, "coef_")` check is redundant (all three have `coef_`) but harmless.
