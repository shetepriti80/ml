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