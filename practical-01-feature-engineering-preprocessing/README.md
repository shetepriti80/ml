# Practical 1 — Feature Engineering and Data Preprocessing

**Dataset:** UCI Wine (via `sklearn.datasets.load_wine`) · **File:** [`practical_01_feature_engineering.py`](practical_01_feature_engineering.py)

## 1. Aim
Take a dataset with numerical **and** categorical columns, handle missing values, encode categories, scale numbers, and produce a clean, model-ready matrix.

## 2. Theory

### Why preprocess?
Real data is rarely model-ready. Most ML algorithms need numeric, complete, comparably-scaled inputs. Poor preprocessing = poor models ("garbage in, garbage out").

### a) Missing-value imputation
| Strategy | Use for | Notes |
|---|---|---|
| **Mean** | numeric, symmetric | sensitive to outliers |
| **Median** | numeric, skewed / outliers | robust — used here |
| **Most frequent (mode)** | categorical | used here for `class_name` |
| Constant / KNN / iterative | special cases | more advanced |

### b) Encoding categorical variables
* **One-Hot Encoding** – one binary column per category; no false ordering (used here). Best for *nominal* data.
* **Ordinal Encoding** – integers 0,1,2…; use only when order is meaningful (Low < Medium < High).
* `handle_unknown="ignore"` – an unseen category at prediction time becomes all-zeros instead of raising an error.

### c) Feature scaling
* **StandardScaler (z-score):** `z = (x − μ) / σ` → mean 0, std 1 (used here).
* **MinMaxScaler:** `x' = (x − min)/(max − min)` → range [0, 1].
* Needed for distance/gradient-based models (KNN, SVM, PCA, logistic regression, neural nets). Not needed for tree models.

### d) Pipeline + ColumnTransformer
A `Pipeline` chains steps (impute → scale). A `ColumnTransformer` applies different pipelines to different column groups (numeric vs categorical) and concatenates the results. This keeps preprocessing reproducible and avoids leakage when used inside cross-validation.

## 3. Code walkthrough
| Step | What the code does |
|---|---|
| Load | `load_wine(as_frame=True)` returns a DataFrame with 13 numeric features + `target` (0/1/2). |
| Create a categorical column | `target` is mapped to names (`class_0`, `class_1`, `class_2`) → `class_name` of dtype *object*; the numeric `target` is dropped. |
| Inject missing values | `df.loc[::20, "alcohol"] = np.nan` blanks every 20th row; `df.loc[::25, "class_name"] = np.nan` every 25th. (Rows 0,20,…,160 → 9 values; rows 0,25,…,175 → 8 values; total **17**.) |
| Separate columns | `select_dtypes(include=np.number)` → numeric list; `select_dtypes(exclude=np.number)` → categorical list. |
| Build preprocessor | Numeric: `SimpleImputer(median)` → `StandardScaler`. Categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder(sparse_output=False)`. |
| Fit & transform | `fit_transform(df)` returns a NumPy array: 13 numeric + 3 one-hot columns = **16 columns**. |
| Verify | `np.isnan(X_ready).sum()` = 0 → no missing values remain. |

### Full code
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

wine = load_wine(as_frame=True)
df = wine.frame.copy()
df["class_name"] = df["target"].map(dict(enumerate(wine.target_names))).astype("object")
df = df.drop(columns=["target"])

# Introduce a small number of missing values for demonstrating imputation
df.loc[::20, "alcohol"] = np.nan
df.loc[::25, "class_name"] = np.nan

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]), categorical_cols)
])

X_ready = preprocessor.fit_transform(df)
print("Original shape:", df.shape)
print("Transformed shape:", X_ready.shape)
print("Missing values before:", int(df.isna().sum().sum()))
print("Missing values after preprocessing:", int(np.isnan(X_ready).sum()))
print("Model-ready dataset created successfully.")
```

### Output
```text
Original shape: (178, 14)
Transformed shape: (178, 16)
Missing values before: 17
Missing values after preprocessing: 0
Model-ready dataset created successfully.
```

## 4. Interpretation
* Shape grows from 14 → 16 columns because the single categorical column became 3 one-hot columns (13 numeric + 3).
* Missing values dropped from 17 to 0 — imputation worked.
* The array is fully numeric, complete and standardized, so it is suitable for any ML model.

## 5. Viva questions
* **Why median instead of mean?** Median is robust to outliers.
* **Why One-Hot instead of Ordinal here?** Wine classes have no natural order.
* **Which models need scaling?** Distance- and gradient-based (KNN, SVM, PCA, LR); trees do not.
* **What is data leakage?** Using information from test data during fitting (e.g. fitting the scaler on all data before splitting).

## 6. Notes from debugging
* Runs without errors. `sparse_output` requires **scikit-learn ≥ 1.2** (older versions use `sparse=`).
* `train_test_split` is imported but unused (harmless).
* The program prints shapes only; the question says "display the transformed dataset" — add `print(X_ready[:5])` if your examiner wants to see rows (code left unchanged here as requested).
