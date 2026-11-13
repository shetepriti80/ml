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