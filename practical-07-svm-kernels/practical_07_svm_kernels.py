import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

results = []
for kernel in ["linear", "rbf", "poly"]:
    model = make_pipeline(StandardScaler(),
                          SVC(kernel=kernel, C=1.0, gamma="scale"))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append([kernel, accuracy_score(y_test, pred),
                    precision_score(y_test, pred),
                    recall_score(y_test, pred),
                    f1_score(y_test, pred)])

print(pd.DataFrame(results, columns=[
    "Kernel", "Accuracy", "Precision", "Recall", "F1-score"]))