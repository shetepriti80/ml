# Practical 6 — Binary Classification using Logistic Regression

**Dataset:** Breast Cancer Wisconsin (569 samples, 30 features, 2 classes) · **File:** [`practical_06_logistic_regression.py`](practical_06_logistic_regression.py)

## 1. Aim
Build a logistic-regression classifier for malignant vs benign tumours and evaluate it with a confusion matrix, precision, recall, F1-score and accuracy.

## 2. Theory

### Logistic regression
Despite the name, it is a **classification** algorithm. It computes a linear score z = wᵀx + b and squashes it with the **sigmoid**:

P(y = 1 | x) = σ(z) = 1 / (1 + e⁻ᶻ)

Predict class 1 if P ≥ 0.5. Parameters are learned by minimizing the **log-loss (cross-entropy)**:
−(1/n) Σ [ y log p + (1 − y) log(1 − p) ] (with L2 regularization by default in sklearn, `C = 1.0`).

### Confusion matrix (binary)
|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN | FP |
| **Actual 1** | FN | TP |

### Metrics
| Metric | Formula | Answers |
|---|---|---|
| Accuracy | (TP+TN)/total | overall correctness |
| Precision | TP/(TP+FP) | of predicted positives, how many are right? |
| Recall (sensitivity) | TP/(TP+FN) | of real positives, how many were found? |
| F1-score | 2·P·R/(P+R) | harmonic mean of precision and recall |

In sklearn's Breast Cancer data the labels are **0 = malignant, 1 = benign**, so "positive" (1) = benign for `precision_score`/`recall_score`.

### Other concepts
* **Stratified split** keeps the class ratio equal in train and test.
* **Standardization** helps the optimizer converge and keeps the regularization fair.

## 3. Code walkthrough
| Step | Explanation |
|---|---|
| Load | `load_breast_cancer(return_X_y=True)`. |
| Split | 80/20, `stratify=y`. |
| Pipeline | `make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))` → scaling is learned from the training set only. |
| Predict | `model.predict(X_test)`. |
| Metrics | `confusion_matrix`, `classification_report` (per-class P/R/F1), plus overall accuracy, precision, recall, F1. |

### Full code
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score,
                             recall_score, f1_score)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=5000))
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=load_breast_cancer().target_names))
print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1-score :", f1_score(y_test, pred))
```

### Output
```text
Confusion Matrix:
 [[41  1]
 [ 1 71]]
              precision    recall  f1-score   support

   malignant       0.98      0.98      0.98        42
      benign       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114

Accuracy : 0.9824561403508771
Precision: 0.9861111111111112
Recall   : 0.9861111111111112
F1-score : 0.9861111111111112
```

## 4. Interpretation
* Confusion matrix `[[41, 1], [1, 71]]` → TN = 41, FP = 1, FN = 1, TP = 71 on 114 test samples.
* **Accuracy 98.2 %**, precision = recall = F1 ≈ 98.6 % for the benign class.
* Only 2 mistakes in total. In cancer screening, **false negatives** (a malignant tumour predicted benign) are the costly error, so recall for the malignant class (0.98) matters most.

## 5. Viva questions
* **Why not use linear regression for classification?** Outputs are unbounded and not probabilities.
* **Precision vs recall?** Precision penalizes false alarms; recall penalizes missed cases.
* **What does `C` do?** Inverse of regularization strength (smaller C = stronger regularization).

## 6. Notes from debugging
Runs without errors; output identical to the notebook.
