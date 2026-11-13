# Practical 7 — Support Vector Machine with Linear and Non-linear Kernels

**Dataset:** Breast Cancer Wisconsin · **File:** [`practical_07_svm_kernels.py`](practical_07_svm_kernels.py)

## 1. Aim
Train SVMs with **linear, RBF and polynomial** kernels and compare accuracy, precision, recall and F1.

## 2. Theory

### Maximum-margin classifier
SVM finds the hyperplane wᵀx + b = 0 that separates the classes with the **largest margin** (distance 2/‖w‖ between the two margin boundaries). The training points that lie on the margin are the **support vectors** — only they define the boundary.

### Soft margin and C
Real data is not perfectly separable, so errors are allowed at a cost:
minimize ½‖w‖² + C Σ ξᵢ.
* **Large C** → few violations allowed, narrow margin (risk of overfitting).
* **Small C** → wider margin, more tolerance (risk of underfitting).

### Kernel trick
A kernel K(x, x′) computes the dot product in a higher-dimensional space **without** explicitly mapping the data, letting a linear separator in that space become a non-linear boundary in the original space.

| Kernel | Formula | Use |
|---|---|---|
| Linear | xᵀx′ | linearly separable data, many features |
| Polynomial | (γ xᵀx′ + r)ᵈ | polynomial interactions (default degree 3) |
| RBF (Gaussian) | exp(−γ‖x − x′‖²) | general-purpose non-linear boundary |

`gamma="scale"` sets γ = 1 / (n_features · X.var()).

**Scaling is mandatory** for SVM because kernels depend on distances.

## 3. Code walkthrough
| Step | Explanation |
|---|---|
| Load + split | 80/20 stratified, `random_state=42`. |
| Loop over kernels | For `linear`, `rbf`, `poly`: build `make_pipeline(StandardScaler(), SVC(kernel=…, C=1.0, gamma="scale"))`, fit, predict. |
| Metrics | accuracy, precision, recall, F1 appended to `results`, then shown as a DataFrame. |

### Full code
```python
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
```

### Output
```text
   Kernel  Accuracy  Precision    Recall  F1-score
0  linear  0.973684   0.985915  0.972222  0.979021
1     rbf  0.982456   0.986111  0.986111  0.986111
2    poly  0.912281   0.878049  1.000000  0.935065
```

## 4. Interpretation
* **RBF is best overall** (accuracy 98.2 %, F1 0.986), the linear kernel is close behind (97.4 %).
* **Polynomial is worst in accuracy (91.2 %)** but has recall 1.0 — it over-predicts the benign class, giving 10 false positives (precision 0.878) and no false negatives.
* Since the data is already almost linearly separable after scaling, the simple linear kernel does very well; the flexible RBF kernel gives a small additional gain.

## 5. Viva questions
* **What are support vectors?** Training points closest to / violating the margin that define the hyperplane.
* **What does the kernel trick do?** Computes high-dimensional dot products implicitly.
* **Effect of gamma in RBF?** Large γ → very local, wiggly boundary (overfit); small γ → smooth boundary.

## 6. Notes from debugging
Runs without errors; output identical to the notebook.
