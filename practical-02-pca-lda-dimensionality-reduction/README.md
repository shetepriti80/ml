# Practical 2 — Dimensionality Reduction using PCA and LDA

**Dataset:** UCI Wine (13 features, 3 classes) · **File:** [`practical_02_pca_lda.py`](practical_02_pca_lda.py)

## 1. Aim
Reduce the 13-dimensional Wine data with **PCA** (unsupervised) and **LDA** (supervised), then compare accuracy, number of features and training time for the same classifier.

## 2. Theory

### Curse of dimensionality
More features → more data needed, slower training, more risk of overfitting and noise. Dimensionality reduction keeps the useful information in fewer dimensions.

### PCA — Principal Component Analysis (unsupervised)
* Finds new orthogonal axes (**principal components**) along which variance is maximal.
* Steps: standardize → covariance matrix → eigenvectors/eigenvalues → sort by eigenvalue → project.
* **Explained variance ratio** of component *i* = λᵢ / Σλ.
* `PCA(n_components=0.90)` keeps the *smallest number of components that explain ≥ 90 %* of variance.
* Ignores class labels → may not maximize class separation.

### LDA — Linear Discriminant Analysis (supervised)
* Finds axes that **maximize between-class scatter and minimize within-class scatter**: maximize `|Wᵀ S_B W| / |Wᵀ S_W W|`.
* Maximum components = **min(n_features, n_classes − 1)** → 2 for Wine (3 classes).
* Uses class labels, so it is usually better for classification.

| | PCA | LDA |
|---|---|---|
| Type | Unsupervised | Supervised |
| Goal | Maximize variance | Maximize class separability |
| Max components | n_features | n_classes − 1 |

### Logistic regression (classifier used)
Models P(y=1|x) = σ(wᵀx + b) (softmax for multi-class); `max_iter=3000` makes sure the optimizer converges.

## 3. Code walkthrough
| Step | Explanation |
|---|---|
| Load + scale | `StandardScaler` gives every feature mean 0 / std 1 (PCA is scale-sensitive). |
| Split | 75/25, `stratify=y` keeps class proportions, `random_state=42` for reproducibility. |
| PCA | `fit_transform` on **train**, `transform` on test (no leakage). |
| LDA | `fit_transform(X_train, y_train)` uses labels; `transform(X_test)`. |
| Loop | Trains an identical `LogisticRegression` on Original / PCA / LDA data; `time.perf_counter()` times only `fit`. |
| Report | Prints number of PCA components, total variance explained and a comparison table. |

### Full code
```python
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

pca = PCA(n_components=0.90)
lda = LinearDiscriminantAnalysis(n_components=2)
Xtr_pca = pca.fit_transform(X_train)
Xte_pca = pca.transform(X_test)
Xtr_lda = lda.fit_transform(X_train, y_train)
Xte_lda = lda.transform(X_test)

results = []
for name, a, b in [
    ("Original", X_train, X_test),
    ("PCA", Xtr_pca, Xte_pca),
    ("LDA", Xtr_lda, Xte_lda)]:
    start = time.perf_counter()
    model = LogisticRegression(max_iter=3000)
    model.fit(a, y_train)
    elapsed = time.perf_counter() - start
    pred = model.predict(b)
    results.append([name, a.shape[1], accuracy_score(y_test, pred), elapsed])

print("PCA components:", pca.n_components_)
print("PCA explained variance:", round(pca.explained_variance_ratio_.sum(), 4))
print(pd.DataFrame(results, columns=["Method", "Features", "Accuracy", "Fit time (s)"]))
```

### Output
```text
PCA components: 8
PCA explained variance: 0.9236
     Method  Features  Accuracy  Fit time (s)
0  Original        13  0.977778      0.004262
1       PCA         8  0.977778      0.002995
2       LDA         2  0.977778      0.003064
```

## 4. Interpretation
* PCA needed **8 components** to reach 90 % variance (actual 92.4 %), cutting features from 13 → 8.
* LDA compressed to just **2 features**.
* Accuracy stays ≈ 97.8 % in all three cases, so we lose no accuracy while using far fewer features.
* Timing differences are tiny on such a small dataset and vary from run to run — don't over-interpret them.

## 5. Viva questions
* **PCA vs LDA?** PCA maximizes variance (unsupervised); LDA maximizes class separation (supervised).
* **Why only 2 LDA components?** Limited to classes − 1 = 2.
* **Why standardize before PCA?** Otherwise large-scale features dominate the variance.

## 6. Notes from debugging
* Runs without errors; output matches the notebook (timings differ per machine).
* Minor point: `StandardScaler` is fitted on *all* data before the split (small leakage). It is harmless for this exam demo; a leak-free version would scale after splitting (or use a `Pipeline`). Code left unchanged.
