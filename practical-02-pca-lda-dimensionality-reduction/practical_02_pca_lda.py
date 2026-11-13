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