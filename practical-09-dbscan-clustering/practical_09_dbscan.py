import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

X, y = load_wine(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# eps is data-dependent; experiment with nearby values if all points
# become noise or a single cluster is produced.
dbscan = DBSCAN(eps=2.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

cluster_labels = set(labels) - {-1}
noise_count = int(np.sum(labels == -1))
print("Number of clusters:", len(cluster_labels))
print("Number of noise points:", noise_count)
print("Cluster labels:", np.unique(labels))

X_2d = PCA(n_components=2).fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
plt.title("DBSCAN clustering (noise label = -1)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Parameter sensitivity experiment
for eps in [1.5, 2.0, 2.5, 3.0]:
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(X_scaled)
    print(f"eps={eps}: clusters={len(set(lab)-{-1})}, "
          f"noise={np.sum(lab == -1)}")