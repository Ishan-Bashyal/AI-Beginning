import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Step 1: Generate 20 closer 2D data points
np.random.seed(42)
cluster1 = np.random.normal(loc=[2, 2], scale=0.4, size=(10, 2))
cluster2 = np.random.normal(loc=[3, 2.5], scale=0.4, size=(10, 2))
X = np.vstack((cluster1, cluster2))

# Step 2: Scatter plot of the raw data
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], color='gray')
plt.title("Original Data Points (Closer Clusters)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# Step 3: Perform hierarchical clustering
linked = linkage(X, method='ward')

# Step 4: Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', distance_sort='ascending', show_leaf_counts=True)
plt.axhline(y=1.0, color='r', linestyle='--', label='Height = 1.0')
plt.title("Dendrogram (Closer Clusters)")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Cut dendrogram at a specific height (e.g., 1.0)
height_threshold = 1.0
cluster_labels = fcluster(linked, t=height_threshold, criterion='distance')

# Step 6: Scatter plot with cluster coloring
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='tab10', s=50)
plt.title(f"Clusters from Hierarchical Clustering (cut at height={height_threshold})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

height_threshold = 2.0
cluster_labels = fcluster(linked, t=height_threshold, criterion='distance')

# Step 6: Scatter plot with cluster coloring
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='tab10', s=50)
plt.title(f"Clusters from Hierarchical Clustering (cut at height={height_threshold})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

height_threshold = 0.5
cluster_labels = fcluster(linked, t=height_threshold, criterion='distance')

# Step 6: Scatter plot with cluster coloring
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='tab10', s=50)
plt.title(f"Clusters from Hierarchical Clustering (cut at height={height_threshold})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()


