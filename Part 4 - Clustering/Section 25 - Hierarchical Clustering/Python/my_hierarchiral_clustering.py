import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values

# import scipy.cluster.hierarchy as sch
# dendrogram = sch.dendrogram(sch.linkage(x, method = "ward"))
# plt.title("Dendrogram")
# plt.xlabel('Customers')
# plt.ylabel('Euclidean distances')
# plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, linkage = 'ward')
y_kmeans = hc.fit_predict(x)


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()