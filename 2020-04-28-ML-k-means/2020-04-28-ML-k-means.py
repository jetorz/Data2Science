import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = np.loadtxt('X.txt', delimiter=' ')
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
