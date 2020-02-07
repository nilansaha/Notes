import time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import completeness_score

class KMeans:
    def __init__(self, n_clusters = 2, max_iter = 10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.labels_ = []
    
    # def _euclidian(self, x, y):
    #     total = 0
    #     for i in range(len(x)):
    #         total += (x[i] - y[i]) ** 2
    #     return total ** 0.5

    def _euclidian(self, x, y):
        return np.sum((x - y) ** 2)**0.5

    # def _get_centroid(self, x):
    #     datapoints = len(x)
    #     features = len(x[0])
    #     centroid = []
    #     for i in range(features):
    #         centroid.append(sum([x[j][i] for j in range(datapoints)])/datapoints)
    #     return centroid
    
    def _get_centroid(self, x):
        return np.mean(x, axis = 0)
    
    def fit(self, X):
        centroid_idx = np.random.choice(len(X), self.n_clusters, replace = False)
        centroids = X[centroid_idx]
        last_centroids = None
        for _ in range(self.max_iter):
            last_centroids = centroids
            labels = []
            for i in range(len(X)):
                local_label = []
                for centroid in centroids:
                    dist = self._euclidian(centroid, X[i])
                    local_label.append(dist)
                labels.append(np.argmin(local_label))
            self.labels_ = labels
            # print(self.labels_)

            sorted_label_idx = np.argsort(self.labels_)

            group = []
            last_label = self.labels_[sorted_label_idx[0]]
            i = 0
            new_centroids = []
            for idx in sorted_label_idx:
                label = self.labels_[idx]
                if label != last_label or i == len(sorted_label_idx) - 1:
                    new_centroids.append(self._get_centroid(group))
                    group = []
                last_label = self.labels_[idx]
                group.append(X[idx])
                i += 1
            centroids = np.array(new_centroids)
            if (centroids == last_centroids).all():
                break

X, y = make_blobs(n_samples = 10000, centers = 4, n_features = 50, random_state = 0)

start = time.process_time()
kmeans = KMeans(n_clusters = 4, max_iter = 20)
kmeans.fit(X)
print('Custom Kmeans Score:', completeness_score(y, kmeans.labels_))
print('Custom Kmeans Time', time.process_time() - start)


from sklearn.cluster import KMeans

start = time.process_time()
kmeans = KMeans(n_clusters = 4)
kmeans.fit(X)
print('Sklearn Kmeans Score:', completeness_score(y, kmeans.labels_))
print('SKlearn Kmeans Time',time.process_time() - start)

