import numpy as np
from k_means import KMeans
class BisectingKMeans:
    def distance(self, x, y):
        return np.sum((x - y)**2)
        
    def __init__(self, n_clusters: int, max_iter=10000, type_init=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.type_init = type_init
    
    def sse(self, x):
        centroid = np.mean(x, 0)
        errors = np.linalg.norm(x-centroid, ord=2, axis=1)
        return np.sum(errors)
    
    def choise_cluster(self, x):
        id_cluster = -1
        best_dist = -1
        sses = [0 for i in range(self.n_clusters)]
        for i in range(self.n_clusters):
            sses[i] = self.sse(x[self.groups == i, :])
            if best_dist == -1 or sses[i] > best_dist:
                best_dist = sses[i]
                id_cluster = i
        return id_cluster
    
    def fit(self, x):
        for i in range(self.n_clusters - 1):
            if i == 0:
                i_k_means = KMeans(2, max_iter=self.max_iter, type_init=self.type_init)
                i_k_means.fit(x)
                self.groups = i_k_means.predict(x)
                self.zeros = i_k_means.zeros;
                continue
            id_cluster = self.choise_cluster(x)
            i_k_means = KMeans(2, max_iter=self.max_iter, type_init=self.type_init)
            i_k_means.fit(x[self.groups == id_cluster, :])
            temp = i_k_means.predict(x[self.groups == id_cluster, :])
            k = 0
            self.zeros[id_cluster] = i_k_means.zeros[0]
            self.zeros.append(i_k_means.zeros[1])
            for j in range(self.groups.size):
                if self.groups[j] == id_cluster:
                    if temp[k] == 1:
                        self.groups[j] = i + 1
                    k += 1
    
    def predict(self, x):
        groupses = np.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            d_min = -1
            for k, j in enumerate(self.zeros):
                if self.distance(j, x[i,:]) < d_min or d_min == -1:
                    d_min = self.distance(j, x[i,:])
                    p  = k
            groupses[i] = p
        return groupses