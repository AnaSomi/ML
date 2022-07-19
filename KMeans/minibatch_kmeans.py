import numpy as np

class MiniBatchKMeans():
    def distance(self, x, y):
        return np.sum((x - y)**2)

    def __init__(self, n_clusters: int, size_batch: int, max_iter=10000, type_init=0):
        self.n_clusters = n_clusters
        self.size_batch = size_batch
        self.max_iter = max_iter
        self.type_init = type_init

    def init_centers(self, x):
        if (self.type_init == 2):
            self.init_adv(x)
        elif (self.type_init == 1):
            self.init_random(x)
        else:
            self.init_first(x)
    
    def init_first(self, x: np.ndarray):
        self.zeros = [list(x[i, :]) for i in range(self.n_clusters)]

    def init_random(self, x: np.ndarray):
        idx = np.random.choice(x.shape[0], size=self.n_clusters, replace=False)
        self.zeros = list(x[idx, :])
        for i in range(self.n_clusters):
            self.zeros[i] = list(self.zeros[i])
    
    def init_adv(self, x: np.ndarray):
        random_indices = np.random.choice(x.shape[0], size=1, replace=False)
        self.zeros = list(x[random_indices, :])
        for i in range(self.n_clusters - 1):
            dist = []
            for j in range(x.shape[0]):
                point = x[j, :]
                d = 0
                for k in range(len(self.zeros)):
                    temp_dist = self.distance(point, self.zeros[k])
                    if k == 0:
                        d = temp_dist
                    d = min(d, temp_dist)
                dist.append(d)
            dist = np.array(dist)
            next_centroid = x[np.argmax(dist), :]
            self.zeros.append(list(next_centroid))
        for i in range(self.n_clusters):
            self.zeros[i] = list(self.zeros[i])

    def fit(self, x, iter_batch):
        self.init_centers(x)
        for i in range(iter_batch):
            x_batch = x[np.random.choice(x.shape[0], size=self.size_batch, replace=True)]
            cnts = np.zeros(self.n_clusters)
            idxs = np.empty(x_batch.shape[0], dtype=np.int)
            for j, y in enumerate(x_batch):
                idxs[j] = np.argmin(((self.zeros - y)**2).sum(1))
            self.zeros = np.array(self.zeros)
            for j, y in enumerate(x_batch):
                cnts[idxs[j]] += 1
                eta = 1.0 / cnts[idxs[j]]
                self.zeros[idxs[j]] = (1.0 - eta) * self.zeros[idxs[j]] + eta * y
            self.zeros = list(self.zeros)
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