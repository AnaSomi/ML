import numpy as np

class KMeans:
    def distance(self, x, y):
        return np.sum((x - y)**2)
    def  __init__(self, n_clusters: int, max_iter=10000, type_init=0):
        self.n_clusters = n_clusters
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
    
    def fit(self, x: np.ndarray):
        self.init_centers(x)
        for p in range(self.max_iter):
            self.new_zeros  = []
            for i in range(self.n_clusters):
                temp = [0 for j in range(x.shape[1])]
                self.new_zeros.append(temp)
            
            cnts = [0 for j in range(self.n_clusters)]
            for i in range(x.shape[0]):
                d_min = -1
                for k, j in enumerate(self.zeros):
                    if self.distance(j, x[i,:]) < d_min or d_min == -1:
                        d_min = self.distance(j, x[i,:])
                        p  = k
                cnts[p] += 1
                self.new_zeros[p] += x[i,:]
            for i in range(self.n_clusters):
                self.new_zeros[i] /= cnts[i]
                self.new_zeros[i] = list(self.new_zeros[i])
            if self.new_zeros == self.zeros:
                break
            self.zeros = self.new_zeros
    
    def predict(self, x: np.ndarray):
        groupses = np.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            d_min = -1
            for k, j in enumerate(self.zeros):
                if self.distance(j, x[i,:]) < d_min or d_min == -1:
                    d_min = self.distance(j, x[i,:])
                    p  = k
            groupses[i] = p
        return groupses