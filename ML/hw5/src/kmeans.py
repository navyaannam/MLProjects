import numpy as np
import math


class KMeans(object):

    def __init__(self, data, option):
        self.data = data
        self.membership = None
        self.centroids = None
        self.option = option
        self.temp_data = None

    def fit(self, K):
        data = np.asmatrix(self.data[0])
        if self.option == 2:
            self.data[:, 2] = np.log(data[:, 2])
            self.data[:, 3] = np.log(data[:, 3])
        elif self.option == 3:
            for j in range(self.data.shape[1]):
                self.data[:, j] -= np.mean(self.data[:, j])
                self.data[:, j] /= np.std(self.data[:, j])
        elif self.option == 5:
            self.temp_data = self.data
            np.random.shuffle(self.data)
            self.data = self.data[0:int(self.data.shape[0]*.06), :]

        centroids = self.data[np.random.choice(self.data.shape[0], K, replace=False), :]
        membership = np.zeros(self.data.shape[0]).astype(int)
        centroids_temp = None
        while not np.array_equal(centroids_temp, centroids):
            centroids_temp = np.copy(centroids)
            for i, d in enumerate(self.data):
                if self.option == 4:
                    membership[i] = np.argmin(np.array([np.abs(d - c).sum() for c in centroids]))
                else:
                    membership[i] = np.argmin(np.array([np.sqrt(((d - c) ** 2).sum()) for c in centroids]))

            for i in range(centroids.shape[0]):
                centroids[i] = self.data[membership == i].mean(axis=0)

        self.centroids = np.copy(centroids)
        self.membership = np.copy(membership)

        if self.option == 5:
            self.data = self.temp_data
            self.membership = np.zeros(self.data.shape[0]).astype(int)
            for i, d in enumerate(self.data):
                self.membership[i] = np.argmin(np.array([np.sqrt(((d - c) ** 2).sum()) for c in centroids]))

    def predict(self):
        error = 0
        for i, c in enumerate(self.centroids):
            subset = self.data[self.membership == i]
            for i, d in enumerate(subset):
                error += ((d - c) ** 2).sum()
        return error
