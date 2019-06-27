import sys
import numpy as np
import pandas as pd
from kmeans import KMeans
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
    print("Invalid amount of arguments")
    sys.exit(1)


file_name = sys.argv[1]
K = int(sys.argv[2])
c_option = sys.argv[3]

data = pd.read_csv(file_name, sep=',', quotechar='"', header=0)
data = data[['latitude', 'longitude', 'reviewCount', 'checkins']]
X = np.array(data)

if __name__ == '__main__':

    km = KMeans(X, int(c_option))
    km.fit(K)
    print("WC-SSE={}".format(km.predict()))
    for ci, c in enumerate(km.centroids):
        print("Centroid{}={}".format(ci + 1, c))

