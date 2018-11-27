import pandas as pd
import numpy as np


def load_data(file_name):
    data = pd.read_csv(file_name)
    data = np.array(data)
    return data

# funtion for standardizing the data (zero-mean and standard deviation = 1)
def self_preprocessing(arg):
    arg_ = (np.array(arg,dtype=np.float64))
    arg_ = (arg_ - arg_.mean(axis=0)) / arg_.std(axis=0)
    return arg_

# A function cluster = mykmeans(X, k) that clusters data X 2 Rnp (n number of
# objects and p number of attributes) into k clusters.
def mykmeans(X, k):

    k = k

    random_C = np.random.choice(X.shape[0], k, False)
    C = X[np.random.choice(X.shape[0], k, replace=False), :]

    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)

    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(X))

    # Error func. - Distance between new centroids and old centroids
    error = np.linalg.norm(C - C_old, None)

    while error != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X)):

            distances = np.linalg.norm(X[i] - C, axis=1)
            cluster = np.argmin(distances)
            clusters[i] = cluster

        # Storing the old centroid values
        C_old = C[:]
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            np.array(points)
            C[i] = np.mean(points, axis=0)

        error = np.linalg.norm(C - C_old, None)


    return clusters, C

if __name__ == '__main__':


    data = load_data("your file name")
    standrardized_data = self_preprocessing(data)
    n = 3 # it means number of clusters. I have written 3 as an example

# for Kmeans function, the data must contain all the numerical values.
# you might need to preprocess the data first (the data contains all the numeric values) in order to run it correctly

    mykmeans(standrardized_data,n)[1] # for clusters
    mykmeans(standrardized_data,n)[0] # for centroids
