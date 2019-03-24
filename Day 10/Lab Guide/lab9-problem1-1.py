import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def findClosestCentroids(X, centroids):
    """
    Returns the closest centroids in indices for a dataset X,
    where each row is a single example.

    :param X:
    :param centroids:
    :return:
    """

    number_of_centroids = centroids.shape[0]
    indices = np.zeros((X.shape[0], 1))
    temp = np.zeros((centroids.shape[0], 1))

    for i in range(X.shape[0]):
        for j in range(number_of_centroids):
            distance = X[i, :] - centroids[j, :]
            length = np.sum(distance ** 2)
            temp[j] = length

        indices[i] = np.argmin(temp) + 1  # + 1 since we start counting from 1 !

    return indices

'''
Step 0: Load the dataset.
'''

dataset = loadmat("./data/lab9data1.mat")
print(dataset.keys(), '\n')

X = dataset["X"]


'''
Step 1: Set up an initial centroids and 
display the closest centroids for the first 3 examples
'''

initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

indices = findClosestCentroids(X, initial_centroids)
print("Closest centroids for the first 3 examples: \n{}".format(indices[0:3]))
