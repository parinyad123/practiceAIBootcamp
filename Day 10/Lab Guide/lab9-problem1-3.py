import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def computeCentroids(X, indices, number_of_centroids):
    """
    Returns the new centroids by computing the means of the data points
    assigned to each centroid.

    :param X:
    :param indices:
    :param number_of_centroids:
    :return:
    """

    number_of_examples = X.shape[0]
    number_of_features = X.shape[1]

    centroids = np.zeros((number_of_centroids, number_of_features))
    count = np.zeros((number_of_centroids, 1))

    for i in range(number_of_examples):
        index = int((indices[i] - 1)[0])
        centroids[index, :] += X[i, :]
        count[index] += 1

    return centroids / count


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


def runKmeans(X, centroids, indices, number_of_iterations):

    number_of_centroids = centroids.shape[0]

    for i in range(number_of_iterations):
        # Visualisation of data
        color = "rgb"

        plt.figure(i, figsize=[3, 3])

        for k in range(1, number_of_centroids + 1):
            cluster = (indices == k)
            plt.scatter(X[cluster[:, 0], 0], X[cluster[:, 0], 1], c=color[k-1], s=15)

        # visualize the new centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x", c="black", linewidth=3)
        plt.title("Iteration Number " + str(i))

        # compute the centroids mean
        centroids = computeCentroids(X, indices, number_of_centroids)

        # assign each training example to the nearest centroid
        indices = findClosestCentroids(X, centroids)

    plt.tight_layout()


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
print("Closest centroids for the first 3 examples: \n{}\n".format(indices[0:3]))

'''
Step 2: For each centroid, recompute the mean of the points that were assigned to it
'''

centroids = computeCentroids(X, indices, len(initial_centroids))
print("Centroids computed after initial finding of closet centroids: \n{}\n".format(centroids))


'''
Step 3: Visualize K-means clustering
'''

runKmeans(X, initial_centroids, indices, 6)
