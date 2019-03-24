import numpy as np
from numpy.linalg import svd
from scipy.io import loadmat
import matplotlib.pyplot as plt


def featureNormalize(X):
    """
    Normalize the dataset X

    :param X:
    :return:
    """

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_normalized = (X - mu) / sigma

    return X_normalized, mu, sigma


def pca(X):
    """
    Compute eigenvectors of the covariance matrix X

    :param X:
    :return:
    """

    number_of_examples = X.shape[0]

    sigma = (1 / number_of_examples) * np.dot(X.T, X)
    U, S, V = svd(sigma) # svd ใช้หา eigen vector ใช้ U ลงขนาด dim

    return U, S, V


'''
Step 0: Load the dataset.
'''

dataset = loadmat("./data/lab9data2.mat")
print(dataset.keys(), '\n')

X = dataset["X"]


'''
Step 1: Normalize the dataset X
'''

X_normalized, mu, std = featureNormalize(X)
print("Values of the first 3 normalized dataset X: \n{}\n".format(X_normalized[:3]))


'''
Step 2: Compute the principal components and the diagonal matrix
'''

U, S, _ = pca(X_normalized)

print("The principal components: \n{}\n".format(U))
print("The diagonal matrix: \n{}\n".format(S))
