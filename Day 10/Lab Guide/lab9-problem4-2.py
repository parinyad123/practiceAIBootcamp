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
    sigma = (1/number_of_examples) * np.dot(X.T, X)
    U, S, V = svd(sigma)

    return U, S, V


def projectData(X, U, K):
    """
    Computes the reduced data representation when projecting only onto
    the top K eigenvectors

    :param X: Dataset
    :param U: Principal components
    :param K: The desired number of dimensions to reduce
    :return:
    """

    number_of_examples = X.shape[0]
    U_reduced = U[:, :K]

    Reduced_representation = np.zeros((number_of_examples, K))

    for i in range(number_of_examples):
        for j in range(K):
            Reduced_representation[i, j] = np.dot(X[i, :], U_reduced[:, j])

    return Reduced_representation


def recoverData(Z, U, K):
    """
    Recovers an approximation of the original data when using the projected data

    :param Z: Reduced representation
    :param U: Principal components
    :param K: The desired number of dimensions to reduce
    :return:
    """

    number_of_examples = Z.shape[0]
    number_of_features = U.shape[0]

    X_recovered = np.zeros((number_of_examples, number_of_features))
    U_reduced = U[:, :K]

    for i in range(number_of_examples):
        X_recovered[i, :] = np.dot(Z[i, :], U_reduced.T)

    return X_recovered


'''
Step 0: Load the dataset
'''

dataset = loadmat("data/lab9faces.mat")
print(dataset.keys(), '\n')

X = dataset["X"]


'''
Step 1: Visualize the dataset
'''

figure1, axes1 = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))

for i in range(0, 100, 10):
    for j in range(10):
        axes1[int(i / 10), j].imshow(X[i + j, :].reshape(32, 32, order="F"), cmap="gray")
        axes1[int(i / 10), j].axis("off")


'''
Step 2: Normalize the dataset and run PCA on it
'''

X_normalized = featureNormalize(X)[0]  # [0] for converting from 'tuple' to 'array'

# Run PCA
U = pca(X_normalized)[0]

# Visualize the top 100 eigenvectors found
U_reduced = U[:, :100].T

figure2, axes2 = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))

for i in range(0, 100, 10):
    for j in range(10):
        axes2[int(i / 10), j].imshow(U_reduced[i + j, :].reshape(32, 32, order="F"), cmap="gray")
        axes2[int(i / 10), j].axis("off")
