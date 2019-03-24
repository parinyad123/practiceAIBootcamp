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

    U_reduced = U[:, :K]

    '''
    Method 1 (with for-loops)
    
    number_of_examples = X.shape[0]
    Reduced_representation = np.zeros((number_of_examples, K))

    for i in range(number_of_examples):
        for j in range(K):
            Reduced_representation[i, j] = np.dot(X[i, :], U_reduced[:, j])

    return Reduced_representation
    '''

    # Method 2 (without for-loop)
    Reduced_representation = np.dot(U_reduced.T, X.T)
    return Reduced_representation.T

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


'''
Step 3: Project the data onto 1 dimension
'''

reduced_representation = projectData(X_normalized, U, 1)
print("Projection of the first example: {}".format(reduced_representation[0][0]))
