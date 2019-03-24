import numpy as np
from scipy.io import loadmat


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

