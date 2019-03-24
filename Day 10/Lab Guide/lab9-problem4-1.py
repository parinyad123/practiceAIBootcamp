import numpy as np
from numpy.linalg import svd
from scipy.io import loadmat
import matplotlib.pyplot as plt


'''
Step 0: Load the dataset
'''

dataset = loadmat("data/lab9faces.mat")
print(dataset.keys(), '\n')

X = dataset["X"]


'''
Step 1: Visualize the dataset
'''

figure, axes = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))

for i in range(0, 100, 10):
    for j in range(10):
        axes[int(i / 10), j].imshow(X[i + j, :].reshape(32, 32, order="F"), cmap="gray")
        axes[int(i / 10), j].axis("off")
