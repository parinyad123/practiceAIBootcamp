import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat


'''
Step 0: Load the data from './data/lab7data1.mat'
'''

dataset = loadmat('data/lab7data1.mat')
print(dataset.keys())

X_train = np.c_[np.ones_like(dataset['X']), dataset['X']]
y_train = dataset['y']

X_cv = np.c_[np.ones_like(dataset['Xval']), dataset['Xval']]
y_cv = dataset['yval']

print("Dimensions of X train: {}".format(X_train.shape))
print("Dimensions of y train: {}".format(y_train.shape))

print("Dimensions of X cv: {}".format(X_cv.shape))
print("Dimensions of y cv: {}".format(y_cv.shape))

print('\n')


'''
Step 1: Make 2D-plot where the x indicates the change in water level and 
the y axis indicates the amount of water flowing out of the dam
'''

plt.scatter(X_train[:,1], y_train, s=50, c='r', marker='x', linewidths=1)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.ylim(bottom=0);

