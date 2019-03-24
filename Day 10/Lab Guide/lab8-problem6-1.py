from scipy.io import loadmat
from sklearn.svm import SVC
import numpy as np


'''
Step 0: Load the data from './data/lab8spam_train.mat' and './data/lab8spam_test.mat'
'''

dataset_train = loadmat('data/lab8spam_train.mat')
print(dataset_train.keys())

dataset_test = loadmat('data/lab8spam_test.mat')
print(dataset_test.keys())

print('\n')

X_train = dataset_train["X"]
y_train = dataset_train["y"]

X_test = dataset_test["Xtest"]
y_test = dataset_test["ytest"]

print("Dimension of X_train: {}".format(X_train.shape))
print("Dimension of y_train: {}".format(y_train.shape))

print('\n')

print("Dimension of X_test: {}".format(X_test.shape))
print("Dimension of y_test: {}".format(y_test.shape))

print('\n')


'''
Step 1: Train SVM for spam classification
'''

C = 0.1
spam_classifier = SVC(C=0.1, kernel="linear")
spam_classifier.fit(X_train, y_train.ravel())

print("Training Accuracy: {} %".format(spam_classifier.score(X_train, y_train.ravel()) * 100))

print("Display predicted y:")
print(spam_classifier.predict(X_test).reshape(-1, 1))
print("Test Accuracy: {} %".format(np.mean(spam_classifier.predict(X_test).reshape(-1, 1) == y_test) * 100))
