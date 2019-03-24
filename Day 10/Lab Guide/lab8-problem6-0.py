from scipy.io import loadmat


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

print("Display X_train:")
print(X_train)

print('\n')

print("Dimension of y_train: {}".format(y_train.shape))

print("Display X_test:")
print(X_test)

print('\n')

print("Dimension of X_test: {}".format(X_test.shape))
print("Dimension of y_test: {}".format(y_test.shape))

print('\n')
