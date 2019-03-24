from scipy.io import loadmat
from sklearn.svm import SVC
import numpy as np
import pandas as pd


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
Step 1: Build a vocabulary
'''

vocab_file = open("data/lab8vocab.txt", "r").read()
vocab_file = vocab_file.split('\n')

vocabulary = {}
for index, value in enumerate(vocab_file):
    if value is not '':
        ind_key, ind_value = value.split('\t')
        vocabulary[ind_key] = ind_value

print('Vocabulary list:')
print(vocabulary)


'''
Step 2: Train SVM for spam classification
'''

C = 0.1
spam_classifier = SVC(C=0.1, kernel="linear")
spam_classifier.fit(X_train, y_train.ravel())

print("Training Accuracy: {} %".format(spam_classifier.score(X_train, y_train.ravel()) * 100))
print('\n')


'''
Step 3: Display top predictors for being spams
'''

weights = spam_classifier.coef_[0] # dimension: (1899, )

weights_column = np.hstack((np.arange(1, 1900).reshape(1899, 1), weights.reshape(1899, 1)))

dataframe = pd.DataFrame(weights_column)
dataframe.sort_values(by=[1], ascending=False, inplace=True)

index_of_predictor = []
predictor_word = []

for i in dataframe[0][:15]:
    word = vocabulary.get(str(int(i)))

    index_of_predictor.append(i)
    predictor_word.append(word)

print("Top predictors of spam:")

for _ in range(15):
    print(predictor_word[_], "\t\t", round(dataframe[1][index_of_predictor[_] - 1], 6))
