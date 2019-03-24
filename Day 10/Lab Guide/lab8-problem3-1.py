import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC


def findBestCandSigma(X, y, step_values):

    best_score = 0
    best_c = 0
    best_gamma = 0

    for i in step_values:
        c = i

        for j in step_values:
            gamma = j

            classifier = SVC(C=c, gamma=gamma)
            classifier.fit(X, y)

            score = classifier.score(X, y)
            # หาจุดที่ดีที่สุด
            if score > best_score:
                best_score = score
                best_c = c
                best_gamma = gamma

    return best_c, best_gamma



'''
Step 0: Load the data from './data/lab8data3.mat'
'''

dataset = loadmat('data/lab8data3.mat')
print(dataset.keys())

X_train = dataset["X"]
y_train = dataset["y"]
X_cv = dataset["Xval"]
y_cv = dataset["yval"]

print("Dimension of X: {}".format(X_train.shape))
print("Dimension of y: {}".format(y_train.shape))

number_of_row = X_train.shape[0]

positives = (y_train == 1).reshape(number_of_row, 1)
negatives = (y_train == 0).reshape(number_of_row, 1)

print('\n')


'''
Step 1: Output the best C and gamma to use. To select these values, 
we suggest trying value in multiplicative steps (e.g. [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
'''
# ลองเพิ่มค่า values
values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
c, gamma = findBestCandSigma(X_cv, y_cv.ravel(), values)

print("Best C: {}".format(c))
print("Best gamma: {}".format(gamma))


'''
Step 2: Instantiate an SVM classifier based on the best C and gamma. 
After that, plot the decision boundary
'''

classifier = SVC(C=c, gamma=gamma)
classifier.fit(X_train, y_train.ravel())

plt.figure()
plt.scatter(X_train[positives[:, 0], 0], X_train[positives[:, 0], 1], c="r", marker="+", s=50)
plt.scatter(X_train[negatives[:, 0], 0], X_train[negatives[:, 0], 1], c="y", marker="o", s=50)

# plotting the decision boundary
X1, X2 = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), num=300),
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), num=300))

plt.contour(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), colors="b")
