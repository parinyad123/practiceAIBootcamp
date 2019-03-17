import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


def gradientDescent(theta, X_with_interceptor, y, learning_rate, training_step, lambda_param):
    """
    Compute gradient descent w.r.t. the given inputs

    :param theta: initial theta
    :param X_with_interceptor: feature matrix
    :param y: target vector
    :param learning_rate:
    :param training_step:
    :param lambda_param:
    :return: theta and the list of costs in each training step
    """

    J_history = []

    for i in range(training_step):
        cost = linearRegressionCostFunction(theta, X_with_interceptor, y, lambda_param)
        gradient = linearRegressionGradientComputation(theta, X_with_interceptor, y, lambda_param)

        theta = theta - (learning_rate * gradient)
        J_history.append(cost)

    return theta, J_history

def learningCurve(theta, X_train, y_train, X_cv, y_cv, lambda_param):
    """

    :param X_train:
    :param y_train:
    :param X_cv:
    :param y_cv:
    :param lambda_param:
    :return:
    """
    number_examples = y_train.shape[0]
    J_train, J_cv = [], []

    for i in range(1, number_examples + 1):
        theta, _ = gradientDescent(theta, X_train[:i, :], y_train[:i, :], 0.001, 3000, lambda_param)

        cost_train = linearRegressionCostFunction(theta, X_train[0:i, :], y_train[:i, :], lambda_param)
        J_train.append(cost_train)

        cost_cv = linearRegressionCostFunction(theta, X_cv, y_cv, lambda_param)
        J_cv.append(cost_cv)

    return J_train, J_cv


def linearRegressionCostFunction(theta, X_with_interceptor, y, lambda_param):
    number_examples = y.shape[0]
    hypothesis_of_x = X_with_interceptor.dot(theta)

    J = (1 / (2 * number_examples)) * np.sum(np.square(hypothesis_of_x - y)) \
        + (lambda_param / number_examples) * np.sum(np.square(theta[1:]))

    return J

def linearRegressionGradientComputation(theta, X_with_interceptor, y, lambda_param):
    number_examples = y.shape[0]
    hypothesis_of_x = X_with_interceptor.dot(theta)

    gradient = (1 / number_examples) * (X_with_interceptor.T.dot(hypothesis_of_x - y)) \
               + (lambda_param / number_examples) * np.r_[[[0]], theta[1:].reshape(-1,1)]

    return gradient


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
Step 1: Test the cost function and gradient computation with initial thetas
'''

initial_theta = np.ones((X_train.shape[1], 1))
cost = linearRegressionCostFunction(initial_theta, X_train, y_train, 0)

print("Initial theta: {}".format(initial_theta.T)) # I just transpose here for beauty when printing out !
print("Cost w.r.t. initial theta is: {}".format(cost))

gradient = linearRegressionGradientComputation(initial_theta, X_train, y_train, 0)
print("Gradient w.r.t. initial theta is: {}".format(gradient.T)) # I just transpose here for beauty when printing out !

theta, J_history = gradientDescent(initial_theta, X_train, y_train, 0.001, 4000, 0)

plt.figure(2)
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

"""
Step 2: Plot the fitted line
"""

plt.figure(3)
plt.scatter(X_train[:, 1] ,y_train, marker="x", color="r")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")

x_value=[x for x in range(-50,40)]
y_value=[theta[0] + y * theta[1] for y in x_value]

plt.plot(x_value, y_value, color="b")
plt.ylim(-5,40)
plt.xlim(-50,40)

'''
Step 3: Plot learning curves
'''

J_train, J_cv = learningCurve(initial_theta, X_train, y_train, X_cv, y_cv, 0)

plt.figure(4)
plt.plot(range(12), J_train, label="Train")
plt.plot(range(12), J_cv, label="Cross Validation", color="r")
plt.title("Learning Curve for Linear Regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()


'''
Step 4: Fit the original X to the degree 8
'''

# Generate a new feature matrix consisting of all polynomial combinations of the features with degree less
# than or equal to degree 8
# See https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
polynomial = PolynomialFeatures(degree=8)
X_train_polynomial = polynomial.fit_transform(X_train[:, 1].reshape(-1, 1))

print('After fitting the training dataset to degree 8')
print(X_train_polynomial)

X_cv_polynomial = polynomial.fit_transform(X_cv[:, 1].reshape(-1, 1))

'''
Step 5: Normalize the polynomial features
'''

# Standardize features by removing the mean and scaling to unit variance
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
normalizer = StandardScaler()

X_train_polynomial_normalized = normalizer.fit_transform(X_train_polynomial)
print('After Normalizing')
print(X_train_polynomial_normalized)

X_cv_polynomial_normalized = normalizer.fit_transform(X_cv_polynomial)

'''
Step 6: Plot learning curves for the polynomial features
'''

initial_theta_polynomial = np.ones((X_train_polynomial_normalized.shape[1],1))
J_train_polynomial, J_cv_polynomial = learningCurve(initial_theta_polynomial, X_train_polynomial_normalized,
                                                    y_train, X_cv_polynomial_normalized, y_cv,0)

plt.figure(5)
plt.plot(range(12), J_train_polynomial, label="Train")
plt.plot(range(12), J_cv_polynomial, label="Cross Validation", color="r")
plt.title("Learning Curve for Linear Regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()
