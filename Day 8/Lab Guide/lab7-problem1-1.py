import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat



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

        cost_train = # Write code to calculate the cost of trainning dataset
        # Write code to store the computed cost in the array J_train

        cost_cv = # Write code to calculate the cost of cross validation dataset
        # Write code to store the computed cost in the array J_cv

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
Step 1: Make 2D-plot where the x indicates the change in water level and 
the y axis indicates the amount of water flowing out of the dam
'''

plt.figure(1)
plt.scatter(X_train[:, 1], y_train, s=50, c='r', marker='x', linewidths=1)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.ylim(bottom=0);

'''
Step 2: Test the cost function and gradient computation with initial thetas
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
Step 3: Plot the fitted line
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
Step 4: Plot learning curves
'''

J_train, J_cv = learningCurve(initial_theta, X_train, y_train, X_cv, y_cv, 0)

plt.figure(4)
plt.plot(range(12), J_train, label="Train")
plt.plot(range(12), J_cv, label="Cross Validation", color="r")
plt.title("Learning Curve for Linear Regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()
