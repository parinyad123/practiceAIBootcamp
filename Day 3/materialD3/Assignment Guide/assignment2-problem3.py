import tensorflow as tf
import numpy as np
import pandas as pd


def normalize(inputs):
    """
    Normalize an input array (feature scaling)

    Parameters
    ----------
    inputs : an input array

    Returns
    -------
    scaled_inputs : an input array in unit scale.
    """

    # Write the whole function

    return scaled_inputs


'''
Step 1: Read data from CSV file using Pandas
'''
data_from_CSV = pd.read_csv("data/boston.csv")

feature_CRIM = data_from_CSV['CRIM']
feature_ZN = data_from_CSV['CRIM']
feature_INDUS = data_from_CSV['INDUS']
feature_CHAS = data_from_CSV['CHAS']
feature_NOX = data_from_CSV['NOX']
feature_RM = data_from_CSV['RM']
feature_AGE = data_from_CSV['AGE']
feature_DIS = data_from_CSV['DIS']
feature_RAD = data_from_CSV['RAD']
feature_TAX = data_from_CSV['TAX']
feature_PTRATIO = data_from_CSV['PTRATIO']
feature_LSTAT = data_from_CSV['LSTAT']

target_MEDV = data_from_CSV['MEDV']

'''
Step 2: Rescale the training dataset
'''
scaled_feature_CRIM = # Write your code here
scaled_feature_ZN = # Write your code here
scaled_feature_INDUS = # Write your code here
scaled_feature_CHAS = # Write your code here
scaled_feature_NOX = # Write your code here
scaled_feature_RM = # Write your code here
scaled_feature_AGE = # Write your code here
scaled_feature_DIS = # Write your code here
scaled_feature_RAD = # Write your code here
scaled_feature_TAX = # Write your code here
scaled_feature_PTRATIO = # Write your code here
scaled_feature_LSTAT = # Write your code here

# Question: should we normalize target_MEDV?

'''
Step 3: Create placeholders for features Xs and target Y
'''
X_CRIM = tf.placeholder(tf.float32, name='X_CRIM')
X_ZN = tf.placeholder(tf.float32, name='X_ZN')
X_INDUS = tf.placeholder(tf.float32, name='X_INDUS')
X_CHAS = tf.placeholder(tf.float32, name='X_CHAS')
X_NOX = tf.placeholder(tf.float32, name='X_NOX')
X_RM = tf.placeholder(tf.float32, name='X_RM')
X_AGE = tf.placeholder(tf.float32, name='X_AGE')
X_DIS = tf.placeholder(tf.float32, name='X_DIS')
X_RAD = tf.placeholder(tf.float32, name='X_RAD')
X_TAX = tf.placeholder(tf.float32, name='X_TAX')
X_PTRATIO = tf.placeholder(tf.float32, name='X_PTRATIO')
X_LSTAT = tf.placeholder(tf.float32, name='X_LSTAT')

Y = tf.placeholder(tf.float32, name='Y')

'''
Step 4: Create thetas, initialized them to 0
'''
theta0 = tf.Variable(0.0, name='theta0')
theta_CRIM = # Write your code here
theta_ZN = # Write your code here
theta_INDUS = # Write your code here
theta_CHAS = # Write your code here
theta_NOX = # Write your code here
theta_RM = # Write your code here
theta_AGE = # Write your code here
theta_DIS = # Write your code here
theta_RAD = # Write your code here
theta_TAX = # Write your code here
theta_PTRATIO = # Write your code here
theta_LSTAT = # Write your code here

'''
Step 5: Define a hypothesis function to predict Y
'''
hypothesis_function = # Write your code here

'''
Step 6: Use the square error as the loss function
'''
loss_function = # Write your code here
tf.summary.scalar('total cost', loss_function)

'''
Step 7: Using gradient descent with learning rate of 0.3 to minimize loss
'''
optimizer = # Write your code here

with tf.Session() as session:
    '''
    Step 8: Initialize the necessary variables
    '''
    session.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./graphs/multivariate_linear_regression_feature_scaling', session.graph)

    '''
    Step 9: Train the model for 1,000 epochs
    '''
    for i in range(1000):
        # Write your code here

        writer.add_summary(summary, i)

        print("Epoch: {0}, cost = {1}".format(i+1, cost))

    '''
    Step 10: Prints the training cost and all thetas
    '''
    print("Optimization Finished!", '\n')
    print("Training cost = {}".format(cost))
    print("theta0 = {}".format(session.run(theta0)))
    print("theta_ZN = {}".format(session.run(theta_ZN)))
    print("theta_INDUS = {}".format(session.run(theta_INDUS)))
    print("theta_CHAS = {}".format(session.run(theta_CHAS)))
    print("theta_NOX = {}".format(session.run(theta_NOX)))
    print("theta_RM = {}".format(session.run(theta_RM)))
    print("theta_AGE = {}".format(session.run(theta_AGE)))
    print("theta_DIS = {}".format(session.run(theta_DIS)))
    print("theta_RAD = {}".format(session.run(theta_RAD)))
    print("theta_TAX = {}".format(session.run(theta_TAX)))
    print("theta_PTRATIO = {}".format(session.run(theta_PTRATIO)))
    print("theta_LSTAT = {}".format(session.run(theta_LSTAT)))


# Close the writer when you finished using it
writer.close()


