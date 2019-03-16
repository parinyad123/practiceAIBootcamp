import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

'''
Step 0: Suppress warning logs
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

''' 
Step 1: Read in data from the .xls file
'''
DATA_FILE = 'data/fire_theft.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)

number_of_rows = len(list(sheet.get_rows()))
data = np.asarray([sheet.row_values(i) for i in range(1, number_of_rows)])
number_of_samples = number_of_rows - 1

'''
Step 2: Create placeholders for feature X (number of fire) and target Y (number of theft)
'''
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

'''
Step 3: Create theta0, theta1 and theta2; then, initialized them to 0
'''
theta0 = # Write your code here
theta1 = # Write your code here
theta2 = # Write your code here

'''
Step 4: Define a hypothesis function to predict Y
'''
hypothesis_function = # Write your code here

'''
Step 5: Use the square error as the loss function
'''
loss_function = # Write your code here
tf.summary.scalar('total cost', loss_function)

'''
Step 6: Using gradient descent with learning rate of 0.00001 to minimize loss
'''
optimizer = # Write your code here

with tf.Session() as session:
    '''
    Step 7: Initialize the necessary variables, i.e. theta0 and theta1
    '''
    session.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./graphs/polynomial_regression', session.graph)

    '''
    Step 8: Train the model for 100,000 epochs
    '''
    for i in range(100000):
        # Write your code here

        print("Epoch: {0}, cost = {1}, theta0 = {2}, theta1 = {3}, theta2 = {4}".
              format(i + 1, cost, session.run(theta0), session.run(theta1), session.run(theta1)))

    '''
    Step 9: Prints the training cost, theta0, and theta1
    '''
    print("Optimization Finished!")
    training_cost = session.run(loss_function, feed_dict={X: data.T[0], Y: data.T[1]})
    print("Training cost =", training_cost, "theta0 = ", session.run(theta0), "theta1 = ", session.run(theta1),
          "theta2 = ", session.run(theta2), '\n')

    '''
    Step 10: Plot the results
    '''
    # Graphic display
    plt.plot(data.T[0], data.T[1], 'ro', label='Original data')
    plt.plot(data.T[0], session.run(theta0) + (session.run(theta1) * data.T[0]) + (session.run(theta2) * data.T[0] * data.T[0]),
             'bo', label='Fitted line')
    plt.xlabel('fire per 1000 housing units')
    plt.ylabel('theft per 1000 population')
    plt.legend()
    plt.show()

# Close the writer when you finished using it
writer.close()
