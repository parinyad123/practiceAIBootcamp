import numpy as np
import matplotlib.pyplot as plt
import xlrd


def gradient_descent(X, Y, alpha=0.01 , converge_criteria=0.0001, max_iteration=10000):
    converged = False
    iteration = 0

    # Number of datasets
    m = X.shape[0]

    # Initial thetas to 0
    theta0 = 0
    theta1 = 0

    # Total error, J(theta)
    J = # Write your code here

    while not converged:
        # For each training sample, compute the gradient
        gradient0 = # Write your code here 
        gradient1 = # Write your code here

        # Update the temporary thetas
        tmp0 = # Write your code here
        tmp1 = # Write your code here

        # Update thetas
        theta0 = # Write your code here
        theta1 = # Write your code here

        # Mean squared error
        error = # Write your code here

        print("Iteration = {0}, Cost = {1}".format(iteration, error.item(0)))

        if abs(J - error) <= converge_criteria:
            print('Converged, iterations: ', iteration, '!!!')
            converged = True

        J = error  # Update error
        iteration += 1  # Update iteration

        if iteration == max_iteration:
            print('Max interactions exceeded!')
            converged = True

    return theta0.item(0), theta1.item(0)


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
Step 2: Compute the gradients
'''
X, Y = # Write your code here

theta0, theta1 = # Call gradient_descent method with alpha = 0.001, converge_criteria=0.000001, and max_iteration=30000

print("theta0 = {0}, theta1 = {1}".format(theta0, theta1))

'''
Step 3: Plot the results
'''

# Graphic display
plt.plot(data.T[0], data.T[1], 'ro', label='Original data')
plt.plot(data.T[0], theta0 + theta1 * data.T[0], 'b', label='Fitted line')
plt.xlabel('fire per 1000 housing units')
plt.ylabel('theft per 1000 population')
plt.legend()
plt.show()

