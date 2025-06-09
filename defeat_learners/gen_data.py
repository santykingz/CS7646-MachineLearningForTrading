""""""
"""
template for generating data to fool learners (c) 2016 Tucker Balch
Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""

import math

import numpy as np


# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best_4_lin_reg(seed=1489683273):
    """
    Returns data that performs significantly better with LinRegLearner than DTLearner.
    The data set should include from 2 to 10 columns in X, and one column in Y.
    The data should contain from 10 (minimum) to 1000 (maximum) rows.

    :param seed: The random seed for your data generation.
    :type seed: int
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.
    :rtype: numpy.ndarray
    """
    np.random.seed(seed)
    # pick a random number of columns to have between 2 and 10
    x_col_count = np.random.randint(low=2, high=10, size=1)[0]

    # pick a random number of rows
    x_row_count = np.random.randint(low=500, high=1000, size=1)[0]

    # loop through the x_col_count and generate some random data
    x = np.zeros((x_row_count, x_col_count))
    for loop in range(0, x_col_count):
        x[:, loop] = np.random.random(size = x_row_count)

    # calculate y
    betas = np.random.random(size = x_col_count + 1)
    y_without_noise = betas[0] + np.dot(x, betas[1:])

    # add some random noise to y
    random_noise = np.random.normal(loc=0.0, scale=0.05, size=x_row_count)
    y = y_without_noise + random_noise

    return x, y


def best_4_dt(seed=1489683273):
    """
    Returns data that performs significantly better with DTLearner than LinRegLearner.
    The data set should include from 2 to 10 columns in X, and one column in Y.
    The data should contain from 10 (minimum) to 1000 (maximum) rows.

    :param seed: The random seed for your data generation.
    :type seed: int
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.
    :rtype: numpy.ndarray
    """
    np.random.seed(seed)
    # pick a random number of columns to have between 2 and 10
    x_col_count = np.random.randint(low=2, high=10, size=1)[0]

    # pick a random number of rows
    x_row_count = np.random.randint(low=500, high=1000, size=1)[0]

    betas = np.random.random(size = x_col_count + 1)
    powers = np.random.randint(low = 1, high=5, size = x_col_count)
    y_without_noise = betas[0] + np.ones(x_row_count)

    # loop through the x_col_count and generate some random data
    x = np.zeros((x_row_count, x_col_count))
    for loop in range(0, x_col_count):
        temp_data = np.random.random(size = x_row_count)
        x[:, loop] = temp_data
        y_without_noise = y_without_noise + temp_data ** powers[loop]
        y_without_noise = y_without_noise + (temp_data > 0.5).astype(float) * np.random.uniform(1, 3)

    # add some random noise to y
    random_noise = np.random.normal(loc=0.0, scale=0.05, size=x_row_count)


    y = y_without_noise + random_noise

    return x, y

def study_group(self):
    """
    : return A comma separated string of GT_Name of each member of your study group

    :rtype: str
    """
    return "gburdell3"

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "sspickard3"  # Change this to your user ID


if __name__ == "__main__":
    print("they call me Tim.")
    best_4_lin_reg()
    best_4_dt()
