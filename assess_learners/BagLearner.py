""""""
"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch

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
"""

import numpy as np


class BagLearner(object):
    """
    This is a Linear Regression Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self,learner,bags,kwargs={},boost=False, verbose=False, bag_data_percent=0.6):
        """
        Constructor method
        """
        self.learner = learner
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.kwargs = kwargs
        self.bag_data_percent = bag_data_percent
        self.learners = []

        pass  # move along, these aren't the drones you're looking for


    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "sspickard3"  # replace tb34 with your Georgia Tech username

    def study_group(self):
        """
        : return A comma separated string of GT_Name of each member of your study group

        :rtype: str
        """
        return "gburdell3"

    def pick_data_with_replacement(self, data_x, data_y, n):
        """
        Return n data points of x and y picked at random and with replacement.
        """
        rows_to_keep = list(np.random.choice(
            range(0, data_x.shape[0]),
            size=n,
            replace=True
        ))
        return data_x[rows_to_keep,:], data_y[rows_to_keep]

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        learners = []

        # get the number of records for each bag
        number_of_records_to_pick = int(self.bag_data_percent * data_x.shape[0])
        # loop through the number of bags we want to do this for
        for _ in range(0, self.bags):
            temp_data_x, temp_data_y = self.pick_data_with_replacement(
                data_x, data_y, number_of_records_to_pick)
            learner = self.learner(**self.kwargs)
            learner.add_evidence(temp_data_x, temp_data_y)
            learners.append(learner)
        self.learners = learners

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        results = []
        for bag in range(0, self.bags):
            learner = self.learners[bag]
            bag_result = learner.query(points)
            results.append(bag_result)
        return np.mean(results, axis=0)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
