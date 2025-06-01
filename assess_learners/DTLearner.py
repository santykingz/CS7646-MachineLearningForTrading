"""

"""

import numpy as np


class DTLearner(object):
    """
    This is a Decision Tree Learner (DTLearner).
    You will need to properly implement this class as necessary.

    Parameters
        leaf_size (int): Is the maximum number of samples to be aggregated at
            a leaf
        verbose (bool): If “verbose” is True, your code can print out
            information for debugging.
            If verbose = False your code should not generate ANY output.
            When we test your code, verbose will be False.
    """
    def __init__(self, leaf_size, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.tree = np.array([])
        self.verbose = verbose
        self.node_counter = 0
        self.flat_tree = []
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

    def best_split(self, data_x, data_y):
        """
        Use correlation to pick the best feature
        """
        data_y = np.asarray(data_y, dtype=float)
        r2 = []
        for c in range(data_x.shape[1]):
            temp_data_x = np.asarray(data_x[:, c], dtype=float)
            temp_r2 = np.corrcoef(temp_data_x, data_y)[0,1]
            if np.isnan(temp_r2):
                temp_r2 = 0
            r2.append(temp_r2)

        best_split_feature = np.argmax(np.array(r2))
        best_split_val = np.median(data_x[:,best_split_feature])

        return best_split_feature, best_split_val

    def only_floats(self, data_x):
        float_cols = []
        for c in range(data_x.shape[1]):
            try:
                data_x[:, c].astype(float)
                float_cols.append(c)
            except:
                continue
        return data_x[:, float_cols]

    def add_evidence(self, data_x, data_y, depth=0):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # clean out any non-float values
        data_x = self.only_floats(data_x)

        # print out the shape if we are verbose
        if self.verbose:
            print("data_x shape: ", data_x.shape)

        # if the the data in y is all the same
        # if the number of leaves in the tree has been reached
        if np.all(data_y == data_y[0]) or data_y.shape[0] <= self.leaf_size:
            node_id = self.node_counter
            self.node_counter = self.node_counter + 1
            self.flat_tree.append([node_id, "Leaf", data_y.mean(), np.nan, np.nan])
            # return [[-1, data_y.mean(), np.nan, np.nan, node_id]]
            return node_id

        # find the best split
        best_split_node, best_split_val = self.best_split(
            data_x=data_x,
            data_y=data_y
        )
        if self.verbose:
            print('Best split column:', best_split_node)
            print('Best split value:', best_split_val)

        # exit if the y didn't change
        left_y = data_y[data_x[:,best_split_node]<= best_split_val]
        right_y = data_y[data_x[:,best_split_node]> best_split_val]
        if left_y.shape==data_y.shape:
            node_id = self.node_counter
            self.node_counter = self.node_counter + 1
            self.flat_tree.append([node_id, "Leaf", data_y.mean(), np.nan, np.nan])
            # return [[-1, data_y.mean(), np.nan, np.nan]]
            return node_id

        # call the left and right trees
        # left_tree = self.add_evidence(
        left_id = self.add_evidence(
            data_x = data_x[data_x[:,best_split_node]<= best_split_val],
            data_y = left_y,
            depth=depth+1
        )

        # right_tree = self.add_evidence(
        right_id = self.add_evidence(
            data_x = data_x[data_x[:,best_split_node]> best_split_val],
            data_y = right_y,
            depth=depth+1
        )

        # set the tree root so we can model querying it
        if self.verbose:
            print('depth:', depth)

        # generate the node id for this loop
        node_id = self.node_counter
        self.node_counter = self.node_counter + 1
        self.flat_tree.append([node_id, f"X{best_split_node}", best_split_val, left_id, right_id])

        if depth==0:
            self.tree = np.array(sorted(self.flat_tree, key=lambda x: x[0]), dtype=object)

        return node_id

    def query_point(self, point):
        """
        Query and return the tree result for a single point
        """
        tree = self.tree.copy()
        root_id = list(set(tree[:,0]) - set(tree[:,3]) - set(tree[:,4]))[0]
        should_continue = True
        counter = 0
        next_node_id = root_id
        while should_continue and counter < 100:
            counter = counter + 1
            loop_row = tree[tree[:,0].astype(int)==next_node_id][0]
            feature = loop_row[1]
            val = loop_row[2]
            # check if the feature is the leaf and then we just return the val
            if feature == 'Leaf':
                return val
            else:
                feature_col = int(feature[1:])
                # check if it should be left
                if point[feature_col] <= val:
                    next_node_id = loop_row[3]
                else:
                    next_node_id = loop_row[4]
        return np.nan


    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        res = []
        for p in points:
            res.append(self.query_point(p))
        return np.array(res)



if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
