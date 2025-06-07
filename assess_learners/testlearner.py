""""""
"""
Test a learner.  (c) 2015 Tucker Balch

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

import math
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rl
import BagLearner as bl
import InsaneLearner as it


def test_all(train_x, train_y, test_x, test_y):
    #===========================================================================
    # LinRegLearner
    #===========================================================================
    # create a learner and train it
    print(50*'-')
    print('Linear Regression Learner')
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")
    print('')
    #===========================================================================
    # Decision Tree Learner
    #===========================================================================
    # create a learner and train it
    print(50*'-')
    print('Decision Tree Learner')
    learner = dt.DTLearner(leaf_size=3, verbose=False)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")
    print('')
    # #===========================================================================
    # # Random Tree Learner
    # #===========================================================================
    # create a learner and train it
    print(50*'-')
    print('Random Tree Learner')
    learner = rl.RTLearner(leaf_size=3, verbose=False)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")
    #===========================================================================
    # Bag Learner
    #===========================================================================
    # create a learner and train it
    print(50*'-')
    print('Bag Learner - Decision Tree')
    # test 1
    learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
    learner.add_evidence(train_x, train_y)
    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")
    #---------------------------------------------------------------------------
    print(50*'-')
    print('Bag Learner - Linear Regression')
    # test 2
    learner = bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 10, boost = False, verbose = False)
    learner.add_evidence(train_x, train_y)
    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")
    #---------------------------------------------------------------------------
    # test 3
    print(50*'-')
    print('Bag Learner -Random tree')
    learner = bl.BagLearner(learner = rl.RTLearner, kwargs = {"leaf_size":3}, bags = 20, boost = False, verbose = False)
    learner.add_evidence(train_x, train_y)
    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")
    #===========================================================================
    # Insane Learner
    #===========================================================================
    # create a learner and train it
    print(50*'-')
    print('Insane Learner')
    learner = it.InsaneLearner(verbose = False) # constructor
    learner.add_evidence(train_x, train_y) # training step
    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")
def expiriment_one(train_x, train_y, test_x, test_y):
    results = pd.DataFrame(columns=[
        'leaf_size',
        'in_sample_corr','in_sample_rmse',
        'out_of_sample_corr','out_of_sample_rmse'])
    for leaf in range(0,20):
        learner = dt.DTLearner(leaf_size=leaf, verbose=False)  # create a LinRegLearner
        learner.add_evidence(train_x, train_y)  # train it

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        in_rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_c = np.corrcoef(pred_y, y=train_y)[0,1]

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        out_rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_c = np.corrcoef(pred_y, y=test_y)[0,1]
        results = pd.concat([
            results,
            pd.DataFrame(
                data=[[leaf,in_c,in_rmse,out_c,out_rmse]],
                columns=[
                    'leaf_size',
                    'in_sample_corr','in_sample_rmse',
                    'out_of_sample_corr','out_of_sample_rmse'])
        ])
    # make the graphs
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(results['leaf_size'], results['in_sample_corr'], label='In Sample')
    axes[0].plot(results['leaf_size'], results['out_of_sample_corr'], label = 'Out of Sample')
    axes[0].set_title('Correlation by Leaf Size')
    axes[0].set_xlabel('Max # of Leaves')
    axes[0].set_ylabel('Correlation')
    axes[0].legend()
    # plt.savefig('figure1.png')
    # plt.close()

    # rmse graphs
    axes[1].plot(results['leaf_size'], results['in_sample_rmse'], label='In Sample')
    axes[1].plot(results['leaf_size'], results['out_of_sample_rmse'], label = 'Out of Sample')
    axes[1].set_title('RMSE by Leaf Size')
    axes[1].set_xlabel('Max # of Leaves')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()

    # draw vertical lines where the leaf size becomes decent
    axes[0].axvline(x=6, color='gray', linestyle='--', linewidth=1)
    axes[1].axvline(x=6, color='gray', linestyle='--', linewidth=1)


    plt.suptitle('Overfitting by Leaf Size')
    plt.tight_layout()
    plt.savefig('figure1.png')
    plt.close()


def expiriment_two(train_x, train_y, test_x, test_y):
    results = pd.DataFrame(columns=[
        'leaf_size',
        'in_sample_corr','in_sample_rmse',
        'out_of_sample_corr','out_of_sample_rmse'])
    for leaf in range(0,20):
        learner = bl.BagLearner(
            learner = dt.DTLearner,
            kwargs = {"leaf_size":leaf},
            bags = 10,
            boost = False,
            verbose = False)
        learner.add_evidence(train_x, train_y)  # train it

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        in_rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_c = np.corrcoef(pred_y, y=train_y)[0,1]

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        out_rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_c = np.corrcoef(pred_y, y=test_y)[0,1]
        results = pd.concat([
            results,
            pd.DataFrame(
                data=[[leaf,in_c,in_rmse,out_c,out_rmse]],
                columns=[
                    'leaf_size',
                    'in_sample_corr','in_sample_rmse',
                    'out_of_sample_corr','out_of_sample_rmse'])
        ])
    # make the graphs
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(results['leaf_size'], results['in_sample_corr'], label='In Sample')
    axes[0].plot(results['leaf_size'], results['out_of_sample_corr'], label = 'Out of Sample')
    axes[0].set_title('Correlation by Leaf Size')
    axes[0].set_xlabel('Max # of Leaves')
    axes[0].set_ylabel('Correlation')
    axes[0].legend()
    # plt.savefig('figure1.png')
    # plt.close()

    # rmse graphs
    axes[1].plot(results['leaf_size'], results['in_sample_rmse'], label='In Sample')
    axes[1].plot(results['leaf_size'], results['out_of_sample_rmse'], label = 'Out of Sample')
    axes[1].set_title('RMSE by Leaf Size')
    axes[1].set_xlabel('Max # of Leaves')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()

    # draw vertical lines where the leaf size becomes decent
    axes[0].axvline(x=6, color='gray', linestyle='--', linewidth=1)
    axes[1].axvline(x=6, color='gray', linestyle='--', linewidth=1)


    plt.suptitle('Bagging Imapact on Overfitting')
    plt.tight_layout()
    plt.savefig('figure2.png')
    plt.close()

def expiriment_three(train_x, train_y, test_x, test_y):
    results = pd.DataFrame(columns=[
        'leaf_size',
        'in_sample_corr_dt','in_sample_rmse_dt',
        'out_of_sample_corr_dt','out_of_sample_rmse_dt',
        'in_sample_corr_rt','in_sample_rmse_rt',
        'out_of_sample_corr_rt','out_of_sample_rmse_rt'])
    for leaf in range(0,20):
        learner_dt = dt.DTLearner(leaf_size=leaf, verbose=False)
        learner_dt.add_evidence(train_x, train_y)  # train it

        learner_rt = rl.RTLearner(leaf_size=leaf, verbose=False)
        learner_rt.add_evidence(train_x, train_y)

        # evaluate in sample
        pred_y_dt = learner_dt.query(train_x)  # get the predictions
        in_rmse_dt = math.sqrt(((train_y - pred_y_dt) ** 2).sum() / train_y.shape[0])
        in_c_dt = np.corrcoef(pred_y_dt, y=train_y)[0,1]

        pred_y_rt = learner_rt.query(train_x)  # get the predictions
        in_rmse_rt = math.sqrt(((train_y - pred_y_rt) ** 2).sum() / train_y.shape[0])
        in_c_rt = np.corrcoef(pred_y_rt, y=train_y)[0,1]

        # evaluate out of sample
        pred_y_dt = learner_dt.query(test_x)  # get the predictions
        out_rmse_dt = math.sqrt(((test_y - pred_y_dt) ** 2).sum() / test_y.shape[0])
        out_c_dt = np.corrcoef(pred_y_dt, y=test_y)[0,1]

        pred_y_rt = learner_rt.query(test_x)  # get the predictions
        out_rmse_rt = math.sqrt(((test_y - pred_y_rt) ** 2).sum() / test_y.shape[0])
        out_c_rt = np.corrcoef(pred_y_rt, y=test_y)[0,1]

        # add the results to the df
        results = pd.concat([
            results,
            pd.DataFrame(
                data=[[leaf,
                        in_c_dt, in_rmse_dt, out_c_dt, out_rmse_dt,
                        in_c_rt, in_rmse_rt, out_c_rt, out_rmse_rt]],
                columns=[
                    'leaf_size',
                    'in_sample_corr_dt','in_sample_rmse_dt',
                    'out_of_sample_corr_dt','out_of_sample_rmse_dt',
                    'in_sample_corr_rt','in_sample_rmse_rt',
                    'out_of_sample_corr_rt','out_of_sample_rmse_rt'])
        ])
    # make the graphs
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(results['leaf_size'],
                    results['in_sample_corr_dt'],
                    color='blue',
                    label='DT: In Sample')
    axes[0].plot(results['leaf_size'],
                    results['out_of_sample_corr_dt'],
                    linestyle='--',
                    color='blue',
                    label = 'DT: Out of Sample')

    axes[0].plot(results['leaf_size'],
                    results['in_sample_corr_rt'],
                    color='green',
                    label='RT: In Sample')
    axes[0].plot(results['leaf_size'],
                    results['out_of_sample_corr_rt'],
                    linestyle='--',
                    color='green',
                    label = 'RT: Out of Sample')

    axes[0].set_title('Correlation by Leaf Size')
    axes[0].set_xlabel('Max # of Leaves')
    axes[0].set_ylabel('Correlation')
    axes[0].legend()
    # plt.savefig('figure1.png')
    # plt.close()

    # rmse graphs
    axes[1].plot(results['leaf_size'],
                    results['in_sample_rmse_dt'],
                    color='blue',
                    label='DT: In Sample')
    axes[1].plot(results['leaf_size'],
                    results['out_of_sample_rmse_dt'],
                    linestyle='--',
                    color='blue',
                    label = 'DT: Out of Sample')

    axes[1].plot(results['leaf_size'],
                    results['in_sample_rmse_rt'],
                    color='green',
                    label='RT: In Sample')
    axes[1].plot(results['leaf_size'],
                    results['out_of_sample_rmse_rt'],
                    color='green',
                    linestyle='--',
                    label = 'RT: Out of Sample')

    axes[1].set_title('RMSE by Leaf Size')
    axes[1].set_xlabel('Max # of Leaves')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()


    plt.suptitle('Decision Tree v Random Tree')
    plt.tight_layout()
    plt.savefig('figure3.png')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    lines = inf.readlines()

    # check if there is a header
    start_row = 0
    try:
        first_val = lines[start_row].strip().split(',')[0]
        float(first_val)
    except:
        start_row = 1

    # split the data into arrays
    data = []
    for line in lines[start_row:]:
        row = line.strip().split(',')
        data.append(row)
    data = np.array(data)

    # check through each column and see if the value is a float
    float_cols = []
    for c in range(data.shape[1]):
        try:
            data[:, c].astype(float)
            float_cols.append(c)
        except:
            continue
    data = data[:, float_cols]
    data = np.array(data, dtype=float)

    # compute how much of the data is training and testing
    train_row_count = int(0.6 * data.shape[0])
    train_rows = random.sample(range(0, data.shape[0]), train_row_count)
    test_rows =[x for x in range(0, data.shape[0]) if x not in train_rows]

    # separate out training and testing data
    train_x = data[train_rows, 0:-1]
    train_y = data[train_rows, -1]
    test_x = data[test_rows, 0:-1]
    test_y = data[test_rows, -1]



    test_all(train_x=train_x, train_y=train_y,test_x=test_x,test_y=test_y)
    expiriment_one(train_x=train_x, train_y=train_y,test_x=test_x,test_y=test_y)
    expiriment_two(train_x=train_x, train_y=train_y,test_x=test_x,test_y=test_y)
    expiriment_three(train_x=train_x, train_y=train_y,test_x=test_x,test_y=test_y)
    print('done')

