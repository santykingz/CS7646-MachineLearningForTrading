import DTLearner as dt
import pandas as pd

df = pd.read_csv('assess_learners/Data/Istanbul.csv')
data = df.values
train_rows = int(0.6 * data.shape[0])
test_rows = data.shape[0] - train_rows

# separate out training and testing data
train_x = data[:train_rows, 0:-1]
train_y = data[:train_rows, -1]
test_x = data[train_rows:, 0:-1]
test_y = data[train_rows:, -1]

print(f"{test_x.shape}")
print(f"{test_y.shape}")

Xtrain = train_x
Ytrain = train_y
Xtest = test_x

learner = dt.DTLearner(leaf_size = 1, verbose = True) # constructor
learner.add_evidence(Xtrain, Ytrain) # training step
Ypred = learner.query(Xtest) # query
