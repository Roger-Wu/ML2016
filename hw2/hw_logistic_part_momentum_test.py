
# coding: utf-8

# In[17]:

import numpy as np
from scipy import special
import pandas as pd
import os
import sys


# In[18]:

METHOD_NAME = "hw_logistic"

MODEL_FILE_PATH = sys.argv[1]
TEST_FILE_PATH = sys.argv[2]
OUTPUT_FILE_PATH = sys.argv[3] # output model

if not MODEL_FILE_PATH.endswith(".npz"):
    MODEL_FILE_PATH = MODEL_FILE_PATH + ".npz"


x_select_col_idxs = [26, 40, 31, 24,  3, 41, 54, 55, 45, 52, 28, 22, 53, 34, 43, 47, 6, 44, 15, 25, 16, 19, 38,  8, 56, 32,  4, 35, 27,  5, 48, 14]

def to_bool(x):
    return 1 if x > 0.5 else 0

to_bool = np.vectorize(to_bool)



# ## Test

test_data = pd.read_csv(TEST_FILE_PATH, header=None)

_id = test_data[0].apply(str)

x_test_data_rows = test_data[list(range(1, 58))].as_matrix()


# In[ ]:

# x_select_col_idxs = [26, 40, 31, 24,  3, 41, 54, 55, 45, 52, 28, 22, 53, 34]
x_test_data_rows = x_test_data_rows[:, x_select_col_idxs]


# read model
model = np.load(MODEL_FILE_PATH)
w = model['w']
x_train_means = model['x_train_means']
x_train_stds = model['x_train_stds']

# In[ ]:

# normalize
x_test_data_rows = (x_test_data_rows - x_train_means) / x_train_stds
# add 1
x_test_data_rows = np.c_[x_test_data_rows, np.ones(x_test_data_rows.shape[0]) ]


# In[ ]:

# test
y_test_rows = to_bool(special.expit(np.dot(x_test_data_rows, w)))

y_test_series = pd.Series(y_test_rows.flatten())

# concat id and y
output = pd.concat([_id, y_test_series], axis=1)
output.columns=["id","label"]

# write file
output.to_csv(OUTPUT_FILE_PATH, index=False)
