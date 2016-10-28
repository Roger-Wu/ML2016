
# coding: utf-8

# In[17]:

import numpy as np
from scipy import special
import pandas as pd
import os
import sys


# In[18]:

METHOD_NAME = "hw_logistic"

TRAIN_FILE_PATH = sys.argv[1]
MODEL_FILE_PATH = sys.argv[2] # output model


# ## Read Training Data

# In[19]:

train_data = pd.read_csv(TRAIN_FILE_PATH, header=None)


# In[20]:

x_data_rows = train_data[list(range(1, 58))].as_matrix()
y_data_rows = train_data[[58]].as_matrix()


# ## Select Data

# In[21]:

x_select_col_idxs = [26, 40, 31, 24,  3, 41, 54, 55, 45, 52, 28, 22, 53, 34, 43, 47, 6, 44, 15, 25, 16, 19, 38,  8, 56, 32,  4, 35, 27,  5, 48, 14]
x_data_rows = x_data_rows[:, x_select_col_idxs]


# In[22]:

x_data_rows.shape


# ## Preprocess Data

# ## Transform Data

# In[23]:

# Normalize
x_train_means = np.mean(x_data_rows, axis=0)
x_train_stds =  np.std(x_data_rows, axis=0)
x_data_rows = (x_data_rows - x_train_means) / x_train_stds


# In[24]:

# print(np.mean(x_data_rows, axis=0), np.std(x_data_rows, axis=0))


# In[25]:

# add 1 in the last column, so that b in including in w
x_data_rows = np.c_[x_data_rows, np.ones(x_data_rows.shape[0]) ]


# In[26]:

# pre-transpose x for matrix multiplication
x_data_rows_trans = np.transpose(x_data_rows)


# In[27]:

x_row_num, x_col_num = x_data_rows.shape  # 4001, 58


# ## Train

# In[28]:

# reset parameters
w = np.random.randn(x_col_num, 1)  # shape = (x_col_num, 1). not (1, x_col_num) for the convenience of matrix multiplication
total_epoch = 0


# In[29]:

# momentum
gamma = 0.9
vw = np.zeros((x_col_num, 1))


# In[30]:

lr = (10) / x_row_num


# In[31]:

def to_bool(x):
    return 1 if x > 0.5 else 0

to_bool = np.vectorize(to_bool)


# In[32]:

# training iterations

for i in range(200000):
    # forward
    y_pred_rows = special.expit(np.dot(x_data_rows, w))  # shape = (x_row_num, 1)
    y_diff_rows = (y_data_rows - y_pred_rows)  # shape = (x_row_num, 1)
    w_grad = -np.dot(x_data_rows_trans, y_diff_rows)

    if total_epoch % 10000 == 0:
        train_error_count = np.sum(np.abs(to_bool(y_pred_rows) - y_data_rows))
        train_accuracy = float(x_row_num - train_error_count) / x_row_num

        model_file_info = "epo" + str(total_epoch) + "_acc" + str(train_accuracy)[:6]

        print(model_file_info)


    # update weights. using momentum
    vw = lr * w_grad + gamma * vw
    w = w - vw

    total_epoch += 1

# write model
train_error_count = np.sum(np.abs(to_bool(y_pred_rows) - y_data_rows))
train_accuracy = float(x_row_num - train_error_count) / x_row_num
model_file_info = "epo" + str(total_epoch) + "_acc" + str(train_accuracy)[:6]
print(model_file_info)
np.savez(MODEL_FILE_PATH, w=w, vw=vw, lr=lr, total_epoch=total_epoch, x_train_means=x_train_means, x_train_stds=x_train_stds)
