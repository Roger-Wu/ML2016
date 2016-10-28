
# coding: utf-8

# In[214]:

import numpy as np
from scipy import special
import pandas as pd
import os
import sys


TRAIN_FILE_PATH = sys.argv[1]
MODEL_FILE_PATH = sys.argv[2]  # output model

EPOCH = 100000

train_data = pd.read_csv(TRAIN_FILE_PATH, header=None)

# In[218]:

x_data_rows = train_data[list(range(1, 58))].as_matrix()
y_data_rows = train_data[[58]].as_matrix()


# ## Select Data

# In[219]:

x_select_col_idxs = [26, 40, 31, 24,  3, 41, 54, 55, 45, 52, 28, 22, 53, 34, 43, 47, 6, 44, 15, 25, 16, 19, 38,  8, 56, 32,  4, 35, 27,  5, 48, 14]
x_data_rows = x_data_rows[:, x_select_col_idxs]


# ## Preprocess Data

# ## Transform Data

# In[220]:

# Normalize
x_train_means = np.mean(x_data_rows, axis=0)
x_train_stds =  np.std(x_data_rows, axis=0)
x_data_rows = (x_data_rows - x_train_means) / x_train_stds


# In[221]:

# print(np.mean(x_data_rows, axis=0), np.std(x_data_rows, axis=0))


# In[222]:

# append 1 in the last column, so that b in including in w
x_data_rows = np.c_[x_data_rows, np.ones(x_data_rows.shape[0]) ]


# In[223]:

# pre-transpose x for matrix multiplication
x_data_rows_trans = np.transpose(x_data_rows)


# In[224]:

x_row_num, x_col_num = x_data_rows.shape  # 4001, 58


# ## Train

# In[225]:

# reset parameters
w = np.random.randn(x_col_num, 1)  # shape = (x_col_num, 1). not (1, x_col_num) for the convenience of matrix multiplication
total_epoch = 0


# In[226]:

layers = [x_col_num, 30, 30, 30, 30, 1]  # node nums of layers, including input layer x and output layer y
# layer_1_node_num = 40
# layer_2_node_num = layer_1_node_num
# layer_3_node_num = layer_2_node_num


# In[227]:

ws = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers) - 1)]  # weights
ws.insert(0, None)  # make ws 1-based
# w_1 = np.random.randn(x_col_num, layer_1_node_num)  # dot(x, w_1) output a x_row_num * layer_1_node_num matrix
# w_2 = np.random.randn(layer_1_node_num, layer_2_node_num)
# w_3 = np.random.randn(layer_2_node_num, layer_3_node_num)
# w_out = np.random.randn(layer_3_node_num, 1)  # dot(layer_1_output, w_2) output a x_row_num * 1 matrix
# v_w_1 = np.random.randn(x_col_num, layer_1_node_num)  # dot(x, w_1) output a x_row_num * layer_1_node_num matrix
# v_w_2 = np.random.randn(layer_1_node_num, 1)  # dot(layer_1_output, w_2) output a x_row_num * 1 matrix


# In[228]:

dropout_rate = 0.5


# In[229]:

# momentum
gamma = 0.9
# vw = np.zeros((x_col_num, 1))


# In[230]:

lr = (10) / x_row_num


# In[231]:

def to_bool(x):
    return 1 if x > 0.5 else 0

to_bool = np.vectorize(to_bool)


# In[232]:

def sigmoid_deriv(x):  # input may be an matrix
    return x * (1-x)  # element-wise


# In[233]:

# a = np.ones((3, 10))
# m = np.random.binomial(1, 0.5, 10)
# a * m


# In[234]:

# training iterations

# special.expit is sigmoid

layer_outputs = [None] * len(layers)
layer_outputs[0] = x_data_rows
layer_errors = [None] * len(layers)
layer_deltas = [None] * len(layers)

# if you can't understand the following code, read nn_part_dropout_layer3.ipynb
for i in range(EPOCH):
    # forward
    for layerIdx in range(1, len(layers)):
        layer_outputs[layerIdx] = special.expit(np.dot(layer_outputs[layerIdx-1], ws[layerIdx]))
        # dropout
        if layerIdx < len(layers)-1:
            layer_outputs[layerIdx] *= np.random.binomial(1, dropout_rate, layers[layerIdx])

    # backpropagation
    for layerIdx in range(len(layers)-1, 0, -1):
        if layerIdx == len(layers)-1:
            layer_errors[layerIdx] = layer_outputs[layerIdx] - y_data_rows
        else:
            layer_errors[layerIdx] = layer_deltas[layerIdx+1].dot(ws[layerIdx+1].T)
        layer_deltas[layerIdx] = layer_errors[layerIdx] * sigmoid_deriv(layer_outputs[layerIdx])

    for layerIdx in range(1, len(layers)):
        ws[layerIdx] -= lr * (layer_outputs[layerIdx-1].T).dot(layer_deltas[layerIdx])

    # write model
    if (total_epoch < 10000 and total_epoch % 1000 == 0) or (total_epoch % 10000 == 0):
        for layerIdx in range(1, len(layers)):
            layer_outputs[layerIdx] = special.expit(np.dot(layer_outputs[layerIdx-1], ws[layerIdx]))
            # dropout
            if layerIdx < len(layers)-1:
                layer_outputs[layerIdx] *= dropout_rate

        train_error_count = np.sum(np.abs(to_bool(layer_outputs[-1]) - y_data_rows))
        train_accuracy = float(x_row_num - train_error_count) / x_row_num

        model_file_info = "epo" + str(total_epoch) + "_err" + str(train_error_count) + "_acc" + str(train_accuracy)[:6]
        print(model_file_info)

    total_epoch += 1

train_error_count = np.sum(np.abs(to_bool(layer_outputs[-1]) - y_data_rows))
train_accuracy = float(x_row_num - train_error_count) / x_row_num

model_file_info = "epo" + str(total_epoch) + "_err" + str(train_error_count) + "_acc" + str(train_accuracy)[:6]
print(model_file_info)

np.savez(MODEL_FILE_PATH, layers=layers, ws=ws, lr=lr, dropout_rate=dropout_rate, total_epoch=total_epoch, x_train_means=x_train_means, x_train_stds=x_train_stds)




#
# # ## Test
#
# # In[ ]:
#
# test_data = pd.read_csv(TEST_FILE_PATH, header=None)
#
#
# # In[ ]:
#
# _id = test_data[0].apply(str)
#
#
# # In[ ]:
#
# x_test_data_rows = test_data[list(range(1, 58))].as_matrix()
#
#
# # In[ ]:
#
# # x_test_data_rows = x_test_data_rows[:, x_select_col_idxs]
#
#
# # In[ ]:
#
# # # read model
# # model_file = 'epo900000_rms0.020180_err97.npz'
# # model = np.load(MODEL_FOLDER + model_file)
# # ws = model['ws']
# # x_train_means = model['x_train_means']
# # x_train_stds = model['x_train_stds']
# # model_file_info = model_file[:-4]
#
#
# # In[ ]:
#
# # normalize
# x_test_data_rows = (x_test_data_rows - x_train_means) / x_train_stds
# # add 1
# x_test_data_rows = np.c_[x_test_data_rows, np.ones(x_test_data_rows.shape[0]) ]
#
#
# # In[ ]:
#
# # test
# layer_outputs = [None] * len(layers)
# layer_outputs[0] = x_test_data_rows
#
# for layerIdx in range(1, len(layers)):
#     layer_outputs[layerIdx] = special.expit(np.dot(layer_outputs[layerIdx-1], ws[layerIdx]))
#     # dropout
#     if layerIdx < len(layers)-1:
#         layer_outputs[layerIdx] *= dropout_rate
# y_test_rows = to_bool(layer_outputs[-1])
#
# y_test_series = pd.Series(y_test_rows.flatten())
#
# # concat id and y
# output = pd.concat([_id, y_test_series], axis=1)
# output.columns=["id","label"]
#
# # write file
# output.to_csv(OUTPUT_FOLDER + "output_" + METHOD_NAME + "_" + model_file_info + ".csv", index=False)
#
#
# # In[ ]:
