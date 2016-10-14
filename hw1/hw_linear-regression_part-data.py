
# coding: utf-8

# In[415]:

# get_ipython().magic('pylab inline')
import numpy as np
import pandas as pd
import os


# In[416]:

DATA_FOLDER = "data/"
MODEL_FOLDER = "models/hw_linear-regression/"
OUTPUT_FOLDER = "outputs/"
TRAIN_FILE_PATH = DATA_FOLDER + "train.csv"
TEST_FILE_PATH = DATA_FOLDER + "test_X.csv"


# In[417]:

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)


# In[418]:

# read train data
train_data = pd.read_csv(TRAIN_FILE_PATH, header=0, encoding="big5")


# In[419]:

# preprocess train data
train_data[train_data=="NR"] = 0


# In[420]:

train_data.head(18)


# In[421]:

# concat 20 days data
train_array_months = []  # 12 numpy.array, array's shape is (18, 480)
for month in range(12):
    temp_train_array = np.concatenate( [train_data[(360*month+18*day):(360*month+18*(day+1))][list(range(3, 27))].apply(pd.to_numeric).as_matrix() for day in range(20)], axis=1)
    train_array_months.append(temp_train_array)


# In[422]:

# use part of data
x_row_idxs = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13]  # > 0.2
x_row_num = len(x_row_idxs)
x_col_num = 9

train_array_months_processed = [ta[x_row_idxs, :] for ta in train_array_months]


# ## Train

# In[423]:

w = np.random.randn(x_row_num, x_col_num)
b = np.random.randn()
total_epoch = 0


# In[424]:

prev_loss = np.inf


# In[425]:

# momentum
gamma = 0.9
vb = 0
vw = np.zeros((x_row_num, x_col_num))


# In[426]:

b = np.random.randn()
w = np.random.randn(x_row_num, x_col_num)


# In[427]:

lr = 4 * 10**(-9)


# In[428]:

# # load model
# model = np.load(MODEL_FOLDER + 'epo2100_los4.9199.npz')
# w = model['w']
# b = model['b']
# vb = model['vb']
# vw = model['vw']
# lr = model['lr']
# total_epoch = model['total_epoch']


# In[429]:

while True:
    # one training epoch

    # reset temp variables
    loss = 0
    b_grad = 0
    w_grad = np.zeros((x_row_num, x_col_num))

    # calculate loss and gradient over all training data
    for month in range(12):
        for hour in range(470):
            # extract x and y from training data
            # train_array_months is a list of 12 arrays
            # x will be an ?x9 numpy array. '?' depends on how we process training data
            x = train_array_months_processed[month][:, hour:hour+9]
            y_data = train_array_months[month][9, hour+9]

            # calculate y from x, w, b
            y_pred= np.einsum('ij,ij', x, w) + b

            # update loss
            diff = y_data - y_pred
            loss += diff * diff

            # update gradient of b and w
            mult = 2 * diff * (-1)
            b_grad += mult
            w_grad += mult * x  # 2 * diff * (-1) * x

    # calculate rms error
    loss = np.sqrt(loss/(12*470))

    # save model
    if total_epoch % 100 == 0:
        model_file_info = "epo" + str(total_epoch) + "_los" + str(loss)[:6]
        print(model_file_info)
        np.savez(MODEL_FOLDER + model_file_info, b=b, w=w, vb=vb, vw=vw, lr=lr, total_epoch=total_epoch)

        # stop training in some cases
        if loss < 6:
            break

    # update parameters
    vb = lr * b_grad + gamma * vb
    vw = lr * w_grad + gamma * vw
    b = b - vb
    w = w - vw
    total_epoch += 1

print('end of training')


# ## Test

# In[430]:

test_data = pd.read_csv(TEST_FILE_PATH, header=None, encoding="big5")
test_data[test_data=="NR"] = 0
test_array = test_data[:][list(range(2, 11))].apply(pd.to_numeric).as_matrix()
data_num = 240


# In[431]:

x_test_arrays = []
for data_idx in range(data_num):
    row_start_idx = data_idx*18
    this_row_idxs = [row_start_idx+i for i in x_row_idxs]
    this_extr_rows = test_array[this_row_idxs, :]
    x_test_arrays.append(this_extr_rows)


# In[432]:

_id = pd.Series(["id_" + str(i) for i in range(data_num)])


# In[433]:

# # load model
# model_file = 'epo1532682_los5.7251.npz'
# model = np.load(MODEL_FOLDER + model_file)
# w = model['w']
# b = model['b']
# model_file_info = model_file[:-4]


# In[434]:

# calc y
y_preds = []
for dataIdx in range(data_num):
    x = x_test_arrays[dataIdx]
    y_pred = np.einsum('ij,ij', x, w) + b
    y_preds.append(y_pred)
y_pred_series = pd.Series(y_preds)

# concat id and y
output = pd.concat([_id, y_pred_series], axis=1)
output.columns=["id","value"]

# set as 0 if value < 0
output.ix[output["value"] < 0, ["value"]] = 0

# write file
# output.to_csv(OUTPUT_FOLDER + "output_hw_linear-regression_" + model_file_info + ".csv", index=False)
output.to_csv("linear_regression.csv", index=False)


# In[ ]:
