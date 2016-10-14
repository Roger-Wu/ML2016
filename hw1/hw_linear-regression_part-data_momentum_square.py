
# coding: utf-8

# In[17]:

# get_ipython().magic('pylab inline')
import numpy as np
import pandas as pd


# In[18]:

DATA_FOLDER = "data/"
MODEL_FOLDER = "models/part-mmt-square/"
OUTPUT_FOLDER = "outputs/"
TRAIN_FILE_PATH = DATA_FOLDER + "train.csv"
TEST_FILE_PATH = DATA_FOLDER + "test_X.csv"


# In[19]:

# read train data
train_data = pd.read_csv(TRAIN_FILE_PATH, header=0, encoding="big5")


# In[20]:

# preprocess train data
train_data[train_data=="NR"] = 0


# In[21]:

train_data


# In[22]:

# concat 20 days data
train_array_months = []  # 12 numpy.array, array's shape is (18, 480)
for month in range(12):
    trainDataArr = np.concatenate( [train_data[(360*month+18*day):(360*month+18*(day+1))][list(range(3, 27))].apply(pd.to_numeric).as_matrix() for day in range(20)], axis=1)
    train_array_months.append(trainDataArr)


# ## Train

# In[23]:

MODEL_FOLDER = "models/part-mmt-square/"


# In[24]:

# use part of data
x_row_idxs = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13]  # > 0.2
x_row_num = len(x_row_idxs) + 1
x_col_num = 9

# train_array_months_part = [train_array_months[month][x_row_idxs, :] for month in range(12)]
train_array_months_part = [ np.concatenate([ train_array_months[month][x_row_idxs, :], np.square(train_array_months[month][9:10, :])/20 ], axis=0 ) for month in range(12)]


# In[25]:

train_array_months_part[0][:,:3]


# In[26]:

w = np.random.randn(x_row_num, x_col_num)
b = np.random.randn()


# In[27]:

prev_loss = np.inf


# In[28]:

# momentum
gamma = 0.9
vb = 0
vw = np.zeros((x_row_num, x_col_num))


# In[29]:

# adadelta
# b_grad_square_sum = 0
# w_grad_square_sum = np.zeros((x_row_num, x_col_num))
# gamma = 0.9
# eps = 10**(-8)


# In[30]:

total_epoch = 0


# In[31]:

lr = 3 * 10**(-9)


# In[32]:

# # load model
# model = np.load(MODEL_FOLDER + 'epo1903136_los5.7155')
# w = model['w']
# b = model['b']
# vb = model['vb']
# vw = model['vw']
# lr = model['lr']
# prev_loss = model['prev_loss']
# total_epoch = model['total_epoch']


# In[33]:


# while True:
# # for save_cycle in range(10):
#     for epoch in range(100):
#         loss = 0
#         b_grad = 0
#         w_grad = np.zeros((x_row_num, x_col_num))
#         for month in range(12):
#             for hour in range(470):
#                 x = train_array_months_part[month][:, hour:hour+9]
#                 y_data = train_array_months[month][9, hour+9]

#                 y_pred= einsum('ij,ij', x, w) + b
#                 diff = y_data - y_pred
#                 loss += diff * diff

#                 mult = 2 * diff * (-1)
#                 b_grad += mult
#                 w_grad += mult * x  # 2 * diff * (-1) * x

#         loss = sqrt(loss/(12*470))

#         if loss > prev_loss:
#             lr *= 0.8

# #         b_grad_square_sum = gamma * b_grad_square_sum + (1-gamma) * np.square(b_grad)
# #         w_grad_square_sum = gamma * w_grad_square_sum + (1-gamma) * np.square(w_grad)
# #         b = b - lr * (b_grad / np.sqrt(b_grad_square_sum))
# #         w = w - lr * (w_grad / np.sqrt(w_grad_square_sum))

#         vb = lr * b_grad + gamma * vb
#         vw = lr * w_grad + gamma * vw
#         b = b - vb
#         w = w - vw

#         total_epoch += 1
#         prev_loss = loss

#     file_name = "epo" + str(total_epoch) + "_los" + str(loss)[:6]
#     print(file_name)
#     print('  ', lr, loss)
#     np.savez(MODEL_FOLDER + file_name, b=b, w=w, vb=vb, vw=vw, lr=lr, prev_loss=prev_loss, total_epoch=total_epoch)

# print('end')


# ## Test

# In[34]:

test_data = pd.read_csv(TEST_FILE_PATH, header=None, encoding="big5")
test_data[test_data=="NR"] = 0
test_array = test_data[:][list(range(2, 11))].apply(pd.to_numeric).as_matrix()
data_num = 240


# In[35]:

test_data.head(18)


# In[36]:

# process test data
# use part of data
x_row_idxs = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13]  # > 0.2
# x_row_num = len(x_row_idxs) + 1
# x_col_num = 9

x_test_arrays = []
for data_idx in range(data_num):
    row_start_idx = data_idx*18
    this_row_idxs = [row_start_idx+i for i in x_row_idxs]
    this_extr_rows = test_array[this_row_idxs, :]
    this_pm25_square = np.square(test_array[row_start_idx+9:row_start_idx+10, :])/20
    this_x = np.concatenate([this_extr_rows, this_pm25_square], axis=0)
    x_test_arrays.append(this_x)


# In[37]:

x_test_arrays[0].shape


# In[38]:

_id = pd.Series(["id_" + str(i) for i in range(data_num)])


# In[39]:

# # load model
model_file = 'epo1903136_los5.7155.npz'
model = np.load(MODEL_FOLDER + model_file)
w = model['w']
b = model['b']


# In[40]:

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

# write file
output.to_csv("kaggle_best.csv", index=False)


# In[ ]:
