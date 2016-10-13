
# coding: utf-8

# In[24]:

get_ipython().magic('pylab inline')
import numpy as np
import pandas as pd


# In[25]:

DATA_FOLDER = "data/"
MODEL_FOLDER = "models/all-momentum/"
OUTPUT_FOLDER = "outputs/"
TRAIN_FILE_PATH = DATA_FOLDER + "train.csv"
TEST_FILE_PATH = DATA_FOLDER + "test_X.csv"


# In[26]:

# read train data
train_data = pd.read_csv(TRAIN_FILE_PATH, header=0, encoding="big5")


# In[27]:

# preprocess train data
train_data[train_data=="NR"] = 0


# In[28]:

train_data.head(18)


# In[29]:

# concat 20 days data
train_array_months = []  # 12 numpy.array, array's shape is (18, 480)
for month in range(12):
    train_array_one_month = np.concatenate( [train_data[(360*month+18*day):(360*month+18*(day+1))][list(range(3, 27))].apply(pd.to_numeric).as_matrix() for day in range(20)], axis=1)
    train_array_months.append(train_array_one_month)


# ## Train

# In[30]:

w = np.random.randn(18, 9)
b = np.random.randn()


# In[31]:

prevLoss = np.inf


# In[32]:

# momentum
gamma = 0.9
vb = 0
vw = np.zeros((18, 9))


# In[33]:

gamma = 0.9


# In[34]:

lr = 4 * 10**(-10)


# In[35]:

# load model
model_file = "los5.688764.npz"
model = np.load(MODEL_FOLDER + model_file)
w = model['w']
b = model['b']
vb = model['vb']
vw = model['vw']
lr = model['lr']
prevLoss = model['prevLoss']


# In[15]:

while True:
    losses = []
    for epoch in range(1000):
        loss = 0
        deltaB = 0
        deltaW = np.zeros((18, 9))
        for month in range(12):
            for hour in range(470):
                x = train_array_months[month][:, hour:hour+9]
                realY = train_array_months[month][9, hour+9]

                predY = einsum('ij,ij', x, w) + b
                diff = realY - predY
                loss += diff * diff

                mult = 2 * diff * (-1)
                deltaB += mult
                deltaW += mult * x  # 2 * diff * (-1) * x

        loss = sqrt(loss/(12*470))
        losses.append(loss)

        if loss > prevLoss:
            lr *= 0.75
        
        vb = gamma * vb + lr * deltaB
        vw = gamma * vw + lr * deltaW
        b = b - vb
        w = w - vw
        
        prevLoss = loss
    
    file_name = "los" + str(loss)[:8]
    print(file_name)
    print(lr, loss)
    
    np.savez(MODEL_FOLDER + "los" + str(loss)[:8], w=w, b=b, vw=vw, vb=vb, lr=lr, prevLoss=prevLoss)

print('end')


# ## Test

# In[17]:

# load test data
test_data = pd.read_csv(TEST_FILE_PATH, header=None, encoding="big5")
test_data[test_data=="NR"] = 0
test_array = test_data[:][list(range(2, 11))].apply(pd.to_numeric).as_matrix()
data_num = 240


# In[14]:

test_data.head(18)


# In[21]:

# load model
model_file = "los5.688764.npz"
model = np.load(MODEL_FOLDER + model_file)
w = model['w']
b = model['b']


# In[22]:

_id = pd.Series(["id_" + str(i) for i in range(data_num)])

# calc y
y_preds = []

for dataIdx in range(data_num):
    rowIdx = dataIdx * 18
    x = test_array[rowIdx:rowIdx+18, :]
    y_pred = einsum('ij,ij', x, w) + b
    y_preds.append(y_pred)
y_pred_series = pd.Series(y_preds)

# concat id and y
output = pd.concat([_id, y_pred_series], axis=1)
output.columns=["id","value"]

# write file
output.to_csv(OUTPUT_FOLDER + "output_linear_all_" + model_file[:-4] + ".csv", index=False)


# In[ ]:




# In[ ]:

test_data = pd.read_csv(TEST_FILE_PATH, header=None, encoding="big5")
# preprocess
test_data[test_data=="NR"] = 0
# to numpy array
test_array = test_data[:][list(range(2, 11))].apply(pd.to_numeric).as_matrix()
test_array.shape


# In[ ]:

dataNum = 240


# In[ ]:

predYs = []
for dataIdx in range(dataNum):
    rowIdx = dataIdx * 18
    x = test_array[rowIdx+9:rowIdx+10, :]
    predY = einsum('ij,ij', x, w) + b
    predYs.append(predY)
predYs


# In[ ]:

predYSeries = pd.Series(predYs)
predYSeries


# In[ ]:

_id = pd.Series(["id_" + str(i) for i in range(dataNum)])
# _id = test_data.ix[test_data[1] == "PM2.5", 0]
output = pd.concat([_id, predYSeries], axis=1)
output.columns=["id","value"]
output


# In[ ]:




# In[ ]:

output.ix[output["value"]<0,["value"]]=0
output


# In[ ]:

output.to_csv(OUTPUT_FOLDER + "output_linear_only-pm2.5.csv", index=False)


# In[ ]:



