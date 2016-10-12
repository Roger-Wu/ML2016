
# coding: utf-8

# In[2]:

import pandas as pd


# In[8]:

train_data = pd.read_csv("data/train.csv", nrows=1000, encoding="big5")


# In[29]:

train_data[train_data.ix[:,1]=="豐原"]


# In[9]:

train_data[(==1) & (b ==2)]


# In[ ]:

usecols("=")

