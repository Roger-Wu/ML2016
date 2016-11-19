
# coding: utf-8

# In[1]:

import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


import sys

DATA_FOLDER_PATH = sys.argv[1]
MODEL_FILE_PATH = sys.argv[2] # output model

if DATA_FOLDER_PATH[-1] != '/':
	DATA_FOLDER_PATH = DATA_FOLDER_PATH + '/'

# In[2]:

LABELLED_DATA_FILE = DATA_FOLDER_PATH + 'all_label.p'
UNLABELED_DATA_FILE = DATA_FOLDER_PATH + 'all_unlabel.p'
TEST_DATA_FILE = DATA_FOLDER_PATH + 'test.p'

METHOD = 'cnn_supervised'

OUTPUT_FOLDER = 'output/'
MODEL_FOLDER = 'model/' + METHOD + '/'


# In[3]:

import os
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


# # Train

# ### Loading Training Data

# In[4]:

labelled_data = np.array(pickle.load(open(LABELLED_DATA_FILE, 'rb')))


# In[5]:

# generate x, y from data
X_train_label = labelled_data.reshape((5000, 3, 32, 32)).astype('float32') / 255

y_train_label_class = np.array([classIdx for classIdx in range(len(labelled_data)) for i in range(len(labelled_data[classIdx]))])
Y_train_label = np_utils.to_categorical(y_train_label_class, len(labelled_data))


# In[6]:

# from scipy.misc import toimage
# toimage(X_train_label[0])


# ### Training

# In[7]:


nb_classes = 10
img_rows, img_cols, img_channels = 32, 32, 3


# In[8]:

model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=X_train_label.shape[1:], dim_ordering="th"))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering="th"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering="th"))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering="th"))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), dim_ordering="th", border_mode='same'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[9]:

batch_size = 64
nb_epoch = 100


# In[10]:

model.fit(X_train_label, Y_train_label,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
#           validation_split=0.1,
          shuffle=True)


# In[11]:

model_name = (METHOD
    + '_filter-64-64-MP-64-64-APS-512-256-128'
    + '_epo-100' # + str(nb_epoch)
)

model.save(MODEL_FOLDER + model_name)


# # Test

# ### Loading test data 

# In[12]:

test_data = pickle.load(open(TEST_DATA_FILE, 'rb'))  # dict
X_test = np.array(test_data['data']).reshape((10000, 3, 32, 32)).astype('float32') / 255


# ### Predict

# In[14]:

result = model.predict_classes(X_test)
csv_content = list(zip(test_data['ID'], result.tolist()))
np.savetxt(OUTPUT_FOLDER + model_name + ".csv", csv_content, fmt="%i", header="ID,class", comments="", delimiter=",")


# In[ ]:




# In[ ]:




# In[ ]:



