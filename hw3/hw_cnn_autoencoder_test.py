
# coding: utf-8

# In[ ]:

import numpy as np
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

import sys

DATA_FOLDER_PATH = sys.argv[1]
MODEL_FILE_PATH = sys.argv[2] # output model
OUTPUT_FILE_PATH = sys.argv[3]

if DATA_FOLDER_PATH[-1] != '/':
  DATA_FOLDER_PATH = DATA_FOLDER_PATH + '/'


# In[ ]:

METHOD = 'cnn_self-training'

LABELLED_DATA_FILE = DATA_FOLDER_PATH + 'all_label.p'
UNLABELED_DATA_FILE = DATA_FOLDER_PATH + 'all_unlabel.p'
TEST_DATA_FILE = DATA_FOLDER_PATH + 'test.p'


# In[ ]:

# import os
# if not os.path.exists(MODEL_FOLDER):
#     os.makedirs(MODEL_FOLDER)
# if not os.path.exists(OUTPUT_FOLDER):
#     os.makedirs(OUTPUT_FOLDER)


# # Train

# ### Loading Training Data

# In[ ]:

# labelled_data = np.array(pickle.load(open(LABELLED_DATA_FILE, 'rb')))
# unlabeled_data = np.array(pickle.load(open(UNLABELED_DATA_FILE, 'rb')))


# # In[ ]:

# nb_classes = 10
# img_rows, img_cols, img_channels = 32, 32, 3


# # In[ ]:

# # generate x, y from data

# # reshape labelled data to (5000, 3, 32, 32)
# X_train_label = labelled_data.reshape((5000, img_channels, img_rows, img_cols)).astype('float32') / 255

# # reshape unlabeled data to (45000, 3, 32, 32)
# X_train_unlabel = unlabeled_data.reshape((unlabeled_data.shape[0], img_channels, img_rows, img_cols)).astype('float32') / 255

# y_train_label_class = np.array([classIdx for classIdx in range(len(labelled_data)) for i in range(len(labelled_data[classIdx]))])
# Y_train_label = np_utils.to_categorical(y_train_label_class, nb_classes)



model = load_model(MODEL_FILE_PATH)


test_data = pickle.load(open(TEST_DATA_FILE, 'rb'))  # dict
X_test = np.array(test_data['data']).reshape((10000, 3, 32, 32)).astype('float32') / 255

result = model.predict_classes(X_test)
csv_content = list(zip(test_data['ID'], result.tolist()))
np.savetxt(OUTPUT_FILE_PATH, csv_content, fmt="%i", header="ID,class", comments="", delimiter=",")

