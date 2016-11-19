
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

labelled_data = np.array(pickle.load(open(LABELLED_DATA_FILE, 'rb')))
unlabeled_data = np.array(pickle.load(open(UNLABELED_DATA_FILE, 'rb')))


# In[ ]:

nb_classes = 10
img_rows, img_cols, img_channels = 32, 32, 3


# In[ ]:

# generate x, y from data

# reshape labelled data to (5000, 3, 32, 32)
X_train_label = labelled_data.reshape((5000, img_channels, img_rows, img_cols)).astype('float32') / 255

# reshape unlabeled data to (45000, 3, 32, 32)
X_train_unlabel = unlabeled_data.reshape((unlabeled_data.shape[0], img_channels, img_rows, img_cols)).astype('float32') / 255

y_train_label_class = np.array([classIdx for classIdx in range(len(labelled_data)) for i in range(len(labelled_data[classIdx]))])
Y_train_label = np_utils.to_categorical(y_train_label_class, nb_classes)


# In[ ]:

# # visualize
# from scipy.misc import toimage
# toimage(X_train_unlabel[5])


# ### Training

# In[ ]:

from scipy.stats import entropy

def certainty(prob_arr):  # higher is more certain
    return -entropy(prob_arr)

def uncertainty(prob_arr):  # lower is better. for descending-ordered sorting
    return entropy(prob_arr)

#     sorted_arr = np.sort(prob_arr)
#     return sorted_arr[-1] / sorted_arr[-2]


# #### First model

# In[ ]:

# first model
batch_size = 64
nb_epoch = 100

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

model.add(Flatten())  # n * 8 * 8, for 64 -> 4096

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


# In[ ]:

model.fit(X_train_label, Y_train_label,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    shuffle=True)

model_1 = model


# #### Second model

# In[ ]:

# use the model to label unlabeled data
Y_train_unlabel_proba = model.predict_proba(X_train_unlabel)  # same as model.predict
Y_train_unlabel_uncertainty = np.apply_along_axis(uncertainty, 1, Y_train_unlabel_proba)

Y_train_unlabel_class = model.predict_classes(X_train_unlabel)
Y_train_unlabel = np_utils.to_categorical(Y_train_unlabel_class, nb_classes)

# sort unlabeled data by uncertainty
sorted_idxs = Y_train_unlabel_uncertainty.argsort()
X_train_unlabel_sorted = X_train_unlabel[sorted_idxs]
Y_train_unlabel_sorted = Y_train_unlabel[sorted_idxs]


# In[ ]:

extract_ratio = 0.4

# extract high-certainty unlabeled data
nb_extract = int(X_train_unlabel.shape[0] * extract_ratio)
X_train_unlabel_certain, X_train_unlabel_uncertain = X_train_unlabel_sorted[:nb_extract], X_train_unlabel_sorted[nb_extract:]
Y_train_unlabel_certain, Y_train_unlabel_uncertain = Y_train_unlabel_sorted[:nb_extract], Y_train_unlabel_sorted[nb_extract:]

# update labelled dataset and unlabeled dataset
X_train_label_pool = np.concatenate((X_train_label, X_train_unlabel_certain))
Y_train_label_pool = np.concatenate((Y_train_label, Y_train_unlabel_certain))
X_train_unlabel_pool = X_train_unlabel_uncertain
Y_train_unlabel_pool = Y_train_unlabel_uncertain


# In[ ]:

# second model

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

model.add(Flatten())  # n * 8 * 8, for 64 -> 4096

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


# In[ ]:

batch_size = 64
nb_epoch = 30 # int(500000 / X_train_label_pool.shape[0])

model.fit(X_train_label_pool, Y_train_label_pool,
    batch_size=batch_size,
    nb_epoch=nb_epoch,  # nb_epoch,
    shuffle=True)



model.save(MODEL_FILE_PATH)


