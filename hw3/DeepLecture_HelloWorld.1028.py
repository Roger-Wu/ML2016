import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2,l1l2
from load_data import load_mnist_data, load_cifar10_data, load_news_data, load_mnist_noisy_data, load_cifar10_cnn_data
#categorical_crossentropy

model = Sequential()
model.add(Dense(input_dim=28*28,output_dim=689))
model.add(Activation('sigmoid'))
model.add(Dense(689))
model.add(Activation('sigmoid'))
model.add(Dense(689))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

(x_train,y_train),(x_test,y_test) = load_mnist_data(10000)

model.fit(x_train,y_train,batch_size=100,nb_epoch=20)

result = model.evaluate(x_test,y_test,batch_size=10000)
print(result[1])

model2 = Sequential()
model2.add(Convolution2D(25,3,3,input_shape=(1,28,28)))
model2.add(MaxPooling2D((2,2)))
model2.add(Convolution2D(50,3,3))
model2.add(MaxPooling2D((2,2)))
model2.add(Flatten())
model2.add(Dense(100))
model2.add(Activation('sigmoid'))
model2.add(Dense(10))
model2.add(Activation('softmax'))

model2.summary()

model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

(x_train,y_train),(x_test,y_test) = load_mnist_data(10000)

model2.fit(x_train.reshape(x_train.shape[0],1,28,28),y_train,batch_size=100,nb_epoch=20)

result = model2.evaluate(x_test.reshape(x_test.shape[0],1,28,28),y_test,batch_size=10000)
print(result[1])










