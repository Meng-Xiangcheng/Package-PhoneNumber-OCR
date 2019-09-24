'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import scipy.io as sio
import numpy as np

batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
##(x_train, y_train), (x_test, y_test) = mnist.load_data()
data = h5py.File('C:\\Users\\Administrator\\Desktop\\ocr\\sample4mat\\data.mat','r')
x = data['X']
x_test = data['Xtest']
y = data['y']
y_test = data['ytest']

temp = np.zeros(x.shape)
temp[:,:] = x[:,:]
x_train = temp.T
temp = np.zeros(y.shape)
temp[:,:] = y[:,:]
y_train = temp.T
temp = np.zeros(x_test.shape)
temp[:,:] = x_test[:,:]
x_test = temp.T
temp = np.zeros(y_test.shape)
temp[:,:] = y_test[:,:]
y_test = temp.T

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),padding='same', 
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score_test = model.evaluate(x_test, y_test, verbose=0)
score_train = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1])
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1])


#valaccy=np.transpose(history.history['val_acc'])
#valloss=np.transpose(history.history['val_loss'])
#output=np.array([valaccy,valloss])
#np.savetxt('save.txt',output)

validation_data=(x_test, y_test)
val_predict=(np.asarray(model.predict(validation_data[0]))).round()
val_targ = validation_data[1]

model.save('model.h5')