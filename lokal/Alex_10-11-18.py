#from __future__ import print_function

import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
import tensorflow as tf
import os
#from sklearn.preprocessing import OneHotEncoder
import PIL
from PIL import Image, ImageOps
from numpy import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# input image dimensions
img_rows, img_cols = 227, 227

# number of channels
img_channels = 3

#  data
#path1 = '/Users/zes3/Documents/BaumbilderBA_augmented'    #path of folder of images
path1 = '/BaumbilderBA_augmented'    #path of folder of images DGX
#path2 = '/Users/zes3/Documents/Bachelor_Thesis/BaumbilderBA_augmented_farbig_227'  #path of folder to save images
path2 = '/BaumbilderBA_augmented_farbig_227'  #path of folder to save images DGX

# find and resize images
listing = os.listdir(path1)

for file in listing:
    #print(file)
    if file == '.DS_Store':
        continue
    im = Image.open(path1 + '/' + file)
    img = im.resize((img_rows,img_cols))
    #need to do some more processing here
    img.save(path2 +'/' + file, "PNG") # warum jpeg?

    # image augmentation
    picture= Image.open(path2 + '/' + file)
    picture.rotate(90).save(path2 + '/' + file[:-4] + '_rotated90.png')
    picture.rotate(180).save(path2 + '/' + file[:-4] + '_rotated180.png')
    picture.rotate(270).save(path2 + '/' + file[:-4] + '_rotated270.png')

# flatten images
imlist = os.listdir(path2)

if '.DS_Store' in imlist:
    imlist.remove('.DS_Store')


# open one image to get size
im1 = array(Image.open(path2 +'/' + imlist[0]))
m,n = im1.shape[0:2]

imnbr = len(imlist) # get the number of images


# prepare parameters and data
batch_size = 128
num_classes = 11
epochs = 25

(X, y) = (train_data[0],train_data[1])

# STEP 1: split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#TODO ist letzter parameter hier auch channels?
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

img=X_train[0]

#plt.imshow(img)
#plt.show()


# prepare cnn and start training
# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

input_shape = (img_rows, img_cols, img_channels)

# build cnn
model = Sequential()
# convolutional layer 1
model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=4,
                 activation='relu',
                 input_shape=input_shape))
# max pooling 1
model.add(MaxPooling2D(pool_size=(3, 3), strides=2)
# convolutional layer 2
model.add(Conv2D(filters=256, kernel_size(5, 5), strides=1,
                 padding=2, activation='relu'))
# max pooling 2
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
# convolutional layer 3
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1,
                padding=2, activation='relu'))
# convolutional layer 4
model.add(Conv2D(filters=384, kernel_size(3, 3), strides=1,
                padding=1, activation='relu'))
# convolutional layer 5
model.add(Conv2D(filters=256, kernel_size(3, 3), strides=1,
                padding=1, activation='relu'))
#max pooling 3
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
# dropout
model.add(Dropout(0.5))
model.add(Flatten())
# fully connected layer 1
model.add(Dense(256*6*6, activation='relu')) #ohne dropout 12*12?
model.add(Dropout(0.5))
# fully connected layer 2
model.add(Dense(4096, activation='relu'))
# fully connected layer 3
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
          
model.save('Alex_18-11-18.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
