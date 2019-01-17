#from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
from numpy import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# input image dimensions
img_rows, img_cols = 227, 227

# number of channels
img_channels = 3

#  data
path1 = 'BaumbilderBA_augmented'    #path of folder of images
path2 = 'BaumbilderBA_augmented_farbig_resized'  #path of folder to save images

# find and resize images
listing = os.listdir(path1)
#num_folders=size(listing)

#for folder in listing:
#    if folder == '.DS_Store':
#        continue
#    for root, dirs, files in os.walk(path1 + "/" + folder):
#        herkunftsort = os.path.basename(os.path.dirname(root))
#        print(herkunftsort)
'''        if root == '.DS_Store':
            continue'''
'''for file in listing:
    #print(file)
    if file == '.DS_Store':
       continue
    path = os.path.join(path1,file)
    im = Image.open(path)
    img = PIL.ImageOps.fit(im, (img_rows, img_cols), method=0, bleed=0.0)
    img = im.resize((img_rows,img_cols))
    #need to do some more processing here
    img.save(path2 +'/' + file, "PNG")
'''
# flatten images
imlist = os.listdir(path2)

if '.DS_Store' in imlist:
    imlist.remove('.DS_Store')

# open one image to get size
im1 = array(Image.open(path2 + '/' + imlist[0]))
m,n = im1.shape[0:2]

imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(path2 + '/' + im2)).flatten()
              for im2 in imlist],'f')
num_samples = imnbr
print(num_samples)
print(immatrix.shape)


# label
label=np.ones((num_samples,),dtype = int)

i=0
while i < len(imlist):
    fileName = imlist[i]
    #print(str(i) + "->" + fileName)
    if "Bergahorn" in fileName:
        label[i] = 0
        #print("Bergahorn")
    if "Spitzahorn" in fileName:
        label[i] = 1
        #print("Spitzahorn")
    if "Feldahorn" in fileName:
        label[i] = 2
        #print("Feldahorn")
    if "Buche" in fileName:
        label[i] = 3
        #print("Buche")
    if "Birke" in fileName:
        label[i] = 4
        #print("Birke")
    if "Eiche" in fileName:
        label[i] = 5
        #print("Eiche")
    if "Stechpalme" in fileName:
        label[i] = 6
        #print("Stechpalme")
    if "Ulme" in fileName:
        label[i] = 7
        #print("Ulme")
    if "Linde" in fileName:
        label[i] = 8
        #print("Linde")
    if "Kirsche" in fileName:
        label[i] = 9
        #print("Kirsche")
    if "Esche" in fileName:
        label[i] = 10
        #print("Esche")
    print(str(i) + "->" + imlist[i])
    print(label[i])
    i += 1

# prepare data and labels
# the method shuffle() randomizes the items of a list in place.
data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

print(immatrix[7683].shape)
#img=immatrix[7683].reshape(img_rows,img_cols)

#plt.imshow(img)
#plt.imshow(img,cmap='gray')
#plt.show()
print (train_data[0].shape)
print (train_data[1].shape)


# prepare parameters and data
batch_size = 128
num_classes = 11
epochs = 25

(X, y) = (train_data[0],train_data[1])

# STEP 1: split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#TODO was bedeutet der letzte parameter?????????
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

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

input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
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
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('keras_trees_17-11-18.h5')
