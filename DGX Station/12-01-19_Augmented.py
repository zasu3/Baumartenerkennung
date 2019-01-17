import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
import tensorflow as tf
import os
#from sklearn.preprocessing import OneHotEncoder
import PIL
from PIL import Image, ImageOps
from numpy import *

from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K

# visualisation imports
#from keras.callbacks import TensorBoard
from time import time

from datetime import datetime

# input image dimensions
img_rows, img_cols = 227, 227

# number of channels
img_channels = 3

# directory for output and logs
log_dir = 'Augmented_31-12-18'

#  data
#path1 = 'BaumbilderBA_augmented_training_resized_227'
path2 = 'BaumbilderBA_augmented_training_resized_227_rotated'
#path_val = 'BaumbilderBA_augmented_validation_resized_227'
path_valrot = 'BaumbilderBA_augmented_validation_resized_227_rotated'

# find images
#imlist1 = os.listdir(path1)
imlist2 = os.listdir(path2)
imlist = imlist2 # imlist1 + imlist2

#imlistval1 = os.listdir(path_val)
imlistvalrot = os.listdir(path_valrot)
imlistval = imlistvalrot # imlistval1 + imlistvalrot

if '.DS_Store' in imlist:
    imlist.remove('.DS_Store')

# get the number of images
imnbr = len(imlist)
imnval = len(imlistval)
#print("Anzahl Training Files:")
#print(imnbr)
#print("Anzahl Validation Files:")
#print(imnval)

# test with some images
#imnbr = 40000
#imlist1 = imlist[:40000]

#imnbr = 10000
#imnval = imnbr
#imlistval = imlistval[:10000]

# create matrix to store all flattened images
#immatrix1 = array([array(Image.open(path1 + '/' + im2)).flatten()
#                  for im2 in imlist1],'f')
immatrix2 = array([array(Image.open(path2 + '/' + im3)).flatten()
                  for im3 in imlist2], 'f')
#immatrix_val1 = array([array(Image.open(path_val + '/' + im4)).flatten()
#                  for im4 in imlistval1],'f')
immatrix_valrot = array([array(Image.open(path_valrot + '/' + im5)).flatten()
                  for im5 in imlistvalrot], 'f')
print("Shape von immatrix1, 2 und gesamt")
#print(immatrix1.shape)
print(immatrix2.shape)
immatrix = immatrix2 #np.concatenate((immatrix1, immatrix2))
print(immatrix.shape)
num_samples = imnbr


print("Shape von immatrix_val1, _valrot und gesamt")
immatrix_val = immatrix_valrot #np.concatenate((immatrix_val1, immatrix_valrot))
#print(immatrix_val1.shape)
print(immatrix_valrot.shape)
print(immatrix_val.shape)
num_samples_val = imnval

# label
label=np.ones((num_samples,),dtype = int)
label_val=np.ones((imnval,),dtype = int)

print("Label setzen")
def get_label(imlist,label):
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
        #print(str(i) + "->" + imlist[i])
        #print(label[i])
        i += 1

get_label(imlist, label)
get_label(imlistval, label_val)

# prepare data and labels
# the method shuffle() randomizes the items of a list in place.
data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]


# prepare parameters and data
batch_size = 100
num_classes = 11
epochs = 50

# prepare training data
(X_train, y_train) = (train_data[0],train_data[1])

# prepare validation data
valdata,vallabel = shuffle(immatrix_val,label_val, random_state=2)
test_data = [valdata,vallabel]

(X_test, y_test) = (test_data[0],test_data[1])

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# save class labels to disk to color data points in TensorBoard accordingly
#with open(log_dir +'/metadata.tsv', 'w') as f:
#    np.savetxt(f, y_test)

# prepare cnn
# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

input_shape = (img_rows, img_cols, img_channels)

# build cnn
model = Sequential()

# convolutional layer 1
conv_1 = model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=4, activation='relu', input_shape=input_shape))
# max pooling 1
max_1 = model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# convolutional layer 2
conv_2 = model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, activation='relu'))
# max pooling 2
max_2 = model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
# convolutional layer 3
conv_3 = model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1,
                 activation='relu'))
# convolutional layer 4
conv_4 = model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1,
                 activation='relu'))
# convolutional layer 5
conv_5 = model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                 activation='relu'))
#max pooling 3
max_3 = model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
# dropout
dropout_1 = model.add(Dropout(0.5))
flatten = model.add(Flatten())
# fully connected layer 1
fully_1 = model.add(Dense(256*6*6, activation='relu'))
#dropout_2 = model.add(Dropout(0.5))
# fully connected layer 2
fully_2= model.add(Dense(4096, activation='relu'))
# fully connected layer 3
fully_3 = model.add(Dense(num_classes, activation='softmax'))

# Visualisation
# Launch the graph in a session.
#sess = tf.Session()
# Create a summary writer, add the 'graph' to the event file.
#writer = tf.summary.FileWriter(log_dir, sess.graph)

#embedding_layer_names = set(layer.name
#                            for layer in model.layers
#                            if layer.name.startswith('conv'))

#tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=5, batch_size = batch_size, write_graph = True, write_grads = True, write_images = True, embeddings_freq = 25, embeddings_layer_names = embedding_layer_names, embeddings_data=X_test, update_freq='epoch')
# embeddings_metadata = [], embeddings_data = []
# embeddings_layer_names:
# a list of names of layers to keep eye on. If NULL or empty list all the embedding layers will be watched.

#embeddings = tf.Variable(tf.random_normal([num_samples, num_classes], -1.0, 1.0, name='tree_embedding'))
#random_uniform?

model.summary()

datagen_train = ImageDataGenerator(
    #featurewise_std_normalization=True,

    #width_shift_range=0.2,  # randomly shift images horizontally 
    #height_shift_range=0.2,# randomly shift images vertically 

    horizontal_flip=True, # randomly flip images horizontally
    vertical_flip=True)

# fit augmented image generator on data
datagen_train.fit(X_train)

optimizer=Adam(lr=0.0001)
# initialize
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              #optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# train and evaluate
'''history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test)#,
 #         callbacks=[tensorboard]
          )'''
history = model.fit_generator(datagen_train.flow(X_train, y_train, batch_size=batch_size),
                                                 validation_data=(X_test, y_test),
                                                 epochs=epochs,
                                                 steps_per_epoch=X_train.shape[0]/batch_size,
                                                 verbose=1)
                                                 #callbacks=[checkpointer,lrate,tensorboard], verbose=1

score = model.evaluate(X_test, y_test, verbose=0)
#model.save('21-12-18_3_Augmented_150epochs.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(datetime.now())


# https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras
# plotting training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()
plt.savefig(log_dir+'/loss.png')
plt.clf()

# plotting training and validation accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, color='blue', label='Training acc')
plt.plot(epochs, val_acc, color='magenta', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()
plt.savefig(log_dir+'/accuracy.png')

print("on validation data")
pred1=model.evaluate(X_test,y_test)
print("accuaracy", str(pred1[1]*100))
print("Total loss",str(pred1[0]*100))

# predict results
#results = model.predict(X_test)

# select the indix with the maximum probability
#results = np.argmax(results,axis = 1)

#submissions=pd.DataFrame({"ImageId": list(range(1,len(results)+1)),
#                         "Label": results})
#submissions.to_csv("validierung.csv", index=False, header=True)

modelpath = log_dir+'/31-12-18_augmented.hdf5'
print("Model saved to: ")
print(modelpath)
model.save(modelpath)
