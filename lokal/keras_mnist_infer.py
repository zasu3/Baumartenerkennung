from random import randint

import numpy as np

import keras
from keras.datasets import mnist
from keras import backend as K

model = keras.models.load_model('my_model.h5')
model.summary()

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train (ignored) and test sets
(_, _), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, num_classes)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('Prediction (probablity):', model.predict(x_test))
print('Prediction (classes):', model.predict_classes(x_test))

img = x_test[randint(0, len(x_test))]
#print(img.shape)
img = img.reshape(1, img_rows, img_cols, 1)
#print(img.shape)

img_prob = model.predict(img)
img_class = model.predict_classes(img)
prediction = img_class[0]
classname = img_class[0]
print("Prediction (probablity): ", img_prob)
print("Prediction (class): ",classname)

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
img[img > 0] = 0xFF
print(img.reshape(img_rows, img_cols))