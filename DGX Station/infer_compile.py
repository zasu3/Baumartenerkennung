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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import datetime

image = Image.open('/home/duser/BaumbilderBA_augmented_farbig_resized/Alle_alten_Bern_10_Birke1_121.png')

model = keras.models.load_model('08-12-2018_Conv1_15epochs.h5')
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(0.0005),
              loss='categorical_crossentropy',
              #optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
a = array(image).reshape(1,227,227,3)
score = model.predict(a)
print(score)
