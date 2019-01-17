#from __future__ import print_function

#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
#import tensorflow as tf
import os
#from sklearn.preprocessing import OneHotEncoder
import PIL
from PIL import Image, ImageOps
#from numpy import *
import datetime

#  data
#path1 = '/Users/zes3/Documents/BaumbilderBA_augmented'    #path of folder of images
path1 = 'BaumbilderBA_augmented_validation'    #path of folder of images DGX
#path2 = '/Users/zes3/Documents/Bachelor_Thesis/BaumbilderBA_augmented_farbig_227'  #path of folder to save images
path2 = 'BaumbilderBA_augmented_validation_resized_227'  #path of folder to save images DGX

img_rows, img_cols = 227, 227

# find images
listing = os.listdir(path1)
print(datetime.datetime.now())
print("Images resized: ")
print(len(listing))

for file in listing:
    #print(file)
    if file == '.DS_Store':
        continue

    # image augmentation
    picture= Image.open(path1 + '/' + file)
    resized_picture = picture.resize((img_rows, img_cols))
    resized_picture.save(path2 + '/' + file, "PNG")
