import os

import PIL
from PIL import Image, ImageOps

from datetime import datetime

#  data
#path1 = '/Users/zes3/Documents/BaumbilderBA_augmented'    #path of folder of images
path1 = 'BaumbilderBA_augmented_validation_resized_200'    #path of folder of images DGX
#path2 = '/Users/zes3/Documents/Bachelor_Thesis/BaumbilderBA_augmented_farbig_227'  #path of folder to save images
path2 = 'BaumbilderBA_augmented_validation_resized_200_rotated'  #path of folder to save images DGX

print("Start at:")
print(datetime.now())
# find images in path1
listing = os.listdir(path1)
print("Rotating "+str(len(listing))+" files")

# open, rotate and save images in path2
for file in listing:
    #print(file)
    if file == '.DS_Store':
        continue

    # image augmentation
    picture= Image.open(path1 + '/' + file)
    picture.rotate(90).save(path2 + '/' + file[:-4] + '_rotated90.png')
    picture.rotate(180).save(path2 + '/' + file[:-4] + '_rotated180.png')
    picture.rotate(270).save(path2 + '/' + file[:-4] + '_rotated270.png')

print("Rotation finished: ")
print(datetime.now())
