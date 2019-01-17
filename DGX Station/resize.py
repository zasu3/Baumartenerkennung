import os

import PIL
from PIL import Image, ImageOps

from datetime import datetime

#  data
#path1 = '/Users/zes3/Documents/BaumbilderBA_augmented'    #path of folder of images
path1 = 'BaumbilderBA_augmented_training_resized_227'    #path of folder of images DGX
#path2 = '/Users/zes3/Documents/Bachelor_Thesis/BaumbilderBA_augmented_farbig_227'  #path of folder to save images
path2 = 'BaumbilderBA_augmented_training_resized_200'  #path of folder to save images DGX

# find images
listing = os.listdir(path1)

print("Started at: ")
print(datetime.now())
print("Resizing " +str(len(listing))+ " Images")

for file in listing:
    #print(file)
    if file == '.DS_Store':
        continue

    # image augmentation
    picture= Image.open(path1 + '/' + file)
    picture.resize((200, 200)).save(path2 + '/' + file)
