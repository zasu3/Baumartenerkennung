import tensorflow as tf
import numpy as np
import os
from PIL import Image

filenames = tf.constant(['BaumbilderBA_augmented_farbig_resized/Alle_alten_Bern_1_Bergahorn2_71.png', 'BaumbilderBA_augmented_farbig_resized/Alle_alten_Bern_1_Buche2_446.png' ])
labels = tf.constant([0, 1])

# step 2: create a dataset returning slices of `filenames`
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# step 3: parse every image in the dataset using `map`
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, label

dataset = dataset.map(_parse_function)
dataset = dataset.batch(2)

# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()


print(images)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
filename,img = sess.run(images)
print (filename)
print(img)
