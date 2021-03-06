from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from PIL import Image

#%pylab inline
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from datetime import datetime

tf.logging.set_verbosity(tf.logging.INFO)


#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""
'''Angepasst'''


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    img_rows,img_cols = 227,227
    color_channel = 3
    print(".-----------------------------------------------------------------")
    print(features["x"])
   # print(len(features["x"]))
    input_layer = tf.reshape(features["x"], [-1, img_rows, img_cols, color_channel])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
                             inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
                             inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
                                inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    tree_classes = 11
    logits = tf.layers.dense(inputs=dropout, units=tree_classes)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                                  loss=loss,
                                  global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
                                        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
                                      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def modify_image(image):
    # random adjustments
    image = tf.image.per_image_standardization(image) # makes model invariant to dynamic range
    image = tf.image.random_brightness(image, 0.25) # +/- 0.25
    image = tf.image.random_contrast(image, 0.5, 1.5)
    image = tf.image.random_hue(image, 0.5) # 0.0 - 0.5, aendert Farbe komplett
    image = tf.image.random_saturation(image, 0.2, 1.5)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    return image

def read_image(filename_queue):
    reader = tf.WholeFileReader()
    key,value = reader.read(filename_queue)
    image = tf.image.decode_png(value)
    return key,image

def main(unused_argv):
    with tf.Graph().as_default():
        # Load training and eval data
        path1 = 'BaumbilderBA_augmented_farbig_resized'
        path2 = 'selected_rotated_resized_227'  #path of folder to save images DGX

        # find images
        listing = os.listdir(path1)
        listing2 = os.listdir(path2)
        filenames = list()
        i = 0
        for file in listing:
            i=i+1
            if i==100:
                break;
            filenames.append(path1 + '/' + file)
        i=100
        for file in listing2:
            i=i-1
            if i==0:
                break;
            filenames.append(path2 + '/' + file)

        # https://stackoverflow.com/questions/34783030/saving-image-files-in-tensorflow
        filename_queue = tf.train.string_input_producer(filenames)
        filename,read_input = read_image(filename_queue)
        modified_image = modify_image(read_input)
        image = filename,modified_image

        # label
        label=np.ones((len(filenames),),dtype = int)
        images = np.ones((len(filenames),))	

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        for i in range(len(filenames)):
            filename,img = sess.run(image)
            #images[i] = img
            #print (filename)
            #img = Image.fromarray(img, "RGB")
            #imgplot = plt.imshow(img)
            #plt.show()
            if b"Bergahorn" in filename:
                label[i] = 0
            if b"Spitzahorn" in filename:
                label[i] = 1
            if b"Feldahorn" in filename:
                label[i] = 2
            if b"Buche" in filename:
                label[i] = 3
            if b"Birke" in filename:
                label[i] = 4
            if b"Eiche" in filename:
                label[i] = 5
            if b"Stechpalme" in filename:
                label[i] = 6
            if b"Ulme" in filename:
                label[i] = 7
            if b"Linde" in filename:
                label[i] = 8
            if b"Kirsche" in filename:
                label[i] = 9
            if b"Esche" in filename:
                label[i] = 10
            #print(str(i) + "->" + str(filename))
            #print(label[i])

        print(".-----------------------------------------------------------------")
        print("label-len")
        print(len(label))
        print(label)
        print("images-len")
        print(images)
        print(len(images))

        # prepare data and labels
        # the method shuffle() randomizes the items of a list in place.
        #data,Label = shuffle(image,label, random_state=2)
        data,Label = images,label
        train_data, eval_data, train_labels, eval_labels = train_test_split(data, Label, test_size=0.2, random_state=4)
        print(".-----------------------------------------------------------------")
        print(train_data)

        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
                                                  model_fn=cnn_model_fn, model_dir="/saved_models/tensorflow_model")

        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
                                                  tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        #train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                                                            x={"x": train_data},
                                                                    y=train_labels,
                                                                    batch_size=100,
                                                                    num_epochs=None,
                                                                    shuffle=True)
        mnist_classifier.train(
                               input_fn=train_input_fn,
                               steps=20000,
                               hooks=[logging_hook])

        # Evaluate the model and print results
        # eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(         
                                                            x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
        print(datetime.now())


if __name__ == "__main__":
     tf.app.run()
