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


# Step 3: parse every image in the dataset using `map`
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, label

def main(unused_argv):
    with tf.Graph().as_default():
        filenames = tf.constant(['dworkspace/BaumbilderBA_augmented_farbig_resized/Alle_alten_Bern_1_Bergahorn2_71.png', 'dworkspace/BaumbilderBA_augmented_farbig_resized/Alle_alten_Bern_1_Buche2_446.png' ])
        labels = tf.constant([0, 1])

        # step 2: create a dataset returning slices of `filenames`
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(2)

        # step 4: create iterator and final input tensor
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()


        print(images)
        print(labels)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        
        filename,img = sess.run(images)
        print (filename)
        # prepare data and labels
        # the method shuffle() randomizes the items of a list in place.
        #data,Label = shuffle(image,label, random_state=2)
        #data,Label = images,labels
        print("hier")

     #   train_data, eval_data, train_labels, eval_labels = train_test_split(data, Label, test_size=0.2, random_state=4)
        print(".-----------------------------------------------------------------")
        #print(train_data)
        print("hier")
        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
                                                  model_fn=cnn_model_fn, model_dir="/saved_models/tensorflow_model")

        print("hier")
        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
                                                  tensors=tensors_to_log, every_n_iter=50)

        print("hier")
        # Train the model
        #train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                                                            x={"x": images},
                                                                    y=labels,
                                                                    batch_size=100,
                                                                    num_epochs=None,
                                                                    shuffle=True)
        
        print("hier")
        mnist_classifier.train(
                               input_fn=train_input_fn,
                               steps=20000,
                               hooks=[logging_hook])

        # Evaluate the model and print results
        # eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(         
                                                            x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
        
        print("hier")
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
        print(datetime.now())


if __name__ == "__main__":
     tf.app.run()
