# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# hyper params
iterations = 10000
learning_rate = 0.05
input_size = 784
classes = 10

# later use
# architecture = [100,100,100,128,128,64,64,32,32] 

def weight_init(shape):
  weight = tf.Variable(tf.random_normal(shape))
  return weight
def bias_init(shape):
  bias = tf.Variable(tf.random_normal(shape))
  return bias

# cool progress bar function from SO
def progressBar(value, endvalue, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  # x is input placeholder
  x = tf.placeholder(tf.float32, [None, 784])

  W_1 = tf.Variable(tf.truncated_normal([784, 256]))
  b_1 = tf.Variable(tf.truncated_normal([256]))

  # random tensor for "probability" of each weight
  #P_1 = tf.Variable(tf.random_uniform([784, 100], minval=0, maxval=10000))

  # output of first layer
  out_1 = tf.matmul(x, W_1) + b_1
  out_1 = tf.sigmoid(out_1)
  
  W_2 = tf.Variable(tf.truncated_normal([256, 256]))
  b_2 = tf.Variable(tf.truncated_normal([256]))

  #P_2 = tf.Variable(tf.random_uniform([100, 100], minval=0, maxval=10000))

  out_2 = tf.matmul(out_1, W_2) + b_2
  out_2 = tf.sigmoid(out_2)

  W_3 = tf.Variable(tf.truncated_normal([256, 10]))
  b_3 = tf.Variable(tf.truncated_normal([10]))

  out_3 = tf.matmul(out_2, W_3) + b_3
  out_3 = tf.sigmoid(out_3)

  #P_3 = tf.Variable(tf.random_uniform([100, 100], minval=0, maxval=10000))

  #W_4 = tf.Variable(tf.truncated_normal([100, 10]))
  #b_4 = tf.Variable(tf.truncated_normal([10]))

  #out_4 = tf.matmul(out_3, W_4) + b_4
  #y = tf.nn.softmax(out_3)
  y = tf.sigmoid(out_3)
  
  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
  # outputs of 'y', and then average across the batch.

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Train
  for _ in range(iterations):
    progressBar(_, iterations)
<<<<<<< HEAD
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
=======
    batch_xs, batch_ys = mnist.train.next_batch(512)
    __train_step , acc = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
    print(acc)

>>>>>>> f5d3ecdb0c380651a1d490a362ac711c17b49632
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("\ntest acc")
  print(sess.run(
      accuracy, feed_dict={
          x: mnist.test.images,
          y_: mnist.test.labels
      }))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
