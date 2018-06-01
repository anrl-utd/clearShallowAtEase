import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import sys

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

batch_size = 256
val_batch_size = 2000

#Prepare input data
classes = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']
num_classes = len(classes)

img_size = 32
num_channels = 3
train_path="train_dir"
val_path = "validation_dir"

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, val_path, img_size, classes)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))



session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

##Network graph params
fc1_layer_size = 256 
fc2_layer_size = 256
fc3_layer_size = 256 
fc4_layer_size = 256 
fc5_layer_size = 256 
fc6_layer_size = 256

def create_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=name)

def create_biases(size, name):
    return tf.Variable(tf.constant(0.05, shape=[size]), name=name)

# First layer must be a flatten layer
def create_flatten_layer(input, batch_size, img_size, num_channels):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    #layer_shape = layer.get_shape()
    layer_shape = [batch_size, img_size, img_size, num_channels]

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    #num_features = layer_shape[1:4].num_elements()
    num_features = img_size * img_size * num_channels

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(input, [-1, num_features])

    return layer

def create_fc_layer(input,
             num_inputs,    
             num_outputs,
             identifier,
             use_relu=True,
             dropout=False,
             dropout_rate=0):
    
    token_weights = identifier + "_weights"
    token_bias = identifier + "_bias"
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs], name=token_weights)
    biases = create_biases(num_outputs, name=token_bias)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    
    if use_relu:
        layer = tf.nn.relu(layer)
    else:
        layer = tf.nn.sigmoid(layer)

    if dropout:
        layer = tf.layers.dropout(layer, rate=dropout_rate, training=True)

    return layer

flatten = create_flatten_layer(x, batch_size, img_size, num_channels)

# fc1_layer_size neurons with relu activation
layer_fc1 = create_fc_layer(input=flatten,
                     num_inputs=img_size*img_size*num_channels,
                     num_outputs=fc1_layer_size,
                     identifier="fc1")

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc1_layer_size,
                     num_outputs=fc2_layer_size,
                 identifier="fc2")

layer_fc3 = create_fc_layer(input=layer_fc2,
                     num_inputs=fc3_layer_size,
                     num_outputs=fc4_layer_size,
                     identifier="fc3",
                     dropout=True,
                     dropout_rate=0.9)

layer_fc4 = create_fc_layer(input=layer_fc3,
                     num_inputs=fc4_layer_size,
                     num_outputs=fc5_layer_size,
                     use_relu=True,
                     identifier="fc4",
                     dropout=True,
                     dropout_rate=0.9)

layer_fc5 = create_fc_layer(input=layer_fc4,
                     num_inputs=fc5_layer_size,
                     num_outputs=fc6_layer_size,
                     identifier="fc5")

layer_fc6 = create_fc_layer(input=layer_fc5,
                     num_inputs=fc6_layer_size,
                     num_outputs=num_classes,
                     use_relu=False,
                     identifier="fc6")

y_pred = tf.nn.softmax(layer_fc6, name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc6,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)  #1e-4
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
  
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

# Save non-dropout layers
saver = tf.train.Saver()

# if we load the model, we restore it from the ckpt
#if load_model:
#    saver.restore(session, "models/test_model_" + ".ckpt")

session.run(tf.global_variables_initializer())

total_iterations = 0

def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(val_batch_size)

        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            print(int(i))

    print(int(num_iteration))
    total_iterations += num_iteration

train(num_iteration=300000)
saver.save(session, "models/test_model"+"_"+".ckpt")
