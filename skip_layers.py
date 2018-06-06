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
val_batch_size = 10000

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
fc1_layer_size = 1024
fc2_layer_size = 1024
fc3_layer_size = 1024
fc4_layer_size = 1024
fc5_layer_size = 1024
fc6_layer_size = 1024
fc7_layer_size = 1024
fc8_layer_size = 1024
'''
fc9_layer_size = 512
fc10_layer_size = 512
fc11_layer_size = 1024
fc12_layer_size = 1024
'''

def create_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=name)
    #return tf.Variable(tf.truncated_normal(shape, mean=1, stddev=0.05), name=name)

# originally 0.05 bias initiliaztion
def create_biases(size, name):
    return tf.Variable(tf.constant(1.05, shape=[size]), name=name)

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
             probability=1,
             activation="relu",
             dropout=False,
             dropout_rate=0):
    
    token_weights = identifier + "_weights"
    token_bias = identifier + "_bias"
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs], name=token_weights)
    weights = probability * weights
    biases = create_biases(num_outputs, name=token_bias)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    
    
    if activation == "relu":
        layer = tf.nn.relu(layer)
    if activation == "sigmoid":
        layer = tf.nn.sigmoid(layer)
    
    # if neither relu nor sigmoid, no activation function required

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
                     num_inputs=fc2_layer_size,
                     num_outputs=fc3_layer_size,
                     identifier="fc3")
'''
# no activation probability layer
layer_fc4 = create_fc_layer(input=layer_fc3,
                     num_inputs=fc3_layer_size,
                     num_outputs=fc4_layer_size,
                     identifier="fc4",
		     activation="",
                     probability=0.3)
'''
# identity mapping
#layer_fc4 = layer_fc2 + layer_fc4

layer_fc5 = create_fc_layer(input=layer_fc3,
                     num_inputs=fc4_layer_size,
                     num_outputs=fc5_layer_size,
                     identifier="fc5")

layer_fc6 = create_fc_layer(input=layer_fc5,
                     num_inputs=fc5_layer_size,
                     num_outputs=fc6_layer_size,            
                     identifier="fc6")

layer_fc7 = create_fc_layer(input=layer_fc6,
                     num_inputs=fc6_layer_size,
                     num_outputs=fc7_layer_size,
                     identifier="fc7")

layer_fc8 = create_fc_layer(input=layer_fc7,
                     num_inputs=fc7_layer_size,
                     num_outputs=num_classes,   
                     identifier="fc8",
	                 activation="sigmoid")
'''
layer_fc9 = create_fc_layer(input=layer_fc8,
                     num_inputs=fc8_layer_size,
                     num_outputs=fc9_layer_size,
                     identifier="fc9",
		dropout=True,
		dropout_rate=0.5)

layer_fc10 = create_fc_layer(input=layer_fc9,
                     num_inputs=fc9_layer_size,
                     num_outputs=fc10_layer_size,                     
                     identifier="fc10",
		dropout=True,
		dropout_rate=0.5)

layer_fc11 = create_fc_layer(input=layer_fc10,
                     num_inputs=fc10_layer_size,
                     num_outputs=fc11_layer_size,
                     identifier="fc11",
		dropout=True,
		dropout_rate=0.5)

layer_fc12 = create_fc_layer(input=layer_fc11,
                     num_inputs=fc11_layer_size,
                     num_outputs=num_classes,
                     use_relu=False,
                     identifier="fc12")
'''
y_pred = tf.nn.softmax(layer_fc8, name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc8,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
#optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)  #1e-4
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def show_progress(epoch, feed_dict_validate, val_loss):
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Validation Accuracy: {1:>6.1%},  Validation Loss: {2:.3f}"
  
    print(msg.format(val_acc, val_loss))

# To restore session we need a saver module
saver = tf.train.Saver()

session.run(tf.global_variables_initializer())
saver.restore(session, "models/test_model_" + ".ckpt")

def train():    
    x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(val_batch_size)
    feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}
    val_loss = session.run(cost, feed_dict=feed_dict_val)
    
    # print acc    
    show_progress(0, feed_dict_val, val_loss)

train()