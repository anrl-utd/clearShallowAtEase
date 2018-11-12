import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import sys
import argparse

# parse the model number for batch training a family of models
parser = argparse.ArgumentParser()
parser.add_argument("-mn", "--modelnumber", type=int, default=1,
                    help="The number models to train")
args = parser.parse_args()

#np:2278
#tf: 5495

#Adding Seed so that random initialization is consistent
from numpy.random import seed
r = random.randint(1,10000)
print("numpy seed: ", r)
seed(r)
#seed(2278)

from tensorflow import set_random_seed
r_tf = random.randint(1,10000)
print("tf seed: ", r_tf)
set_random_seed(r_tf)
#set_random_seed(5495)

# save random seeds to seed file
with open("models/fixedGuard/seeds.txt", "a") as myfile:
    myfile.write(str(args.modelnumber))
    myfile.write("\n")
    myfile.write(str(r))
    myfile.write("\n")
    myfile.write(str(r_tf))
    myfile.write("\n")


batch_size = 64
val_batch_size = 64
iter_ = 20200
lr_ = 1e-1

# Prepare input data
classes = ['person_images', 'car_images', 'bus_images']
num_classes = len(classes)

## This is our survivability vector, defining our weighted additions with our hueristic presented in the paper.
#survive = [1, 0.99, 0.95, 0.95, 0.9, 0.9, 0.9, 0.9]
survive = [0.99, 0.98, 0.94, 0.93, 0.9, 0.9, 0.87, 0.87]

# we assign each index in the vector to it's corresponding fog or edge node (Defined in our model diagram)
f_3 = survive[0]
f_2 = survive[1]
f_1_1 = survive[2]
f_1_2 = survive[3]
e_1 = survive[4]
e_2 = survive[5]
e_3 = survive[6]
e_4 = survive[7]

img_size = 32
num_channels = 3
num_cameras = 6
train_path="/home/sid/datasets/mvmc_p/train_dir/"
val_path = "/home/sid/datasets/mvmc_p/test_dir/"

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, val_path, img_size, classes)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, num_cameras, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

## Network graph params. We use homogeneous layer size so that we may do the weighted addition to preserve information during layer failures.
fc1_layer_size = 32
fc2_layer_size = 32
fc3_layer_size = 32
fc4_layer_size = 32
fc5_layer_size = 32
fc6_layer_size = 32
fc7_layer_size = 32
fc8_layer_size = 32
fc9_layer_size = 32
fc10_layer_size = 32
fc11_layer_size = 32
fc12_layer_size = 32


## Abstraction of layer creation to functions for easier creation.

def create_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=name)
    #return tf.Variable(tf.truncated_normal(shape, mean=1, stddev=0.05), name=name)

# originally 0.05 bias initiliaztion
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
    #split0, split1, split2, split3, split4, split5 = tf.split(layer, 6, 1)
    #output = [split0, split1, split2, split3, split4, split5]

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

## Now we begin defining our network graph. We first start by taking the initial input of 6 images, and 
split0, split1, split2, split3, split4, split5 = tf.split(x, 6, 1)
inputs = [split0, split1, split2, split3, split4, split5]

print(inputs[0].get_shape())
print(inputs)
flatten_out = []
for i in range(6):
    flatten_out.append(create_flatten_layer(inputs[i], batch_size, img_size, num_channels))

flatten_combine = []
flatten_combine.append(flatten_out[0] + flatten_out[1])
flatten_combine.append(flatten_out[2])
flatten_combine.append(flatten_out[3] + flatten_out[4])
flatten_combine.append(flatten_out[5])

layer1_fc = []
counter = 0
for camera in flatten_combine:
    layer_tmp = create_fc_layer(input=camera,
                     num_inputs=img_size*img_size*num_channels,
                     num_outputs=fc1_layer_size,
                     identifier='fc1_'+str(counter))
    counter += 1
    layer1_fc.append(layer_tmp)
print(layer1_fc[0].get_shape())

layer2_1_sum = layer1_fc[0]

w_1 = e_2 / (e_2 + e_3 + e_4)
w_2 = e_3 / (e_2 + e_3 + e_4)
w_3 = e_4 / (e_2 + e_3 + e_4)

layer1_fc[1] = w_1 * layer1_fc[1]
layer1_fc[2] = w_2 * layer1_fc[2]
layer1_fc[3] = w_3 * layer1_fc[3]
layer2_2_sum = sum(layer1_fc[1:])

layer2_1_fc = create_fc_layer(input=layer2_1_sum,
                     num_inputs=fc1_layer_size,
                     num_outputs=fc2_layer_size,
                     identifier='fc2_1')
 
layer2_2_fc = create_fc_layer(input=layer2_2_sum,
                     num_inputs=fc1_layer_size,
                     num_outputs=fc2_layer_size,
                     identifier='fc2_2')

layer3_1_fc = create_fc_layer(input=layer2_2_fc,
                     num_inputs=fc2_layer_size,
                     num_outputs=fc3_layer_size,
                     identifier='fc3_1')

layer3_out = (f_1_1 / (f_1_1 + f_1_2)) * layer2_1_fc + (f_1_2 / (f_1_1 + f_1_2)) * layer3_1_fc

layer_fc4 = create_fc_layer(input=layer3_out,
                     num_inputs=fc3_layer_size,
                     num_outputs=fc4_layer_size,
                     identifier="fc4")

layer_fc5 = create_fc_layer(input=layer_fc4,
                     num_inputs=fc4_layer_size,
                     num_outputs=fc5_layer_size,
                     identifier="fc5")

w_1 = f_2 / (f_2 + f_1_1 + f_1_2)
w_2 = f_1_2 / (f_1_1 + f_1_2 + f_2)
w_3 = f_1_1 / (f_1_1 + f_1_2 + f_2)

layer_fc6 = create_fc_layer(input=w_1*layer_fc5 + w_2*layer3_1_fc + w_3*layer2_1_fc,
                     num_inputs=fc5_layer_size,
                     num_outputs=fc6_layer_size,
                     identifier="fc6")

layer_fc7 = create_fc_layer(input=layer_fc6,
                     num_inputs=fc6_layer_size,
                     num_outputs=fc7_layer_size,
                     identifier="fc7")

w_1 = f_3 / (f_2 + f_3)
w_2 = f_2 / (f_2 + f_3)

layer_fc8 = create_fc_layer(input=w_1*layer_fc7 + w_2*layer_fc5,
                     num_inputs=fc7_layer_size,
                     num_outputs=fc8_layer_size,
                     identifier="fc8")

layer_fc9 = create_fc_layer(input=layer_fc8,
                     num_inputs=fc8_layer_size,
                     num_outputs=fc9_layer_size,
                     identifier="fc9")

layer_fc10 = create_fc_layer(input=layer_fc9,
                     num_inputs=fc9_layer_size,
                     num_outputs=fc10_layer_size,
                     identifier="fc10")

layer_fc11 = create_fc_layer(input=layer_fc10,
                     num_inputs=fc10_layer_size,
                     num_outputs=num_classes,
                     identifier="fc11")

y_pred = tf.nn.softmax(layer_fc11, name='y_pred')
print(y_pred.get_shape())

y_pred_cls = tf.argmax(y_pred, dimension=1)
print(y_pred_cls.get_shape())

session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc11,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdagradOptimizer(learning_rate=lr_).minimize(cost)  #1e-4
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

precision = tf.metrics.precision(y_true_cls, y_pred_cls)

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
  
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

# Save non-dropout layers
saver = tf.train.Saver()

session.run(tf.global_variables_initializer())

total_iterations = 0

def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(val_batch_size)
        r_ = [random.random() for x in range(len(survive))]
        
        # calc prob of one component failing, which we take to be the average failure given a
        # survivability mapping 'survive'
        #fail = [1 - x for x in survive]
        #fail_prob = sum(fail) / len(fail)
        # something has failed, so we generate rand in range [0, sum(survive)]
        #if(r_ < 0.5):
        #    r_ = random.uniform(0, sum(survive))
        #else:
        #    r_ = -1

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

# around 400 works best
train(num_iteration=iter_)

# dyn save model based on argsparsed
saver.save(session, "models/fixedGuard/fguard_" + str(args.modelnumber) + ".ckpt")
