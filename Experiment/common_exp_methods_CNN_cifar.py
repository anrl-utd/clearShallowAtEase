import os
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np
from sklearn.model_selection import train_test_split

def init_data():
    # get cifar10 data 
    (training_data, training_labels), (test_data, test_labels) = cifar10.load_data()
    # normalize input
    training_data = training_data / 255
    test_data = test_data / 255
    # Concatenate train and test images
    data = np.concatenate((training_data,test_data))
    labels = np.concatenate((training_labels,test_labels))

    # split data in to train, validation, and holdout set (80/10/10)
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .20, shuffle = True)
    val_data, test_data, val_labels, test_labels = train_test_split(test_data,test_labels,random_state = 42, test_size = .50, shuffle = True)
    return  training_data, test_data, training_labels, test_labels, val_data, val_labels

def init_common_experiment_params():
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    survivability_settings = [
        [1,1],
        [.96,.98],
        [.90,.95],
        [.80,.85],
    ]
    strides = (1,1)
    num_iterations = 10
    batch_size = 128
    epochs = 75
    progress_verbose = 2
    checkpoint_verbose = 1
    use_GCP = True
    alpha = .5
    input_shape = (32,32,3)
    classes = 10
    return num_iterations, classes, survivability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, use_GCP, alpha, input_shape, strides