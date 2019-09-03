import os
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing.image import load_img
def init_data(use_GCP, num_gpus, pc = 1):
    if pc == 1: # google cloud
        train_dir = "/home/yousefpour_ashkan/new_disk/train"
        test_dir = "/home/yousefpour_ashkan/val"
    elif pc == 2: # our AWS
        train_dir = "/home/yousefpour_ashkan/new_disk/train"
        test_dir = "/home/yousefpour_ashkan/val"
    elif pc == 3: # local
        train_dir = "/home/user1/externalDrive/ashkan-imagenet/train"
        test_dir = "/home/user1/externalDrive/ashkan-imagenet/val"
        num_gpus = 1
    elif pc == 4: # Guanhua AWS
        train_dir = "/home/ubuntu/imagenet/train"
        test_dir = "/home/ubuntu/imagenet/val"
        num_gpus = 1
    input_shape = (256,256)
    batch_size = 8
    datagen = ImageDataGenerator(
        rescale = 1./255
    )
    train_generator = datagen.flow_from_directory(
        directory = train_dir,
        target_size = input_shape,
        batch_size = batch_size * num_gpus,
        class_mode = "sparse"
    )
    test_generator = datagen.flow_from_directory(
        directory = test_dir,
        target_size = input_shape,
        batch_size = batch_size * num_gpus,
        class_mode = "sparse"
    )
    return train_generator, test_generator

def init_common_experiment_params():
    num_train_examples = 1300000
    num_test_examples = 50000
    input_shape = (256,256,3)
    alpha = .5
    num_iterations = 10
    # need to change this to be accurate
    survivability_settings = [
        [1,1],
        [.96,.98],
        [.90,.95],
        [.80,.85],
    ]
    num_classes = 1000
    epochs = 100
    num_gpus = 2
    strides = (2,2)
    return num_iterations, num_train_examples,num_test_examples, survivability_settings, input_shape, num_classes, alpha, epochs, num_gpus, strides
