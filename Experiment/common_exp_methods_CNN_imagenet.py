import os
from keras.preprocessing.image import ImageDataGenerator 
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint

def init_data(use_GCP, num_gpus, pc = 4):
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
    input_shape = (160,160)
    batch_size = 1024
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        # preprocessing_function=preprocess_input
    )
    test_datagen = ImageDataGenerator(
        rescale = 1./255
    )
    train_generator = train_datagen.flow_from_directory(
        directory = train_dir,
        target_size = input_shape,
        batch_size = batch_size * num_gpus,
        class_mode = "sparse",
        shuffle=True,
        seed=42
    )
    test_generator = test_datagen.flow_from_directory(
        # shuffle = False,
        directory = test_dir,
        target_size = input_shape,
        batch_size = batch_size * num_gpus,
        class_mode = "sparse",
        seed=42
    )
    return train_generator, test_generator

def init_common_experiment_params():
    num_train_examples = 1300000
    num_test_examples = 50000
    input_shape = (160,160,3)
    alpha = .75
    num_iterations = 1
    # need to change this to be accurate
    reliability_settings = [
        [1,1],
        [.98,.96],
        [.95,.90],
        [.85,.80],
    ]
    num_classes = 1000
    epochs = 10
    num_gpus = 4
    num_workers = 32
    strides = (2,2)
    return num_iterations, num_train_examples,num_test_examples, reliability_settings, input_shape, num_classes, alpha, epochs, num_gpus, strides, num_workers

def get_model_weights_CNN_imagenet(model, parallel_model, model_name, load_for_inference, continue_training, model_file, train_generator, val_generator, num_train_examples, epochs, num_gpus, num_workers):
    if load_for_inference:
        parallel_model.load_weights(model_file)
        model = parallel_model.layers[-2]
        return model
    else:
        print(model_name)
        verbose = 1
        if num_gpus > 1:
            if continue_training:
                parallel_model.load_weights(model_file)
                model = parallel_model.layers[-2]
            modelCheckPoint = ModelCheckpoint(model_file, save_weights_only=True)
            parallel_model.fit_generator(
                generator = train_generator,
                steps_per_epoch = num_train_examples / train_generator.batch_size,
                epochs = epochs,
                workers = num_workers,
                class_weight = None,
                verbose = verbose,
                callbacks = [modelCheckPoint]
                )
            # save the weights
            parallel_model.save_weights(model_file)
            return model
        else:
            model.fit_generator(
                generator = train_generator,
                steps_per_epoch = num_train_examples / train_generator.batch_size,
                epochs = epochs,
                workers = num_workers,
                class_weight = None,
                verbose = verbose
                )
             # save the weights
            model.save_weights(model_file)
            return model