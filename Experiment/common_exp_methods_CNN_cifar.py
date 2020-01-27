import os
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import multi_gpu_model

class CustomModelCheckpoint(Callback):

    def __init__(self, model, path):

        super().__init__()

        # This is the argument that will be modify by fit_generator
        # self.model = model
        self.path = path

        # We set the model (non multi gpu) under an other name
        self.model_for_saving = model

    def on_epoch_end(self, epoch, logs=None):

        loss = logs['val_loss']
        # Here we save the original one
        print("\nSaving model to : {}".format(self.path.format(epoch=epoch, val_loss=loss)))
        self.model_for_saving.save_weights(self.path.format(epoch=epoch, val_loss=loss), overwrite=True)

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
    reliability_settings = [
        [1,1],
        [.98,.96],
        [.95,.90],
        [.85,.80],
    ]
    strides = (1,1)
    num_iterations = 1
    batch_size = 128
    epochs = 75
    progress_verbose = 1
    checkpoint_verbose = 1
    use_GCP = False
    alpha = .75
    input_shape = (32,32,3)
    classes = 10
    num_gpus = 1
    return num_iterations, classes, reliability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, use_GCP, alpha, input_shape, strides, num_gpus

def get_model_weights_CNN_cifar(model, parallel_model, model_name, load_for_inference, model_file, training_data, training_labels, val_data, val_labels, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus):
    if load_for_inference:
        model.load_weights(model_file)
    else:
        print(model_name)
        if num_gpus > 1:
            modelCheckPoint = CustomModelCheckpoint(model, model_file)
            parallel_model.fit_generator(
                train_datagen.flow(training_data,training_labels,batch_size = batch_size),
                epochs = epochs,
                validation_data = (val_data,val_labels), 
                steps_per_epoch = train_steps_per_epoch, 
                verbose = progress_verbose, 
                validation_steps = val_steps_per_epoch,
                callbacks = [modelCheckPoint])
            # load weights with the highest val accuracy
            model.load_weights(model_file)
            return model
        else:
            modelCheckPoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=checkpoint_verbose, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            model.fit_generator(
                train_datagen.flow(training_data,training_labels,batch_size = batch_size),
                epochs = epochs,
                validation_data = (val_data,val_labels), 
                steps_per_epoch = train_steps_per_epoch, 
                verbose = progress_verbose, 
                validation_steps = val_steps_per_epoch,
                callbacks = [modelCheckPoint])
            # load weights with the highest val accuracy
            model.load_weights(model_file)
            return model
