

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from keras.models import Model
import keras.backend as K
import datetime
import os

from Experiment.mlp_deepFogGuardPlus_health import define_deepFogGuardPlus_MLP
from Experiment.mlp_deepFogGuard_health import define_deepFogGuard_MLP
from Experiment.mlp_Vanilla_health import define_vanilla_model_MLP
from Experiment.random_guess import model_guess
from keras.callbacks import ModelCheckpoint
import numpy as np

def fail_node(model,node_array):
    """fails node by making the specified node/nodes output 0
    ### Arguments
        model (Model): Keras model to have nodes failed
        node_array (list): bit list that corresponds to the node arrangement, 1 in the list represents to alive and 0 corresponds to failure 
    ### Returns
        return a boolean whether the model failed was a cnn or not
    """
    is_cnn = False
    # determines type of network by the first layer input shape
    first_layer = model.get_layer(index = 0)
    if len(first_layer.input_shape) == 4:
        # cnn input shape has 4 dimensions
        is_cnn = True
    # cnn failure
    if is_cnn:
        nodes = ["conv_pw_3","conv_pw_8"]
        for index,node in enumerate(node_array):
            # node failed
            if node == 0:
                layer_name = nodes[index]
                layer = model.get_layer(name=layer_name)
                layer_weights = layer.get_weights()
                # make new weights for the connections
                new_weights = np.zeros(layer_weights[0].shape)
                layer.set_weights([new_weights])
                print(layer_name, "was failed")
    # regular NN failure
    else:
        nodes = ["edge_output_layer","fog2_output_layer","fog1_output_layer"]
        for index,node in enumerate(node_array):
            # node failed
            if node == 0:
                layer_name = nodes[index]
                layer = model.get_layer(name=layer_name)
                layer_weights = layer.get_weights()
                # make new weights for the connections
                new_weights = np.zeros(layer_weights[0].shape)
                #new_weights[:] = np.nan # set weights to nan
                # make new weights for biases
                new_bias_weights = np.zeros(layer_weights[1].shape)
                layer.set_weights([new_weights,new_bias_weights])
                print(layer_name, "was failed")
    return is_cnn

def average(list):
    """function to return average of a list 
    ### Arguments
        list (list): list of numbers
    ### Returns
        return sum of list
    """
    if len(list) == 0:
        return 0
    else:
        return sum(list) / len(list)
        
def get_model_weights_MLP_health(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, verbose):
    if load_model:
        model.load_weights(model_file)
    else:
        print(model_name)
        modelCheckPoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        model.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [modelCheckPoint],validation_data=(val_data,val_labels))
        # load weights from epoch with the highest val acc
        model.load_weights(model_file)

def get_model_weights_MLP_camera(model, model_name, load_model, model_file, train_generator, val_generator, verbose):
    if load_model:
        model.load_weights(model_file)
    else:
        print(model_name)
        modelCheckPoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        model.fit_generator(
            generator = train_generator,
            steps_per_epoch = 10,
            epochs = 10,
            validation_data = val_generator,
            validation_steps = 10,
            class_weight = None,
            callbacks = [modelCheckPoint],
            verbose = verbose
            )
        # load weights from epoch with the highest val acc
        model.load_weights(model_file)

def get_model_weights_CNN(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch):
    if load_model:
        model.load_weights(model_file)
    else:
        # checkpoints to keep track of model with best validation accuracy 
        print(model_name)
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
<<<<<<< HEAD


def get_model_weights_cnn_imagenet(model, model_name, load_model, model_file, train_generator, val_generator, verbose):
    if load_model:
        model.load_weights(model_file)
    else:
        print(model_name)
        modelCheckPoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        num_train_examples = 1300000
        model.fit_generator(
            generator = train_generator,
            steps_per_epoch = num_train_examples / train_generator.batch_size,
            epochs = 75,
            class_weight = None,
            callbacks = [modelCheckPoint],
            verbose = verbose
            )
        # load weights from epoch with the highest val acc
        model.load_weights(model_file)
=======
        
>>>>>>> ae17332290b1d5b6f270a6e7996909ff333e9136
