
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
import os 
from Experiment.cnn_deepFogGuard import define_deepFogGuard_CNN
from Experiment.FailureIteration import calculateExpectedAccuracy
from Experiment.common_exp_methods import make_results_folder, make_output_dictionary_hyperconnection_weight, write_n_upload
from Experiment.common_exp_methods_CNN_cifar import init_data, init_common_experiment_params
from Experiment.utility import get_model_weights_CNN
import numpy as np
from Experiment.utility import average
import gc


def define_and_train(iteration, model_name, load_model, survivability_setting, weight_scheme, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch):
    model_file = "cifar_hyperconnection_weight_results_" + str(survivability_setting) + str(weight_scheme) + str(iteration) + ".h5"
    model = define_deepFogGuard_CNN(classes=classes,input_shape = input_shape, alpha = alpha,survivability_setting=survivability_setting, hyperconnection_weights_scheme = weight_scheme, strides = strides)
    get_model_weights_CNN(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch)
    return model
           

# deepFogGuard hyperconnection weight experiment      
if __name__ == "__main__":
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data() 

    num_iterations, classes, survivability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, use_GCP, alpha, input_shape, strides = init_common_experiment_params()

    output, weight_schemes = make_output_dictionary_hyperconnection_weight(survivability_settings, num_iterations)
    
    weights = None

    load_model = False
    train_steps_per_epoch = math.ceil(len(training_data) / batch_size)
    val_steps_per_epoch = math.ceil(len(val_data) / batch_size)
    
    make_results_folder()
    output_name = 'results/cifar_hyperconnection_weight_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        for survivability_setting in survivability_settings:
            for weight_scheme in weight_schemes:
                model = define_and_train(iteration, "DeepFogGuard Hyperconnection Weight", load_model, survivability_setting, weight_scheme, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch)
                
                output_list.append(str(survivability_setting) + str(weight_scheme) + '\n')
                print(survivability_setting,weight_scheme)
                output["DeepFogGuard Hyperconnection Weight"][weight_scheme][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(model,survivability_setting,output_list, training_labels, test_data, test_labels)
                
                # clear session so that model will recycled back into memory
                K.clear_session()
                gc.collect()
                del model
    
    for survivability_setting in survivability_settings:
        for weight_scheme in weight_schemes:
            output_list.append(str(survivability_setting) + str(weight_scheme) + '\n')
            deepFogGuard_acc = average(output["DeepFogGuard Hyperconnection Weight"][weight_scheme][str(survivability_setting)])
            output_list.append(str(survivability_setting) + str(weight_scheme) +  str(deepFogGuard_acc) + '\n')
            print(str(survivability_setting), weight_scheme, deepFogGuard_acc)
    write_n_upload(output_name, output_list, use_GCP)
    print(output)