
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
import os 
from Experiment.cnn_deepFogGuard import define_deepFogGuard_CNN
from Experiment.FailureIteration import calculateExpectedAccuracy
from Experiment.common_exp_methods_CNN_cifar import init_data, init_common_experiment_params, get_model_weights_CNN_cifar
from Experiment.common_exp_methods import average, make_results_folder, convert_to_string, write_n_upload, make_results_folder
import numpy as np
import gc


def make_output_dictionary(survivability_settings, num_iterations):
    no_failure, normal, poor, hazardous = convert_to_string(survivability_settings)

    # convert hyperconnection configuration into strings to be used as keys for dictionary
    config = [0] * 5
    for i in range(0,4):
        config[i] = str(skip_hyperconnection_configurations[i])

    # dictionary to store all the results
    output = {
        "DeepFogGuard Hyperconnection Weight Sensitivity":
        {
            hazardous:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations
            },
            poor:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations
            },
            normal:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations
            },
            no_failure:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations
            },
        }
    }
    return output

def define_and_train(iteration, model_name, load_model, skip_hyperconnection_configuration, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch):
    model_file = 'models/' + str(iteration) + " " + str(skip_hyperconnection_configuration) + " " + 'cifar_skiphyperconnection_sensitivity.h5'
    model = define_deepFogGuard_CNN(classes=classes,input_shape = input_shape,alpha = alpha,skip_hyperconnection_config = skip_hyperconnection_configuration, strides = strides)
    
    get_model_weights_CNN_cifar(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch)
    return model

# deepFogGuard hyperconnection failure configuration ablation experiment
if __name__ == "__main__":
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data() 
    
    num_iterations, classes, survivability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, use_GCP, alpha, input_shape, strides = init_common_experiment_params()
    output = make_output_dictionary(survivability_settings, num_iterations)
    
    skip_hyperconnection_configurations = [
        [0,0],
        [1,0],
        [0,1],
        [1,1],
    ]
    num_iterations = 20
    load_model = False
    train_steps_per_epoch = math.ceil(len(training_data) / batch_size)
    val_steps_per_epoch = math.ceil(len(val_data) / batch_size)

    make_results_folder()
    output_name = 'results/cifar_skiphyperconnection_sensitivity_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            model = define_and_train(iteration, "DeepFogGuard Hyperconnection Weight Sensitivity", load_model, skip_hyperconnection_configuration, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch)
            for survivability_settings in survivability_settings:
                output_list.append(str(survivability_settings) + '\n')
                print(survivability_settings)
                output["DeepFogGuard Hyperconnection Weight Sensitivity"][str(survivability_settings)][str(skip_hyperconnection_configuration)][iteration-1] = calculateExpectedAccuracy(model, survivability_settings,output_list, training_labels, test_data, test_labels)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del model
    
    for survivability_settings in survivability_settings:
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            output_list.append(str(survivability_settings) + '\n')
            deepFogGuard_acc = average(output["DeepFogGuard Hyperconnection Weight Sensitivity"][str(survivability_settings)][str(skip_hyperconnection_configuration)])
            deepFogGuard_std = np.std(output["DeepFogGuard Hyperconnection Weight Sensitivity"][str(survivability_settings)][str(skip_hyperconnection_configuration)],ddof=1)
            output_list.append(str(survivability_settings) + str(skip_hyperconnection_configuration) + str(deepFogGuard_acc) + '\n')
            output_list.append(str(survivability_settings) + str(skip_hyperconnection_configuration) + str(deepFogGuard_std) + '\n')
            print(str(survivability_settings),deepFogGuard_acc)
            print(str(survivability_settings), deepFogGuard_std)
    write_n_upload(output_name, output_list, use_GCP)
    print(output)