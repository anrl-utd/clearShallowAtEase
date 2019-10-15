
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
import os 
from Experiment.cnn_deepFogGuard import define_deepFogGuard_CNN
from Experiment.cnn_ResiliNet import define_ResiliNet_CNN
from Experiment.Accuracy import accuracy
from Experiment.common_exp_methods_CNN_cifar import init_data, init_common_experiment_params, get_model_weights_CNN_cifar
from Experiment.common_exp_methods import average, make_results_folder, convert_to_string, write_n_upload, make_results_folder
import numpy as np
import gc
from Experiment.common_exp_methods import make_no_information_flow_map
from Experiment.cnn_deepFogGuard import default_skip_hyperconnection_config

def make_output_dictionary(reliability_settings, num_iterations, skip_hyperconnection_configurations):
    no_failure, normal, poor, hazardous = convert_to_string(reliability_settings)

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

def define_and_train(iteration, model_name, load_model, reliability_setting, skip_hyperconnection_configuration, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus):
    K.set_learning_phase(1)
    if model_name == "DeepFogGuard Hyperconnection Weight Sensitivity":
        model_file = 'models/' + str(iteration) + " " + str(skip_hyperconnection_configuration) + " " + 'cifar_skiphyperconnection_sensitivity_deepFogGuard.h5'
        model, parallel_model = define_deepFogGuard_CNN(classes=classes,input_shape = input_shape,alpha = alpha, reliability_setting=reliability_setting, skip_hyperconnection_config = skip_hyperconnection_configuration, strides = strides, num_gpus=num_gpus)
    else: # model_name is "ResiliNet Hyperconnection Weight Sensitivity"
        model_file = 'models/' + str(iteration) + " " + str(skip_hyperconnection_configuration) + " " + 'cifar_skiphyperconnection_sensitivity_ResiliNet.h5'
        model, parallel_model = define_ResiliNet_CNN(classes=classes,input_shape = input_shape,alpha = alpha, reliability_setting=reliability_setting, skip_hyperconnection_config = skip_hyperconnection_configuration, strides = strides, num_gpus=num_gpus)
    get_model_weights_CNN_cifar(model, parallel_model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
    return model

# deepFogGuard hyperconnection failure configuration ablation experiment
if __name__ == "__main__":
    accuracy = accuracy("CIFAR")
    calculateExpectedAccuracy = accuracy.calculateExpectedAccuracy
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data() 
    
    num_iterations, classes, reliability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, use_GCP, alpha, input_shape, strides, num_gpus = init_common_experiment_params()
    skip_hyperconnection_configurations = [
        # [e1,IoT]
        [0,0],
        [1,0],
        [0,1],
        [1,1],
    ]
    default_reliability_setting = [1.0,1.0,1.0]
    output = make_output_dictionary(reliability_settings, num_iterations, skip_hyperconnection_configurations)
    
    no_information_flow_map = {}
    for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
        no_information_flow_map[tuple(skip_hyperconnection_configuration)] = make_no_information_flow_map("CIFAR/Imagenet", skip_hyperconnection_configuration)
    
    load_model = False
    train_steps_per_epoch = math.ceil(len(training_data) / batch_size)
    val_steps_per_epoch = math.ceil(len(val_data) / batch_size)

    make_results_folder()
    output_name = 'results/cifar_skiphyperconnection_sensitivity_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            
            model = define_and_train(iteration, "DeepFogGuard Hyperconnection Weight Sensitivity", load_model, default_reliability_setting, skip_hyperconnection_configuration, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
            for reliability_setting in reliability_settings:
                output_list.append(str(reliability_setting) + '\n')
                print(reliability_setting)
                output["DeepFogGuard Hyperconnection Weight Sensitivity"][str(reliability_setting)][str(skip_hyperconnection_configuration)][iteration-1] = calculateExpectedAccuracy(model, no_information_flow_map[tuple(skip_hyperconnection_configuration)],reliability_setting,output_list, training_labels= training_labels, test_data= test_data, test_labels= test_labels)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del model
    
    for reliability_setting in reliability_settings:
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            output_list.append(str(reliability_setting) + '\n')
            deepFogGuard_acc = average(output["DeepFogGuard Hyperconnection Weight Sensitivity"][str(reliability_setting)][str(skip_hyperconnection_configuration)])
            deepFogGuard_std = np.std(output["DeepFogGuard Hyperconnection Weight Sensitivity"][str(reliability_setting)][str(skip_hyperconnection_configuration)],ddof=1)
            output_list.append(str(reliability_setting) + str(skip_hyperconnection_configuration) + str(deepFogGuard_acc) + '\n')
            output_list.append(str(reliability_setting) + str(skip_hyperconnection_configuration) + str(deepFogGuard_std) + '\n')
            print(str(reliability_setting),deepFogGuard_acc)
            print(str(reliability_setting), deepFogGuard_std)
    write_n_upload(output_name, output_list, use_GCP)
    print(output)