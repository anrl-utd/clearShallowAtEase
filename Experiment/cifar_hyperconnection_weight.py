
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
from Experiment.common_exp_methods import average, make_results_folder, make_output_dictionary_hyperconnection_weight, write_n_upload
from Experiment.common_exp_methods_CNN_cifar import init_data, init_common_experiment_params, get_model_weights_CNN_cifar
import numpy as np
import gc
from Experiment.common_exp_methods import make_no_information_flow_map
from Experiment.cnn_deepFogGuard import default_skip_hyperconnection_config

def define_and_train(iteration, model_name, load_for_inference, reliability_setting, weight_scheme, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus):
    K.set_learning_phase(1)
    if model_name == "DeepFogGuard Hyperconnection Weight":
        model_file = 'models/' + str(iteration) + "_" + str(reliability_setting) + "_" + str(weight_scheme) + 'cifar_hyperconnection_deepFogGuard.h5'
        model, parallel_model = define_deepFogGuard_CNN(classes=classes,input_shape = input_shape, alpha = alpha,reliability_setting=reliability_setting, hyperconnection_weights_scheme = weight_scheme, strides = strides, num_gpus=num_gpus)
    else: # model_name is "ResiliNet Hyperconnection Weight"
        model_file = 'models/' + str(iteration) + "_" + str(reliability_setting) + "_" + str(weight_scheme) + 'cifar_hyperconnection_ResiliNet.h5'
        model, parallel_model = define_ResiliNet_CNN(classes=classes,input_shape = input_shape, alpha = alpha,reliability_setting=reliability_setting, hyperconnection_weights_scheme = weight_scheme, strides = strides, num_gpus=num_gpus)
    get_model_weights_CNN_cifar(model, parallel_model, model_name, load_for_inference, model_file, training_data, training_labels, val_data, val_labels, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
    return model
           
#  hyperconnection weight experiment      
if __name__ == "__main__":
    accuracy = accuracy("CIFAR")
    calculateExpectedAccuracy = accuracy.calculateExpectedAccuracy
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data() 

    num_iterations, classes, reliability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, use_GCP, alpha, input_shape, strides, num_gpus = init_common_experiment_params()

    model_name = "ResiliNet Hyperconnection Weight"
    output, weight_schemes = make_output_dictionary_hyperconnection_weight(model_name, reliability_settings, num_iterations)
    
    no_information_flow_map = make_no_information_flow_map("CIFAR/Imagenet", default_skip_hyperconnection_config)

    load_for_inference = False
    train_steps_per_epoch = math.ceil(len(training_data) / batch_size)
    val_steps_per_epoch = math.ceil(len(val_data) / batch_size)
    
    make_results_folder()
    output_name = 'results/cifar_hyperconnection_weight_results.txt'
    output_list = []
    default_reliability_setting = [1,1]
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        for weight_scheme in weight_schemes:
            if weight_scheme == 2 or weight_scheme == 3: # if the weight scheme depends on reliability
                for reliability_setting in reliability_settings:
                    model = define_and_train(iteration, model_name, load_for_inference, reliability_setting, weight_scheme, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
                    output_list.append(str(reliability_setting) + str(weight_scheme) + '\n')
                    output[model_name][weight_scheme][str(reliability_setting)][iteration-1] = calculateExpectedAccuracy(model,no_information_flow_map,reliability_setting,output_list, training_labels= training_labels, test_data= test_data, test_labels= test_labels)
                    # clear session so that model will recycled back into memory
                    K.clear_session()
                    gc.collect()
                    del model
            else:
                model = define_and_train(iteration, model_name, load_for_inference, default_reliability_setting, weight_scheme, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
                for reliability_setting in reliability_settings:
                    output_list.append(str(reliability_setting) + str(weight_scheme) + '\n')
                    output[model_name][weight_scheme][str(reliability_setting)][iteration-1] = calculateExpectedAccuracy(model,no_information_flow_map,reliability_setting,output_list, training_labels= training_labels, test_data= test_data, test_labels= test_labels)
                # clear session so that model will recycled back into memory
                K.clear_session()
                gc.collect()
                del model
    
    for reliability_setting in reliability_settings:
        for weight_scheme in weight_schemes:
            output_list.append(str(reliability_setting) + str(weight_scheme) + '\n')
            acc = average(output[model_name][weight_scheme][str(reliability_setting)])
            output_list.append(str(reliability_setting) + str(weight_scheme) +  str(acc) + '\n')
            print(str(reliability_setting), weight_scheme, acc)

            std = np.std(output[model_name][weight_scheme][str(reliability_setting)],ddof=1)
            output_list.append(str(reliability_setting) + str(weight_scheme) +  str(std) + '\n')
            print(str(reliability_setting), weight_scheme, std)
    write_n_upload(output_name, output_list, use_GCP)
    print(output)