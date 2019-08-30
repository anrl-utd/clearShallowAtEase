
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
from Experiment.cifar_common_exp_methods import init_data, init_common_experiment_params
import numpy as np
from Experiment.utility import average
import datetime
import gc
from sklearn.model_selection import train_test_split


# deepFogGuard hyperconnection weight experiment      
if __name__ == "__main__":
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data() 

    num_iterations, classes, survivability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, use_GCP, alpha, input_shape = init_common_experiment_params()

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
                model_name = "cifar_hyperconnection_weight_results_" + str(survivability_setting) + str(weight_scheme) + str(iteration) + ".h5"
                model = define_deepFogGuard_CNN(weights = weights,classes=classes,input_shape = input_shape, alpha = alpha,hyperconnection_weights=survivability_setting, hyperconnection_weights_scheme = weight_scheme)
                modelCheckPoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=checkpoint_verbose, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                model.fit_generator(train_datagen.flow(training_data,training_labels,batch_size = batch_size),
                epochs = epochs,
                validation_data = (val_data,val_labels), 
                steps_per_epoch = train_steps_per_epoch, 
                verbose = progress_verbose, 
                validation_steps = val_steps_per_epoch,
                callbacks = [modelCheckPoint])
                #load weights with the highest validaton acc
                model.load_weights(model_name)
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