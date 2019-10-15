
from Experiment.mlp_deepFogGuard_camera import define_deepFogGuard_MLP
from Experiment.mlp_ResiliNet_camera import define_ResiliNet_MLP
from Experiment.Accuracy import accuracy
from Experiment.common_exp_methods_MLP_camera import init_data, init_common_experiment_params, get_model_weights_MLP_camera
from Experiment.common_exp_methods import average, convert_to_string, write_n_upload, make_results_folder, make_output_dictionary_hyperconnection_weight
import keras.backend as K
import gc
import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from Experiment.common_exp_methods import make_no_information_flow_map
from Experiment.mlp_deepFogGuard_camera import default_skip_hyperconnection_config

def define_and_train(iteration, model_name, load_model, weight_scheme, reliability_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose):
    K.set_learning_phase(1)
    if model_name == "DeepFogGuard Hyperconnection Weight":
        model = define_deepFogGuard_MLP(input_shape,num_classes,hidden_units, reliability_setting=reliability_setting, hyperconnection_weights_scheme = weight_scheme)
        model_file = "models/" + str(iteration) + "_" + str(reliability_setting) + "_" + str(weight_scheme) + 'camera_hyperconnection_deepFogGuard.h5'
    else: # model_name is "ResiliNet Hyperconnection Weight"
        model = define_ResiliNet_MLP(input_shape,num_classes,hidden_units, reliability_setting=reliability_setting, hyperconnection_weights_scheme = weight_scheme)
        model_file = "models/" + str(iteration) + "_" + str(reliability_setting) + "_" + str(weight_scheme) + 'camera_hyperconnection_ResiliNet.h5'
    get_model_weights_MLP_camera(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, verbose)
    return model

# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    accuracy = accuracy("Camera")
    calculateExpectedAccuracy = accuracy.calculateExpectedAccuracy
    use_GCP = False
    training_data,val_data, test_data, training_labels,val_labels,test_labels = init_data(use_GCP)

    reliability_settings, input_shape, num_classes, hidden_units, batch_size, num_train_epochs, num_iterations = init_common_experiment_params()
    no_information_flow_map = make_no_information_flow_map("Camera", default_skip_hyperconnection_config)
   
    load_model = False
    # file name with the experiments accuracy output
    output_name = "results/camera_hyperconnection_weight.txt"
    verbose = 2
    model_name = "DeepFogGuard Hyperconnection Weight"
    hyperconnection_weightedbyreliability_config = 2
    # keep track of output so that output is in order
    output_list = []
    
    output, weight_schemes = make_output_dictionary_hyperconnection_weight(reliability_settings, num_iterations)
    default_reliability_setting = [1,1,1,1,1,1,1,1] 
    make_results_folder()
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        # loop through all the weight schemes
        for weight_scheme in weight_schemes:
            if weight_scheme == 2 or weight_scheme == 3: # if the weight scheme depends on reliability
                for reliability_setting in reliability_settings:
                    deepFogGuard_hyperconnection_weight = define_and_train(iteration, model_name, load_model, weight_scheme, reliability_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose)
                    output[model_name][weight_scheme][str(reliability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuard_hyperconnection_weight, no_information_flow_map,reliability_setting,output_list,training_labels= training_labels, test_data= test_data, test_labels= test_labels)
                    # clear session so that model will recycled back into memory
                    K.clear_session()
                    gc.collect()
                    del deepFogGuard_hyperconnection_weight
            else:
                deepFogGuard_hyperconnection_weight = define_and_train(iteration, model_name, load_model, weight_scheme, default_reliability_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose)
                for reliability_setting in reliability_settings:
                    output[model_name][weight_scheme][str(reliability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuard_hyperconnection_weight, no_information_flow_map,reliability_setting,output_list,training_labels= training_labels, test_data= test_data, test_labels= test_labels)
                # clear session so that model will recycled back into memory
                K.clear_session()
                gc.collect()
                del deepFogGuard_hyperconnection_weight
                
   # calculate average accuracies 
    for reliability_setting in reliability_settings:
        for weight_scheme in weight_schemes:
            deepFogGuard_hyperconnection_weight_acc = average(output[model_name][weight_scheme][str(reliability_setting)])
            output_list.append(str(reliability_setting) + str(weight_scheme) + " "+ model_name +": " + str(deepFogGuard_hyperconnection_weight_acc) + '\n')
            print(str(reliability_setting),weight_scheme,model_name,":",deepFogGuard_hyperconnection_weight_acc)

            deepFogGuard_hyperconnection_weight_std = np.std(output[model_name][weight_scheme][str(reliability_setting)],ddof=1)
            output_list.append(str(reliability_setting) + str(weight_scheme) + " "+ model_name +" std: " + str(deepFogGuard_hyperconnection_weight_std) + '\n')
            print(str(reliability_setting),weight_scheme,model_name,"std:",deepFogGuard_hyperconnection_weight_std)
    write_n_upload(output_name, output_list, use_GCP)
    print(output)