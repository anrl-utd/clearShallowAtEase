
from Experiment.mlp_deepFogGuard_camera import define_deepFogGuard_MLP
from Experiment.mlp_ResiliNet_camera import define_ResiliNet_MLP
from Experiment.common_exp_methods_MLP_camera import init_data, init_common_experiment_params, get_model_weights_MLP_camera
from Experiment.Accuracy import accuracy
from Experiment.common_exp_methods import average, convert_to_string, write_n_upload, make_results_folder
import keras.backend as K
import os
import gc 
from keras.callbacks import ModelCheckpoint
import numpy as np
from Experiment.common_exp_methods import make_no_information_flow_map
from Experiment.mlp_deepFogGuard_camera import default_skip_hyperconnection_config

def make_output_dictionary(model_name, reliability_settings, num_iterations, skip_hyperconnection_configurations):
    no_failure, normal, poor, hazardous = convert_to_string(reliability_settings)

    # convert hyperconnection configuration into strings to be used as keys for dictionary
    config = [0] * 9
    for i in range(0,8):
        config[i] = str(skip_hyperconnection_configurations[i])

    # dictionary to store all the results
    output = {
        model_name:
        {
            hazardous:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations,
                config[4]:[0] * num_iterations,
                config[5]:[0] * num_iterations,
                config[6]:[0] * num_iterations,
                config[7]:[0] * num_iterations
            },
            poor:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations,
                config[4]:[0] * num_iterations,
                config[5]:[0] * num_iterations,
                config[6]:[0] * num_iterations,
                config[7]:[0] * num_iterations
            },
            normal:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations,
                config[4]:[0] * num_iterations,
                config[5]:[0] * num_iterations,
                config[6]:[0] * num_iterations,
                config[7]:[0] * num_iterations
            },
            no_failure:
            {
                config[0]:[0] * num_iterations,
                config[1]:[0] * num_iterations,
                config[2]:[0] * num_iterations,
                config[3]:[0] * num_iterations,
                config[4]:[0] * num_iterations,
                config[5]:[0] * num_iterations,
                config[6]:[0] * num_iterations,
                config[7]:[0] * num_iterations
            },
        }
    }
    return output

def define_and_train(iteration, model_name, load_for_inference, reliability_setting, skip_hyperconnection_configuration, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose):
    K.set_learning_phase(1)
    if model_name == "DeepFogGuard Hyperconnection Weight Sensitivity":
        model = define_deepFogGuard_MLP(input_shape,num_classes,hidden_units, reliability_setting=reliability_setting,skip_hyperconnection_config=skip_hyperconnection_configuration)
        model_file = 'models/' + str(iteration) + " " + str(skip_hyperconnection_configuration) + " " + 'camera_skiphyperconnection_sensitivity_deepFogGuard.h5'
    else: # model_name is "ResiliNet Hyperconnection Weight Sensitivity"
        model = define_ResiliNet_MLP(input_shape,num_classes,hidden_units, reliability_setting=reliability_setting,skip_hyperconnection_config=skip_hyperconnection_configuration)
        model_file = 'models/' + str(iteration) + " " + str(skip_hyperconnection_configuration) + " " + 'camera_skiphyperconnection_sensitivity_ResiliNet.h5'
    get_model_weights_MLP_camera(model, model_name, load_for_inference, model_file, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, verbose)
    return model

def calc_accuracy(iteration, model_name, model, no_information_flow_map, reliability_setting, skip_hyperconnection_configuration, output_list,training_labels,test_data,test_labels):
    output_list.append(model_name + str(skip_hyperconnection_configuration)+ '\n')
    print(model_name+ str(skip_hyperconnection_configuration))
    output[model_name][str(reliability_setting)][str(skip_hyperconnection_configuration)][iteration-1] = calculateExpectedAccuracy(model, no_information_flow_map,reliability_setting,output_list,training_labels= training_labels, test_data= test_data, test_labels= test_labels)


if __name__ == "__main__":
    accuracy = accuracy("Camera")
    calculateExpectedAccuracy = accuracy.calculateExpectedAccuracy
    use_GCP = False
    training_data,val_data, test_data, training_labels,val_labels,test_labels = init_data(use_GCP)

    reliability_settings, input_shape, num_classes, hidden_units, batch_size, num_train_epochs, num_iterations = init_common_experiment_params()
    skip_hyperconnection_configurations = [
        # [f2,f3,f4,e1,e2,e3,e4]
        [1,1,1,1,0,0,0],
        [1,0,0,0,0,0,1],
        [1,0,0,0,1,0,0],
        [0,0,0,0,1,0,1],
        [0,0,0,0,0,0,1],
        [0,0,0,0,0,1,1],
        [1,1,1,1,0,0,1],
        [1,1,1,1,1,1,1]
    ]
    no_information_flow_map = {}
    for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
        no_information_flow_map[tuple(skip_hyperconnection_configuration)] = make_no_information_flow_map("Camera", skip_hyperconnection_configuration)
    
    default_reliability_setting = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    load_for_inference = False
    output_name = 'results/camera_skiphyperconnection_sensitivity.txt'
    
    verbose = 2
    # keep track of output so that output is in order
    output_list = []
    model_name = "ResiliNet Hyperconnection Weight Sensitivity"
    output = make_output_dictionary(model_name, reliability_settings, num_iterations, skip_hyperconnection_configurations)
    
    make_results_folder()
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            
            weight_sesitivity = define_and_train(iteration, model_name, load_for_inference, default_reliability_setting, skip_hyperconnection_configuration, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose)
            # test models
            for reliability_setting in reliability_settings:
                print(reliability_setting)
                output_list.append(str(reliability_setting) + '\n')
                calc_accuracy(iteration, model_name, weight_sesitivity, no_information_flow_map[tuple(skip_hyperconnection_configuration)], reliability_setting, skip_hyperconnection_configuration, output_list,training_labels,test_data,test_labels)
            K.clear_session()
            gc.collect()
            del weight_sesitivity
    
    for reliability_setting in reliability_settings:
        output_list.append(str(reliability_setting) + '\n')
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            output_list.append(str(skip_hyperconnection_configuration) + '\n')
            acc = average(output[model_name][str(reliability_setting)][str(skip_hyperconnection_configuration)])
            std = np.std(output[model_name][str(reliability_setting)][str(skip_hyperconnection_configuration)],ddof=1)
            # write to output list
            output_list.append(str(reliability_setting) + " " + str(skip_hyperconnection_configuration) + " Accuracy: " + str(acc) + '\n')
            print(str(reliability_setting),str(skip_hyperconnection_configuration),"Accuracy:",acc)
            output_list.append(str(reliability_setting) + " " + str(skip_hyperconnection_configuration) + " std: " + str(std) + '\n')
            print(str(reliability_setting),str(skip_hyperconnection_configuration),"std:",std)
    
    write_n_upload(output_name, output_list, use_GCP)
    print(output)