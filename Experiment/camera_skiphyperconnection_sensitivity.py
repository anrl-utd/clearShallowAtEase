
from Experiment.mlp_deepFogGuard_camera import define_deepFogGuard_MLP
from Experiment.common_exp_methods_MLP_camera import init_data, init_common_experiment_params, get_model_weights_MLP_camera
from Experiment.FailureIteration import calculateExpectedAccuracy
from Experiment.common_exp_methods import average, convert_to_string, write_n_upload, make_results_folder
import keras.backend as K
import os
import gc 
from keras.callbacks import ModelCheckpoint
import numpy as np

def make_output_dictionary(survivability_settings, num_iterations):
    no_failure, normal, poor, hazardous = convert_to_string(survivability_settings)

    # convert hyperconnection configuration into strings to be used as keys for dictionary
    config = [0] * 9
    for i in range(0,8):
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

def define_and_train(iteration, model_name, load_model, default_survivability_setting, skip_hyperconnection_configuration, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose):
    model = define_deepFogGuard_MLP(input_shape,num_classes,hidden_units, default_survivability_setting,skip_hyperconnection_configuration)
    model_file = 'models/' + str(iteration) + " " + str(skip_hyperconnection_configuration) + " " + 'camera_skiphyperconnection_sensitivity.h5'
    get_model_weights_MLP_camera(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, verbose)
    return model


def calc_accuracy(iteration, model_name, model, survivability_setting, skip_hyperconnection_configuration, output_list,training_labels,test_data,test_labels):
    output_list.append(model_name + '\n')
    print(model_name)
    output[model_name][str(survivability_setting)][str(skip_hyperconnection_configuration)][iteration-1] = calculateExpectedAccuracy(model,survivability_setting,output_list,training_labels,test_data,test_labels)


# runs all hyperconnection configurations for both deepFogGuard survival configurations
if __name__ == "__main__":
    use_GCP = True
    training_data,val_data, test_data, training_labels,val_labels,test_labels = init_data(use_GCP)

    survivability_settings, input_shape, num_classes, hidden_units, batch_size, num_train_epochs, num_iterations = init_common_experiment_params()
    skip_hyperconnection_configurations = [
        # [e1,e2,e3,e4,f3,f4,f2]
        [1,1,1,1,0,0,0],
        [1,0,0,0,0,0,1],
        [1,0,0,0,1,0,0],
        [0,0,0,0,1,0,1],
        [0,0,0,0,0,0,1],
        [0,0,0,0,0,1,1],
        [1,1,1,1,0,0,1],
        [1,1,1,1,1,1,1]
    ]
    default_survivability_setting = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    num_iterations = 20
    load_model = False
    output_name = 'results/camera_skiphyperconnection_sensitivity.txt'
    
    verbose = 2
    # keep track of output so that output is in order
    output_list = []
    output = make_output_dictionary(survivability_settings, num_iterations)
    model_name = "DeepFogGuard Hyperconnection Weight Sensitivity"
    make_results_folder()
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
          
            deepFogGuard_weight_sesitivity = define_and_train(iteration, model_name, load_model, default_survivability_setting, skip_hyperconnection_configuration, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose)
            # test models
            for survivability_setting in survivability_settings:
                print(survivability_setting)
                output_list.append(str(survivability_setting) + '\n')
                calc_accuracy(iteration, model_name, deepFogGuard_weight_sesitivity, survivability_setting, skip_hyperconnection_configuration, output_list,training_labels,test_data,test_labels)
            # clear session to remove old graphs from memory so that subsequent training is not slower
            K.clear_session()
            gc.collect()
            del deepFogGuard_weight_sesitivity
    
    for survivability_setting in survivability_settings:
        output_list.append(str(survivability_setting) + '\n')
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            output_list.append(str(skip_hyperconnection_configuration) + '\n')
            deepFogGuard_acc = average(output["DeepFogGuard Hyperconnection Weight Sensitivity"][str(survivability_setting)][str(skip_hyperconnection_configuration)])
            deepFogGuard_std = np.std(output["DeepFogGuard Hyperconnection Weight Sensitivity"][str(survivability_setting)][str(skip_hyperconnection_configuration)],ddof=1)
            # write to output list
            output_list.append(str(survivability_setting) + " " + str(skip_hyperconnection_configuration) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
            print(str(survivability_setting),str(skip_hyperconnection_configuration),"deepFogGuard Accuracy:",deepFogGuard_acc)
            output_list.append(str(survivability_setting) + " " + str(skip_hyperconnection_configuration) + " deepFogGuard std: " + str(deepFogGuard_std) + '\n')
            print(str(survivability_setting),str(skip_hyperconnection_configuration),"deepFogGuard std:",deepFogGuard_std)
    
    write_n_upload(output_name, output_list, use_GCP)
    print(output)