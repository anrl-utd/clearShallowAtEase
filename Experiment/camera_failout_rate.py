from Experiment.mlp_ResiliNet_camera import define_ResiliNet_MLP
from Experiment.FailureIteration import calculateExpectedAccuracy
from Experiment.common_exp_methods_MLP_camera import init_data, init_common_experiment_params, get_model_weights_MLP_camera
from Experiment.common_exp_methods import average, convert_to_string, write_n_upload,  make_results_folder, make_output_dictionary_failout_rate, make_output_dictionary_failout_rate
import keras.backend as K
import gc
import os
from keras.callbacks import ModelCheckpoint
import numpy as np

def define_and_train(iteration, model_name, load_model, failout_survival_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose):
    model = define_ResiliNet_MLP(input_shape,num_classes,hidden_units,failout_survival_setting)
    model_file = 'models/' + str(iteration) + " " + str(failout_survival_setting) + 'camera_failout_rate.h5'
    get_model_weights_MLP_camera(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, verbose)
    return model

def multiply_hyperconnection_weights(dropout_like_failout, failout_survival_setting, model):
    if dropout_like_failout == True:
        nodes = ["edge_output_layer","fog2_output_layer","fog1_output_layer"]
        for i, node in enumerate(nodes):
            survival_rate = failout_survival_setting[i]
            # node failed
            layer_name = node
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            # make new weights for the connections
            new_weights = layer_weights[0] * survival_rate

            # make new weights for biases
            new_bias_weights = layer_weights[1] * survival_rate
            layer.set_weights([new_weights,new_bias_weights])
            
# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    use_GCP = False
    training_data,val_data, test_data, training_labels,val_labels,test_labels = init_data(use_GCP)

    num_iterations, num_vars, num_classes, survivability_settings, num_train_epochs, hidden_units, batch_size = init_common_experiment_params(training_data)
    load_model = False
    failout_survival_settings = [
        [.95,.95,.95],
        [.9,.9,.9],
        [.7,.7,.7],
        [.5,.5,.5],
    ]
    # file name with the experiments accuracy output
    output_name = "results/health_failout_rate.txt"
    verbose = 2
    # keep track of output so that output is in order
    output_list = []
    
    output = make_output_dictionary_failout_rate(failout_survival_settings, survivability_settings, num_iterations)
    dropout_like_failout = False
    make_results_folder()
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        output_list.append('ResiliNet' + '\n')  
        # variable failout rate                
        for survivability_setting in survivability_settings:
            ResiliNet_failout_rate_variable = define_and_train(iteration, "Variable Failout 1x", load_model, survivability_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose)
            multiply_hyperconnection_weights(dropout_like_failout, survivability_setting, ResiliNet_failout_rate_variable)
            output["Variable Failout 1x"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(ResiliNet_failout_rate_variable,survivability_setting,output_list,training_labels,test_data,test_labels)
        
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del ResiliNet_failout_rate_variable
        # fixed failout rate
        for failout_survival_setting in failout_survival_settings:
            ResiliNet_failout_rate_fixed = define_and_train(iteration, "Fixed Failout 1x", load_model, failout_survival_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose)
            multiply_hyperconnection_weights(dropout_like_failout, failout_survival_setting, ResiliNet_failout_rate_fixed)   
                
            for survivability_setting in survivability_settings:
                output_list.append(str(survivability_setting)+ '\n')
                print(survivability_setting)
                output[str(failout_survival_setting)][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(ResiliNet_failout_rate_fixed,survivability_setting,output_list,training_labels,test_data,test_labels)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del ResiliNet_failout_rate_fixed

    # calculate average accuracies for variable failout rate
    for survivability_setting in survivability_settings:
        ResiliNet_failout_rate_acc = average(output["Variable Failout 1x"][str(survivability_setting)])
        output_list.append(str(survivability_setting) + " Variable Failout 1x: " + str(ResiliNet_failout_rate_acc) + '\n')
        print(survivability_setting,"Variable Failout 1x:",ResiliNet_failout_rate_acc)  

        ResiliNet_failout_rate_std = np.std(output["Variable Failout 1x"][str(survivability_setting)],ddof=1)
        output_list.append(str(survivability_setting) + " Variable Failout 1x std: " + str(ResiliNet_failout_rate_std) + '\n')
        print(str(survivability_setting), " Variable Failout 1x std:",ResiliNet_failout_rate_std)
    # calculate average accuracies for fixed failout rate
    for failout_survival_setting in failout_survival_settings:
        print(failout_survival_setting)
        for survivability_setting in survivability_settings:
            ResiliNet_failout_rate_acc = average(output[str(failout_survival_setting)][str(survivability_setting)])
            output_list.append(str(failout_survival_setting) + str(survivability_setting) + " Fixed Failout: " + str(ResiliNet_failout_rate_acc) + '\n')
            print(failout_survival_setting,survivability_setting,"Fixed Failout:",ResiliNet_failout_rate_acc)  

            ResiliNet_failout_rate_std = np.std(output[str(failout_survival_setting)][str(survivability_setting)],ddof=1)
            output_list.append(str(survivability_setting) + " Fixed Failout std: " + str(ResiliNet_failout_rate_std) + '\n')
            print(str(survivability_setting), "Fixed Failout std:",ResiliNet_failout_rate_std)

    # write experiments output to file
    write_n_upload(output_name, output_list, use_GCP)
    print(output)

