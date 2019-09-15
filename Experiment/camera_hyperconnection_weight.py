
from Experiment.mlp_deepFogGuard_camera import define_deepFogGuard_MLP
from Experiment.Accuracy import calculateExpectedAccuracy
from Experiment.common_exp_methods_MLP_camera import init_data, init_common_experiment_params, get_model_weights_MLP_camera
from Experiment.common_exp_methods import average, convert_to_string, write_n_upload, make_results_folder, make_output_dictionary_hyperconnection_weight
import keras.backend as K
import gc
import os
import numpy as np
from keras.callbacks import ModelCheckpoint



def define_and_train(iteration, model_name, load_model, weight_scheme, survivability_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose):
    model = define_deepFogGuard_MLP(input_shape,num_classes,hidden_units, survivability_setting, hyperconnection_weights_scheme = weight_scheme)
    model_file = "models/" + str(iteration) + "_" + str(survivability_setting) + "_" + str(weight_scheme) + 'camera_hyperconnection.h5'
    get_model_weights_MLP_camera(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, verbose)
    return model

# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    use_GCP = True
    training_data,val_data, test_data, training_labels,val_labels,test_labels = init_data(use_GCP)

    survivability_settings, input_shape, num_classes, hidden_units, batch_size, num_train_epochs, num_iterations = init_common_experiment_params()
   
    load_model = False
    # file name with the experiments accuracy output
    output_name = "results/camera_hyperconnection_fixed_random_weight.txt"
    verbose = 2
    model_name = "DeepFogGuard Hyperconnection Weight"
    hyperconnection_weightedbysurvivability_config = 2
    # keep track of output so that output is in order
    output_list = []
    
    output, weight_schemes = make_output_dictionary_hyperconnection_weight(survivability_settings, num_iterations)
        
    make_results_folder()
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        for survivability_setting in survivability_settings:
            # loop through all the weight schemes
            for weight_scheme in weight_schemes:
                # deepFogGuard hyperconnection weight 
                deepFogGuard_hyperconnection_weight = define_and_train(iteration, model_name, load_model, weight_scheme, survivability_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, input_shape, num_classes, hidden_units, verbose)
                output[model_name][weight_scheme][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuard_hyperconnection_weight,survivability_setting,output_list,training_labels,test_data,test_labels)
                # clear session so that model will recycled back into memory
                K.clear_session()
                gc.collect()
                del deepFogGuard_hyperconnection_weight
   # calculate average accuracies 
    for survivability_setting in survivability_settings:
        for weight_scheme in weight_schemes:
            deepFogGuard_hyperconnection_weight_acc = average(output[model_name][weight_scheme][str(survivability_setting)])
            output_list.append(str(survivability_setting) + str(weight_scheme) + " "+ model_name +": " + str(deepFogGuard_hyperconnection_weight_acc) + '\n')
            print(str(survivability_setting),weight_scheme,model_name,":",deepFogGuard_hyperconnection_weight_acc)

            deepFogGuard_hyperconnection_weight_std = np.std(output[model_name][weight_scheme][str(survivability_setting)],ddof=1)
            output_list.append(str(survivability_setting) + str(weight_scheme) + " "+ model_name +" std: " + str(deepFogGuard_hyperconnection_weight_std) + '\n')
            print(str(survivability_setting),weight_scheme,model_name,"std:",deepFogGuard_hyperconnection_weight_std)
    write_n_upload(output_name, output_list, use_GCP)
    print(output)