
from Experiment.mlp_deepFogGuard_health import define_deepFogGuard_MLP
from Experiment.Accuracy import calculateExpectedAccuracy
from Experiment.common_exp_methods_MLP_health import init_data, init_common_experiment_params, get_model_weights_MLP_health
from Experiment.common_exp_methods import average, convert_to_string, write_n_upload, make_results_folder, make_output_dictionary_hyperconnection_weight
import keras.backend as K
import gc
import os
from keras.callbacks import ModelCheckpoint
import numpy as np



def define_and_train(iteration, model_name, load_model, weight_scheme, reliability_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose):
    K.set_learning_phase(1)
    model = define_deepFogGuard_MLP(num_vars,num_classes,hidden_units, reliability_setting, hyperconnection_weights_scheme = weight_scheme)
    model_file = 'models/' + str(iteration) + "_" + str(reliability_setting) + "_" + str(weight_scheme) + 'health_hyperconnection.h5'
    get_model_weights_MLP_health(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, verbose)
    return model

# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    use_GCP = False
    training_data, val_data, test_data, training_labels, val_labels, test_labels = init_data(use_GCP)

    num_iterations, num_vars, num_classes, reliability_settings, num_train_epochs, hidden_units, batch_size = init_common_experiment_params(training_data)
   
    load_model = False
    # file name with the experiments accuracy output
    output_name = "results/health_hyperconnection_fixed_random_weight.txt"
    verbose = 2
    model_name = "DeepFogGuard Hyperconnection Weight"
    hyperconnection_weightedbyreliability_config = 2
    # keep track of output so that output is in order
    output_list = []
    
    output, weight_schemes = make_output_dictionary_hyperconnection_weight(reliability_settings, num_iterations)
        
    make_results_folder()
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        for reliability_setting in reliability_settings:
            # loop through all the weight schemes
            for weight_scheme in weight_schemes:
                # deepFogGuard hyperconnection weight 
                deepFogGuard_hyperconnection_weight = define_and_train(iteration, model_name, load_model, weight_scheme, reliability_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose)
                output[model_name][weight_scheme][str(reliability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuard_hyperconnection_weight,reliability_setting,output_list,training_labels= training_labels, test_data= test_data, test_labels= test_labels)
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