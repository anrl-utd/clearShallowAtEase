
from Experiment.mlp_deepFogGuard_health import define_deepFogGuard_MLP
from Experiment.FailureIteration import calculateExpectedAccuracy
from Experiment.utility import average, get_model_weights_MLP
from Experiment.health_common_exp_methods import init_data, init_common_experiment_params
from Experiment.common_exp_methods import convert_to_string, write_n_upload, make_results_folder
import keras.backend as K
import gc
import os
from keras.callbacks import ModelCheckpoint

def make_output_dictionary(survivability_settings, num_iterations):
    no_failure, normal, poor, hazardous = convert_to_string(survivability_settings)

    # define weight schemes for hyperconnections
    one_weight_scheme = 1 # weighted by 1
    normalized_survivability_weight_scheme = 2 # normalized survivability
    survivability_weight_scheme = 3 # survivability
    random_weight_scheme = 4 # randomly weighted between 0 and 1
    random_weight_scheme2 = 5 # randomly weighted between 0 and 10
    fifty_weight_scheme = 6  # randomly weighted by .5

    weight_schemes = [
        one_weight_scheme,
        normalized_survivability_weight_scheme,
        survivability_weight_scheme,
        random_weight_scheme,
        random_weight_scheme2,
        fifty_weight_scheme,
    ]

    # dictionary to store all the results
    output = {
        "DeepFogGuard Hyperconnection Weight": 
        {
            one_weight_scheme:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            },
            normalized_survivability_weight_scheme:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            },
            survivability_weight_scheme:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            },
            random_weight_scheme:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            },
            random_weight_scheme2:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            },
            fifty_weight_scheme:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            }
        },
    }
    return output, weight_schemes

def define_and_train(iteration, model_name, load_model, weight_scheme, survivability_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose):
    model = define_deepFogGuard_MLP(num_vars,num_classes,hidden_units, survivability_setting, hyperconnection_weights_scheme = weight_scheme)
    model_file = str(iteration) + "_" + str(survivability_setting) + "_" + str(weight_scheme) + 'health_hyperconnection_fixed_random_weight.h5'
    get_model_weights_MLP(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, verbose)
    return model

# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    use_GCP = True
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data(use_GCP)

    num_iterations, num_vars, num_classes, survivability_settings, num_train_epochs, hidden_units, batch_size = init_common_experiment_params(training_data)
   
    load_model = False
    # file name with the experiments accuracy output
    output_name = "results/health_hyperconnection_fixed_random_weight.txt"
    verbose = 2
    model_name = "DeepFogGuard Hyperconnection Weight"
    hyperconnection_weightedbysurvivability_config = 2
    # keep track of output so that output is in order
    output_list = []
    
    output, weight_schemes = make_output_dictionary(survivability_settings, num_iterations)
        
    make_results_folder()
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        for survivability_setting in survivability_settings:
            # loop through all the weight schemes
            for weight_scheme in weight_schemes:
                # deepFogGuard hyperconnection weight 
                deepFogGuard_hyperconnection_weight = define_and_train(iteration, model_name, load_model, weight_scheme, survivability_setting, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose)
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
    write_n_upload(output_name, output_list, use_GCP)
    print(output)