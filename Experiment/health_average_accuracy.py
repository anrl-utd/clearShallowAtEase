
from Experiment.mlp_ResiliNet_health import define_ResiliNet_MLP
from Experiment.mlp_deepFogGuard_health import define_deepFogGuard_MLP
from Experiment.mlp_Vanilla_health import define_vanilla_model_MLP
from Experiment.Accuracy import calculateExpectedAccuracy
from Experiment.common_exp_methods_MLP_health import init_data, init_common_experiment_params, get_model_weights_MLP_health
from Experiment.common_exp_methods import average, convert_to_string, make_output_dictionary_average_accuracy, write_n_upload, make_results_folder
import keras.backend as K
import datetime
import gc
import os
import numpy as np

def define_and_train(iteration, model_name, load_model, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose):
    K.set_learning_phase(1)
    # ResiliNet
    if model_name == "ResiliNet":
        model = define_ResiliNet_MLP(num_vars,num_classes,hidden_units)
        model_file = 'models/' + "Health" + str(iteration) + 'average_accuracy_ResiliNet.h5'
    # deepFogGuard
    if model_name == "deepFogGuard":
        model = define_deepFogGuard_MLP(num_vars, num_classes, hidden_units)
        model_file = 'models/' + "Health" + str(iteration) + 'average_accuracy_deepFogGuard.h5'
    # Vanilla model
    if model_name == "Vanilla":
        model = define_vanilla_model_MLP(num_vars,num_classes,hidden_units)
        model_file = 'models/' + "Health" + str(iteration) + 'average_accuracy_vanilla.h5'
    
    get_model_weights_MLP_health(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, verbose)
    return model

def calc_accuracy(iteration, model_name, model, reliability_setting, output_list,training_labels,test_data,test_labels):
    output_list.append(model_name + "\n")
    print(model_name)
    output[model_name][str(reliability_setting)][iteration-1] = calculateExpectedAccuracy(model,reliability_setting,output_list,training_labels= training_labels, test_data= test_data, test_labels= test_labels)


# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    use_GCP = False
    training_data, val_data, test_data, training_labels, val_labels, test_labels = init_data(use_GCP) 
    
    num_iterations, num_vars, num_classes, reliability_settings, num_train_epochs, hidden_units, batch_size = init_common_experiment_params(training_data)

    load_model = False

    # file name with the experiments accuracy output
    output_name = "results/health_average_accuracy.txt"
    
    verbose = 2
    
    # keep track of output so that output is in order
    output_list = []
    
    output = make_output_dictionary_average_accuracy(reliability_settings, num_iterations)

    make_results_folder()
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        ResiliNet = define_and_train(iteration, "ResiliNet", load_model, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose)
        deepFogGuard = define_and_train(iteration, "deepFogGuard", load_model, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose)
        Vanilla = define_and_train(iteration, "Vanilla", load_model, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose)
 
        # test models
        for reliability_setting in reliability_settings:
            calc_accuracy(iteration, "ResiliNet", ResiliNet, reliability_setting, output_list,training_labels,test_data,test_labels)
            calc_accuracy(iteration, "deepFogGuard", deepFogGuard, reliability_setting, output_list,training_labels,test_data,test_labels)
            calc_accuracy(iteration, "Vanilla", Vanilla, reliability_setting, output_list,training_labels,test_data,test_labels)
            
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del deepFogGuard
        del ResiliNet
        del Vanilla
   # calculate average accuracies from all expected accuracies
    for reliability_setting in reliability_settings:
        ResiliNet_acc = average(output["ResiliNet"][str(reliability_setting)])
        deepFogGuard_acc = average(output["deepFogGuard"][str(reliability_setting)])
        Vanilla_acc = average(output["Vanilla"][str(reliability_setting)])

        output_list.append(str(reliability_setting) + " ResiliNet Accuracy: " + str(ResiliNet_acc) + '\n')
        output_list.append(str(reliability_setting) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
        output_list.append(str(reliability_setting) + " Vanilla Accuracy: " + str(Vanilla_acc) + '\n')

        print(str(reliability_setting),"ResiliNet Accuracy:",ResiliNet_acc)
        print(str(reliability_setting),"deepFogGuard Accuracy:",deepFogGuard_acc)
        print(str(reliability_setting),"Vanilla Accuracy:",Vanilla_acc)

        ResiliNet_std = np.std(output["ResiliNet"][str(reliability_setting)],ddof=1)
        deepFogGuard_std = np.std(output["deepFogGuard"][str(reliability_setting)],ddof=1)
        Vanilla_std = np.std(output["Vanilla"][str(reliability_setting)],ddof=1)

        output_list.append(str(reliability_setting) + " ResiliNet std: " + str(ResiliNet_std) + '\n')
        output_list.append(str(reliability_setting) + " deepFogGuard std: " + str(deepFogGuard_std) + '\n')
        output_list.append(str(reliability_setting) + " Vanilla std: " + str(Vanilla_std) + '\n')

        print(str(reliability_setting),"ResiliNet std:",ResiliNet_std)
        print(str(reliability_setting),"deepFogGuard std:",deepFogGuard_std)
        print(str(reliability_setting),"Vanilla std:",Vanilla_std)

    write_n_upload(output_name, output_list, use_GCP)
    print(output)