
from KerasSingleLaneExperiment.mlp_deepFogGuardPlus_health import define_deepFogGuardPlus_MLP
from KerasSingleLaneExperiment.mlp_deepFogGuard_health import define_deepFogGuard_MLP
from KerasSingleLaneExperiment.mlp_Vanilla_health import define_vanilla_model_MLP
from KerasSingleLaneExperiment.FailureIteration import calculateExpectedAccuracy
from KerasSingleLaneExperiment.main import average
from KerasSingleLaneExperiment.health_common_exp_methods import init_data, init_common_experiment_params, convert_to_string, write_n_upload
import keras.backend as K
import datetime
import gc
import os
from keras.callbacks import ModelCheckpoint

def make_output_dictionary(survivability_settings, num_iterations):
    no_failure, normal, poor, hazardous = convert_to_string(survivability_settings)

    # dictionary to store all the results
    output = {
        "ResiliNet":
        {
            hazardous:[0] * num_iterations,
            poor:[0] * num_iterations,
            normal:[0] * num_iterations,
            no_failure:[0] * num_iterations,
        }, 
        "deepFogGuard":
        {
            hazardous:[0] * num_iterations,
            poor:[0] * num_iterations,
            normal:[0] * num_iterations,
            no_failure:[0] * num_iterations,
        },
        "Vanilla": 
        {
            hazardous:[0] * num_iterations,
            poor:[0] * num_iterations,
            normal:[0] * num_iterations,
            no_failure:[0] * num_iterations,
        },
    }
    return output

def define_and_train(iteration, model_name, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose, default_failout_survival_rate, default_survivability_setting, allpresent_skip_hyperconnections_configuration):
    # ResiliNet
    if model_name == "ResiliNet":
        model = define_deepFogGuardPlus_MLP(num_vars,num_classes,hidden_units,default_failout_survival_rate)
        model_file = "new_split_" + str(iteration) + '_deepFogGuardPlus.h5'
    # deepFogGuard
    if model_name == "deepFogGuard":
        model = define_deepFogGuard_MLP(num_vars, num_classes, hidden_units, default_survivability_setting, allpresent_skip_hyperconnections_configuration)
        model_file = "new_split_" + str(iteration) + '_deepFogGuard.h5'
    # Vanilla model
    if model_name == "Vanilla":
        model = define_vanilla_model_MLP(num_vars,num_classes,hidden_units)
        model_file = "new_split_" + str(iteration) + '_vanilla.h5'
    
    if load_model:
        model.load_weights(model_file)
    else:
        print("Training " + model_name)
        modelCheckPoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        model.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [modelCheckPoint],validation_data=(val_data,val_labels))
        # load weights from epoch with the highest val acc
        model.load_weights(model_file)
    return model

def calc_accuracy(iteration, model_name, model, survivability_setting, output_list,training_labels,test_data,test_labels):
    output_list.append(model_name + "\n")
    print(model_name)
    output[model_name][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(model,survivability_setting,output_list,training_labels,test_data,test_labels)

# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    use_GCP = True
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data(use_GCP) 
    num_iterations, num_vars, num_classes, survivability_settings, num_train_epochs, hidden_units, batch_size = init_common_experiment_params(training_data)

    default_failout_survival_rate = [.95,.95,.95]
    allpresent_skip_hyperconnections_configuration = [1,1,1]
    default_survivability_setting = [1,1,1]
    load_model = False

    # file name with the experiments accuracy output
    output_name = "results/health_normal_testfordfg.txt"
    num_iterations = 1
    verbose = 2

    # keep track of output so that output is in order
    output_list = []
    
    output = make_output_dictionary(survivability_settings, num_iterations)

    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        ResiliNet = define_and_train(iteration, "ResiliNet", training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose, default_failout_survival_rate, default_survivability_setting, allpresent_skip_hyperconnections_configuration)
        deepFogGuard = define_and_train(iteration, "deepFogGuard", training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose, default_failout_survival_rate, default_survivability_setting, allpresent_skip_hyperconnections_configuration)
        Vanilla = define_and_train(iteration, "Vanilla", training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, num_vars, num_classes, hidden_units, verbose, default_failout_survival_rate, default_survivability_setting, allpresent_skip_hyperconnections_configuration)
 
        # test models
        for survivability_setting in survivability_settings:
            calc_accuracy(iteration, "ResiliNet", ResiliNet, survivability_setting, output_list,training_labels,test_data,test_labels)
            calc_accuracy(iteration, "deepFogGuard", deepFogGuard, survivability_setting, output_list,training_labels,test_data,test_labels)
            calc_accuracy(iteration, "Vanilla", Vanilla, survivability_setting, output_list,training_labels,test_data,test_labels)
            
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del deepFogGuard
        del ResiliNet
        del Vanilla
   # calculate average accuracies from all expected accuracies
    for survivability_setting in survivability_settings:
        ResiliNet_acc = average(output["ResiliNet"][str(survivability_setting)])
        deepFogGuard_acc = average(output["deepFogGuard"][str(survivability_setting)])
        Vanilla_acc = average(output["Vanilla"][str(survivability_setting)])

        output_list.append(str(survivability_setting) + " ResiliNet Accuracy: " + str(ResiliNet_acc) + '\n')
        output_list.append(str(survivability_setting) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
        output_list.append(str(survivability_setting) + " Vanilla Accuracy: " + str(Vanilla_acc) + '\n')

        print(str(survivability_setting),"ResiliNet Accuracy:",ResiliNet_acc)
        print(str(survivability_setting),"deepFogGuard Accuracy:",deepFogGuard_acc)
        print(str(survivability_setting),"Vanilla Accuracy:",Vanilla_acc)

    write_n_upload(output_name, output_list, use_GCP)
    print(output)