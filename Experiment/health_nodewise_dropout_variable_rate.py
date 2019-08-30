from Experiment.mlp_deepFogGuardPlus_health import define_deepFogGuardPlus_MLP
from Experiment.FailureIteration import calculateExpectedAccuracy
from Experiment.utility import average
from Experiment.health_common_exp_methods import init_data, init_common_experiment_params
from Experiment.common_exp_methods import convert_to_string, make_results_folder
import keras.backend as K
import gc
import os
from keras.callbacks import ModelCheckpoint
import numpy as np


# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    use_GCP = True
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data(use_GCP)

    num_iterations, num_vars, num_classes, survivability_settings, num_train_epochs, hidden_units, batch_size = init_common_experiment_params(training_data)
    
    load_model = False
    # file name with the experiments accuracy output
    output_name = "results/health_variable_dropoutlike_nodewise_dropout.txt"
    verbose = 2
    # keep track of output so that output is in order
    output_list = []
    
    no_failure, normal, poor, hazardous = convert_to_string(survivability_settings)
    
    # failout based on survivaility  should not have the `no failure`, as with `no failure` there would be no 
    survivability_settings.remove([1,1,1])
    # dictionary to store all the results
    output = {
         "deepFogGuardPlus Node-wise Variable Dropout": 
        {
            hazardous:[0] * num_iterations,
            poor :[0] * num_iterations,
            normal:[0] * num_iterations,
        },
        "deepFogGuardPlus Node-wise 10x Variable Dropout": 
        {
            hazardous:[0] * num_iterations,
            poor :[0] * num_iterations,
            normal:[0] * num_iterations,
        },
    }
    standard_dropout = True
    make_results_folder()
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        output_list.append('deepFogGuardPlus Node-wise Dropout' + '\n')                  
        for survivability_setting in survivability_settings:
            # variable node-wise dropout
            deepFogGuardPlus_variable_nodewise_dropout_file = str(iteration) + " " + str(survivability_setting) + 'health_variable_nodewise_dropoutlike_dropout.h5'
            deepFogGuardPlus_variable_nodewise_dropout = define_deepFogGuardPlus_MLP(num_vars,num_classes,hidden_units,failout_survival_setting=survivability_setting, standard_dropout= True)

            if load_model:
                deepFogGuardPlus_variable_nodewise_dropout.load_weights(deepFogGuardPlus_variable_nodewise_dropout_file)
            else:
                print("Training deepFogGuardPlus Variable Node-wise Dropout")
                print(str(survivability_setting))
                # node-wise dropout
                deepFogGuardPlus_variable_nodewise_dropout_CheckPoint = ModelCheckpoint(deepFogGuardPlus_variable_nodewise_dropout_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuardPlus_variable_nodewise_dropout.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [deepFogGuardPlus_variable_nodewise_dropout_CheckPoint],validation_data=(val_data,val_labels))
                deepFogGuardPlus_variable_nodewise_dropout.load_weights(deepFogGuardPlus_variable_nodewise_dropout_file)
                if standard_dropout == True:
                    nodes = ["edge_output_layer","fog2_output_layer","fog1_output_layer"]
                    for index, node in enumerate(nodes):
                        survival_rate = survivability_setting[index]
                        # node failed
                        layer_name = node
                        layer = deepFogGuardPlus_variable_nodewise_dropout.get_layer(name=layer_name)
                        layer_weights = layer.get_weights()
                        # make new weights for the connections
                        new_weights = layer_weights[0] * survival_rate
    
                        # make new weights for biases
                        new_bias_weights = layer_weights[1] * survival_rate
                        layer.set_weights([new_weights,new_bias_weights])
                        print(layer_name, "was multiplied")
                output["deepFogGuardPlus Node-wise Variable Dropout"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuardPlus_variable_nodewise_dropout,survivability_setting,output_list,training_labels,test_data,test_labels)
        
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del deepFogGuardPlus_variable_nodewise_dropout
    # calculate average accuracies for deepFogGuardPlus Node-wise Dropout
    for survivability_setting in survivability_settings:
        deepFogGuardPlus_variable_nodewise_dropout_acc = average(output["deepFogGuardPlus Node-wise Variable Dropout"][str(survivability_setting)])
        output_list.append(str(survivability_setting) + " deepFogGuardPlus Node-wise Variable Dropout: " + '\n')
        print(survivability_setting,"deepFogGuardPlus Node-wise Variable Dropout:",deepFogGuardPlus_variable_nodewise_dropout_acc)  

        deepGuardPlus_std = np.std(output["deepFogGuardPlus Node-wise Variable Dropout"][str(survivability_setting)],ddof=1)
        output_list.append(str(survivability_setting) + " failout_survival_rate std: " + str(deepGuardPlus_std) + '\n')
        print(str(survivability_setting), " variable failout_survival_rate std:",deepGuardPlus_std)
    # write experiments output to file
    with open(output_name,'w') as file:
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(output_name))
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')

