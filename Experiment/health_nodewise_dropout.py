
from Experiment.mlp_deepFogGuardPlus_health import define_deepFogGuardPlus_MLP
from Experiment.FailureIteration import calculateExpectedAccuracy
from Experiment.utility import average
from Experiment.health_common_exp_methods import init_data, init_common_experiment_params
from Experiment.common_exp_methods import convert_to_string, make_results_folder, make_output_dictionary_failout_rate
import keras.backend as K
import gc
import os
from keras.callbacks import ModelCheckpoint
import numpy as np
# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    

    make_results_folder()
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        output_list.append('deepFogGuardPlus Node-wise Dropout' + '\n')                  
        print("deepFogGuardPlus Node-wise Dropout")
        for failout_survival_setting in failout_survival_rates:
            # node-wise dropout
            deepFogGuardPlus_nodewise_dropout_file = str(iteration) + " " + str(failout_survival_setting) + 'health_updated_nodewise_dropout.h5'
            deepFogGuardPlus_nodewise_dropout = define_deepFogGuardPlus_MLP(num_vars,num_classes,hidden_units,failout_survival_setting)
            # adjusted node_wise dropout
            deepFogGuardPlus_adjusted_nodewise_dropout_file = str(iteration) + " " + str(failout_survival_setting) + 'health_dropoutlike_failout_nodewise_dropout_05.h5'
            deepFogGuardPlus_adjusted_nodewise_dropout = define_deepFogGuardPlus_MLP(num_vars,num_classes,hidden_units,failout_survival_setting)
            if load_model:
                deepFogGuardPlus_nodewise_dropout.load_weights(deepFogGuardPlus_nodewise_dropout_file)
                    .load_weights(deepFogGuardPlus_adjusted_nodewise_dropout_file)
            else:
                print("Training deepFogGuardPlus Node-wise Dropout")
                print(str(failout_survival_setting))
                # node-wise dropout
                deepFogGuardPlus_nodewise_dropout_CheckPoint = ModelCheckpoint(deepFogGuardPlus_nodewise_dropout_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuardPlus_nodewise_dropout.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [deepFogGuardPlus_nodewise_dropout_CheckPoint],validation_data=(val_data,val_labels))
                deepFogGuardPlus_nodewise_dropout.load_weights(deepFogGuardPlus_nodewise_dropout_file)
                # adjusted node-wise dropout
                print("Training deepFogGuardPlus Adjusted Node-wise Dropout")
                deepFogGuardPlus_adjusted_nodewise_dropout_CheckPoint = ModelCheckpoint(deepFogGuardPlus_adjusted_nodewise_dropout_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuardPlus_adjusted_nodewise_dropout.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [deepFogGuardPlus_adjusted_nodewise_dropout_CheckPoint],validation_data=(val_data,val_labels))
                deepFogGuardPlus_adjusted_nodewise_dropout.load_weights(deepFogGuardPlus_adjusted_nodewise_dropout_file)
                if dropout_like_failout == True:
                    nodes = ["edge_output_layer","fog2_output_layer","fog1_output_layer"]
                    default_survival_rate = .95
                    for node in nodes:
                        # node failed
                        layer_name = node
                        layer = deepFogGuardPlus_adjusted_nodewise_dropout.get_layer(name=layer_name)
                        layer_weights = layer.get_weights()
                        # make new weights for the connections
                        new_weights = layer_weights[0] * default_survival_rate
    
                        # make new weights for biases
                        new_bias_weights = layer_weights[1] * default_survival_rate
                        layer.set_weights([new_weights,new_bias_weights])
                        print(layer_name, "was multiplied")
                print("Test on normal survival rates")
                output_list.append("Test on normal survival rates" + '\n')
                for survivability_setting in survivability_settings:
                    output_list.append(str(survivability_setting)+ '\n')
                    print(survivability_setting)
                    output["deepFogGuardPlus Node-wise Dropout"][str(failout_survival_setting)][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuardPlus_nodewise_dropout,survivability_setting,output_list,training_labels,test_data,test_labels)
                    K.set_learning_phase(0)
                    output["deepFogGuardPlus Adjusted Node-wise Dropout"][str(failout_survival_setting)][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuardPlus_adjusted_nodewise_dropout,survivability_setting,output_list,training_labels,test_data,test_labels)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del deepFogGuardPlus_nodewise_dropout
            del deepFogGuardPlus_adjusted_nodewise_dropout

    # calculate average accuracies for deepFogGuardPlus Node-wise Dropout
    for failout_survival_setting in failout_survival_rates:
        print(failout_survival_setting)
        for survivability_setting in survivability_settings:
            deepFogGuardPlus_nodewise_dropout_acc = average(output["deepFogGuardPlus Node-wise Dropout"][str(failout_survival_setting)][str(survivability_setting)])
            output_list.append(str(failout_survival_setting) + str(survivability_setting) + " deepFogGuardPlus Node-wise Dropout: " + str(deepFogGuardPlus_nodewise_dropout_acc) + '\n')
            print(failout_survival_setting,survivability_setting,"deepFogGuardPlus Node-wise Dropout:",deepFogGuardPlus_nodewise_dropout_acc)  

            deepGuardPlus_std = np.std(output["deepFogGuardPlus Node-wise Dropout"][str(failout_survival_setting)][str(survivability_setting)],ddof=1)
            output_list.append(str(survivability_setting) + " failout_survival_setting std: " + str(deepGuardPlus_std) + '\n')
            print(str(survivability_setting), "failout_survival_setting std:",deepGuardPlus_std)

            deepFogGuardPlus_adjusted_nodewise_dropout_acc = average(output["deepFogGuardPlus Adjusted Node-wise Dropout"][str(failout_survival_setting)][str(survivability_setting)])
            output_list.append(str(failout_survival_setting) + str(survivability_setting) + " deepFogGuardPlus Adjusted Node-wise Dropout: " + str(deepFogGuardPlus_adjusted_nodewise_dropout_acc) + '\n')
            print(failout_survival_setting,survivability_setting,"deepFogGuardPlus Adjusted Node-wise Dropout:",deepFogGuardPlus_adjusted_nodewise_dropout_acc) 

            adjusted_deepGuardPlus_std = np.std(output["deepFogGuardPlus Adjusted Node-wise Dropout"][str(failout_survival_setting)][str(survivability_setting)], ddof=1)
            output_list.append(str(survivability_setting) + " adjusted failout_survival_setting std: " + str(adjusted_deepGuardPlus_std) + '\n')
            print(str(survivability_setting), "adjusted failout_survival_setting std:",adjusted_deepGuardPlus_std) 
    # write experiments output to file
    with open(output_name,'w') as file:
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(output_name))
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
