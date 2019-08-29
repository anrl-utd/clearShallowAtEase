
from KerasSingleLaneExperiment.mlp_deepFogGuardPlus_health import define_deepFogGuardPlus_MLP
from KerasSingleLaneExperiment.FailureIteration import calculateExpectedAccuracy
from KerasSingleLaneExperiment.main import average
from KerasSingleLaneExperiment.health_common_exp_methods import init_data, init_common_experiment_params, convert_to_string
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
    nodewise_survival_rates = [
        [.95,.95,.95],
        [.9,.9,.9],
        [.7,.7,.7],
        [.5,.5,.5],
    ]

    # file name with the experiments accuracy output
    output_name = "results/health_nodewise_dropout.txt"
    verbose = 2
    # keep track of output so that output is in order
    output_list = []

    no_failure, normal, poor, hazardous = convert_to_string(survivability_settings)
    
    # convert dropout rates into strings
    nodewise_dropout_rate_05 =  str(nodewise_survival_rates[0])
    nodewise_dropout_rate_10 = str(nodewise_survival_rates[1])
    nodewise_dropout_rate_30 = str(nodewise_survival_rates[2])
    nodewise_dropout_rate_50 = str(nodewise_survival_rates[3])
    # dictionary to store all the results
    output = {
        "deepFogGuardPlus Node-wise Dropout": 
        {
            nodewise_dropout_rate_05:
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            nodewise_dropout_rate_10 :
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            nodewise_dropout_rate_30:
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            nodewise_dropout_rate_50:
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
        },
        "deepFogGuardPlus Adjusted Node-wise Dropout": 
        {
            nodewise_dropout_rate_05:
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            nodewise_dropout_rate_10 :
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            nodewise_dropout_rate_30:
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            nodewise_dropout_rate_50:
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
        }
    }

    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        output_list.append('deepFogGuardPlus Node-wise Dropout' + '\n')                  
        print("deepFogGuardPlus Node-wise Dropout")
        for nodewise_survival_rate in nodewise_survival_rates:
            # node-wise dropout
            deepFogGuardPlus_nodewise_dropout_file = str(iteration) + " " + str(nodewise_survival_rate) + 'health_nodewise_dropout.h5'
            deepFogGuardPlus_nodewise_dropout = define_deepFogGuardPlus_MLP(num_vars,num_classes,hidden_units,nodewise_survival_rate)
            # adjusted node_wise dropout
            deepFogGuardPlus_adjusted_nodewise_dropout_file = str(iteration) + " " + str(nodewise_survival_rate) + 'health_nodewise_dropout.h5'
            deepFogGuardPlus_adjusted_nodewise_dropout = define_deepFogGuardPlus_MLP(num_vars,num_classes,hidden_units,nodewise_survival_rate,standard_dropout=True)
            if load_model:
                deepFogGuardPlus_nodewise_dropout.load_weights(deepFogGuardPlus_nodewise_dropout_file)
                deepFogGuardPlus_adjusted_nodewise_dropout.load_weights(deepFogGuardPlus_adjusted_nodewise_dropout_file)
            else:
                #print("Training deepFogGuardPlus Node-wise Dropout")
                print(str(nodewise_survival_rate))
                # node-wise dropout
                deepFogGuardPlus_nodewise_dropout_CheckPoint = ModelCheckpoint(deepFogGuardPlus_nodewise_dropout_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuardPlus_nodewise_dropout.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [deepFogGuardPlus_nodewise_dropout_CheckPoint],validation_data=(val_data,val_labels))
                deepFogGuardPlus_nodewise_dropout.load_weights(deepFogGuardPlus_nodewise_dropout_file)
                # adjusted node-wise dropout
                print("Training deepFogGuardPlus Adjusted Node-wise Dropout")
                deepFogGuardPlus_adjusted_nodewise_dropout_CheckPoint = ModelCheckpoint(deepFogGuardPlus_adjusted_nodewise_dropout_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuardPlus_adjusted_nodewise_dropout.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [deepFogGuardPlus_adjusted_nodewise_dropout_CheckPoint],validation_data=(val_data,val_labels))
                deepFogGuardPlus_adjusted_nodewise_dropout.load_weights(deepFogGuardPlus_adjusted_nodewise_dropout_file)
                print("Test on normal survival rates")
                output_list.append("Test on normal survival rates" + '\n')
                for survivability_setting in survivability_settings:
                    output_list.append(str(survivability_setting)+ '\n')
                    print(survivability_setting)
                    output["deepFogGuardPlus Node-wise Dropout"][str(nodewise_survival_rate)][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuardPlus_nodewise_dropout,survivability_setting,output_list,training_labels,test_data,test_labels)
                    output["deepFogGuardPlus Adjusted Node-wise Dropout"][str(nodewise_survival_rate)][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuardPlus_adjusted_nodewise_dropout,survivability_setting,output_list,training_labels,test_data,test_labels)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            #del deepFogGuardPlus_nodewise_dropout
            del deepFogGuardPlus_adjusted_nodewise_dropout

    # calculate average accuracies for deepFogGuardPlus Node-wise Dropout
    for nodewise_survival_rate in nodewise_survival_rates:
        print(nodewise_survival_rate)
        for survivability_setting in survivability_settings:
            deepFogGuardPlus_nodewise_dropout_acc = average(output["deepFogGuardPlus Node-wise Dropout"][str(nodewise_survival_rate)][str(survivability_setting)])
            output_list.append(str(nodewise_survival_rate) + str(survivability_setting) + " deepFogGuardPlus Node-wise Dropout: " + str(deepFogGuardPlus_nodewise_dropout_acc) + '\n')
            print(nodewise_survival_rate,survivability_setting,"deepFogGuardPlus Node-wise Dropout:",deepFogGuardPlus_nodewise_dropout_acc)  

            deepGuardPlus_std = np.std(output["deepFogGuardPlus Node-wise Dropout"][str(survivability_setting)],ddof=1)
            output_list.append(str(survivability_setting) + " nodewise_survival_rate std: " + str(deepGuardPlus_std) + '\n')
            print(str(survivability_setting), "nodewise_survival_rate std:",deepGuardPlus_std)

            deepFogGuardPlus_adjusted_nodewise_dropout_acc = average(output["deepFogGuardPlus Adjusted Node-wise Dropout"][str(nodewise_survival_rate)][str(survivability_setting)])
            output_list.append(str(nodewise_survival_rate) + str(survivability_setting) + " deepFogGuardPlus Adjusted Node-wise Dropout: " + str(deepFogGuardPlus_adjusted_nodewise_dropout_acc) + '\n')
            print(nodewise_survival_rate,survivability_setting,"deepFogGuardPlus Adjusted Node-wise Dropout:",deepFogGuardPlus_adjusted_nodewise_dropout_acc) 

            adjusted_deepGuardPlus_std = np.std(output["deepFogGuardPlus Adjusted Node-wise Dropout"][str(nodewise_survival_rate)][str(survivability_setting)], ddof=1)
            output_list.append(str(survivability_setting) + " adjusted nodewise_survival_rate std: " + str(adjusted_deepGuardPlus_std) + '\n')
            print(str(survivability_setting), "adjusted nodewise_survival_rate std:",adjusted_deepGuardPlus_std) 
    # write experiments output to file
    with open(output_name,'w') as file:
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(output_name))
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
