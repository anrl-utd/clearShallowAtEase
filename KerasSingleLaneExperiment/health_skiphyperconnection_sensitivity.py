
from KerasSingleLaneExperiment.mlp_deepFogGuard_health import define_deepFogGuard_MLP
from KerasSingleLaneExperiment.health_common_exp_methods import init_data, init_common_experiment_params, convert_to_string
from KerasSingleLaneExperiment.FailureIteration import calculateExpectedAccuracy
from KerasSingleLaneExperiment.main import average
import keras.backend as K
import datetime
import os
import gc 
from keras.callbacks import ModelCheckpoint
import numpy as np
# runs all hyperconnection configurations for both deepFogGuard survival configurations
if __name__ == "__main__":
    use_GCP = True
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data(use_GCP)

    num_iterations, num_vars, num_classes, survivability_settings, num_train_epochs, hidden_units, batch_size = init_common_experiment_params(training_data)
    skip_hyperconnection_configurations = [
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [1,1,1],
    ]
    default_survivability_setting = [1,1,1]

    load_model = False
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    file_name = 'results/health_skiphyperconnection_sensitivity.txt'
    
    verbose = 2
    # keep track of output so that output is in order
    output_list = []

    # convert hyperconnection configuration into strings to be used as keys for dictionary
    config_1 = str(skip_hyperconnection_configurations[0])
    config_2 = str(skip_hyperconnection_configurations[1])
    config_3 = str(skip_hyperconnection_configurations[2])
    config_4 = str(skip_hyperconnection_configurations[3])
    config_5 = str(skip_hyperconnection_configurations[4])
    config_6 = str(skip_hyperconnection_configurations[5])
    config_7 = str(skip_hyperconnection_configurations[6])
    config_8 = str(skip_hyperconnection_configurations[7])

    no_failure, normal, poor, hazardous = convert_to_string(survivability_settings)

    # dictionary to store all the results
    output = {
        "deepFogGuard":
        {
            hazardous:
            {
                config_1:[0] * num_iterations,
                config_2:[0] * num_iterations,
                config_3:[0] * num_iterations,
                config_4:[0] * num_iterations,
                config_5:[0] * num_iterations,
                config_6:[0] * num_iterations,
                config_7:[0] * num_iterations,
                config_8:[0] * num_iterations
            },
            poor:
            {
                config_1:[0] * num_iterations,
                config_2:[0] * num_iterations,
                config_3:[0] * num_iterations,
                config_4:[0] * num_iterations,
                config_5:[0] * num_iterations,
                config_6:[0] * num_iterations,
                config_7:[0] * num_iterations,
                config_8:[0] * num_iterations
            },
            normal:
            {
                config_1:[0] * num_iterations,
                config_2:[0] * num_iterations,
                config_3:[0] * num_iterations,
                config_4:[0] * num_iterations,
                config_5:[0] * num_iterations,
                config_6:[0] * num_iterations,
                config_7:[0] * num_iterations,
                config_8:[0] * num_iterations
            },
            no_failure:
            {
                config_1:[0] * num_iterations,
                config_2:[0] * num_iterations,
                config_3:[0] * num_iterations,
                config_4:[0] * num_iterations,
                config_5:[0] * num_iterations,
                config_6:[0] * num_iterations,
                config_7:[0] * num_iterations,
                config_8:[0] * num_iterations
            },
        }
    }
    # make folder for outputs 
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
          
            # deepFogGuard
            deepFogGuard = define_deepFogGuard_MLP(num_vars,num_classes,hidden_units,default_survivability_setting,skip_hyperconnection_configuration)
            deepFogGuard_file = str(iteration) + " " + str(skip_hyperconnection_configuration) +  'health_skiphyperconnection_sensitivity_deepFogGuard.h5'
            if load_model:
                deepFogGuard.load_weights(deepFogGuard_file)
            else:
                print("Training deepFogGuard")
                dFGCheckPoint = ModelCheckpoint(deepFogGuard_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuard.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [dFGCheckPoint], validation_data=(val_data,val_labels))
                # load weights from epoch with the highest val acc
                deepFogGuard.load_weights(deepFogGuard_file)

            # test models
            for survivability_setting in survivability_settings:
                # write results to a file 
                # survival setting
                print(survivability_setting)
                output_list.append(str(survivability_setting) + '\n')

                # deepFogGuard
                output_list.append('deepFogGuard' + '\n')
                print("deepFogGuard")
                output["deepFogGuard"][str(survivability_setting)][str(skip_hyperconnection_configuration)][iteration-1] = calculateExpectedAccuracy(deepFogGuard,survivability_setting,output_list,training_labels,test_data,test_labels)

            # clear session to remove old graphs from memory so that subsequent training is not slower
            K.clear_session()
            gc.collect()
            del deepFogGuard

   # write average accuracies to a file 
    with open(file_name,'a+') as file:
        for survivability_setting in survivability_settings:
            output_list.append(str(survivability_setting) + '\n')
            for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
                output_list.append(str(skip_hyperconnection_configuration) + '\n')
                deepFogGuard_acc = average(output["deepFogGuard"][str(survivability_setting)][str(skip_hyperconnection_configuration)])
                deepFogGuard_std = np.std(output["deepFogGuard"][str(survivability_setting)][str(skip_hyperconnection_configuration)],ddof=1)
                # write to output list
                output_list.append(str(survivability_setting) + " " + str(skip_hyperconnection_configuration) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
                print(str(survivability_setting),str(skip_hyperconnection_configuration),"deepFogGuard Accuracy:",deepFogGuard_acc)
                output_list.append(str(survivability_setting) + " " + str(skip_hyperconnection_configuration) + " deepFogGuard std: " + str(deepFogGuard_std) + '\n')
                print(str(survivability_setting),str(skip_hyperconnection_configuration),"deepFogGuard std:",deepFogGuard_std)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    print(output)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')