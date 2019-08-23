from KerasSingleLaneExperiment.deepFogGuardPlus import define_deepFogGuardPlus
from KerasSingleLaneExperiment.deepFogGuard import define_deepFogGuard
from KerasSingleLaneExperiment.loadData import load_data
from sklearn.model_selection import train_test_split
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
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/mHealth_complete.log ./')
        os.mkdir('models/')
    data,labels= load_data('mHealth_complete.log')
    # split data into train, val, and test
    # 80/10/10 split
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .20, shuffle = True)
    val_data, test_data, val_labels, test_labels = train_test_split(test_data,test_labels,random_state = 42, test_size = .50, shuffle = True)
    
    num_vars = len(training_data[0])
    num_classes = 13
    survivability_settings = [
        [1,1,1],
        [.92,.96,.99],
        [.87,.91,.95],
        [.78,.8,.85],
    ]
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
    num_train_epochs = 25 
    hidden_units = 250
    batch_size = 1028
    load_model = False
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    file_name = 'results/health_skiphyperconnection_sensitivity.txt'
    num_iterations = 10
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

    # convert survivability settings into strings so it can be used in the dictionary as keys
    no_failure = str(survivability_settings[0])
    normal = str(survivability_settings[1])
    poor = str(survivability_settings[2])
    hazardous = str(survivability_settings[3])
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
            deepFogGuard = define_deepFogGuard(num_vars,num_classes,hidden_units,default_survivability_setting,skip_hyperconnection_configuration)
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