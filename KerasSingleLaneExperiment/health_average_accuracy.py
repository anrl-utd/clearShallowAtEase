
from KerasSingleLaneExperiment.deepFogGuardPlus import define_deepFogGuardPlus
from KerasSingleLaneExperiment.deepFogGuard import define_deepFogGuard
from KerasSingleLaneExperiment.Vanilla import define_vanilla_model
from KerasSingleLaneExperiment.loadData import load_data
from sklearn.model_selection import train_test_split
from KerasSingleLaneExperiment.FailureIteration import calculateExpectedAccuracy
from KerasSingleLaneExperiment.main import average
import keras.backend as K
import datetime
import gc
import os
from keras.callbacks import ModelCheckpoint

# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    use_GCP = True
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/mHealth_complete.log ./')
        os.mkdir('models/')
    data,labels= load_data('mHealth_complete.log')
    # split data into train, val, and test, 80/10/10 split
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
    default_nodewise_survival_rate = [.95,.95,.95]
    allpresent_skip_hyperconnections_configuration = [1,1,1]
    default_survivability_setting = [1,1,1]
    hidden_units = 250
    batch_size = 1028
    load_model = False
    num_train_epochs = 25 

    # file name with the experiments accuracy output
    output_name = "results/health_normal.txt"
    num_iterations = 10
    verbose = 2

    # keep track of output so that output is in order
    output_list = []
    
    # convert survivability settings into strings so it can be used in the dictionary as keys
    no_failure = str(survivability_settings[0])
    normal = str(survivability_settings[1])
    poor = str(survivability_settings[2])
    hazardous = str(survivability_settings[3])

    # dictionary to store all the results
    output = {
        "deepFogGuard Plus":
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

    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)

        # deepFogGuardPlus
        deepFogGuardPlus = define_deepFogGuardPlus(num_vars,num_classes,hidden_units,default_nodewise_survival_rate)
        deepFogGuardPlus_file = "new_split_" + str(iteration) + '_deepFogGuardPlus.h5'
        if load_model:
            deepFogGuardPlus.load_weights(deepFogGuardPlus_file)
        else:
            print("Training deepFogGuardPlus")
            dFGPlusCheckPoint = ModelCheckpoint(deepFogGuardPlus_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            deepFogGuardPlus.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [dFGPlusCheckPoint],validation_data=(val_data,val_labels))
            # load weights from epoch with the highest val acc
            deepFogGuardPlus.load_weights(deepFogGuardPlus_file)

        # deepFogGuard
        deepFogGuard = define_deepFogGuard(num_vars,num_classes,hidden_units,default_survivability_setting,allpresent_skip_hyperconnections_configuration)
        deepFogGuard_file = "new_split_" + str(iteration) + '_deepFogGuard.h5'
        if load_model:
            deepFogGuard.load_weights(deepFogGuard_file)
        else:
            print("Training deepFogGuard")
            dFGCheckPoint = ModelCheckpoint(deepFogGuard_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            deepFogGuard.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [dFGCheckPoint], validation_data=(val_data,val_labels))
            # load weights from epoch with the highest val acc
            deepFogGuard.load_weights(deepFogGuard_file)

        # vanilla model
        vanilla = define_vanilla_model(num_vars,num_classes,hidden_units)
        vanilla_file = "new_split_" + str(iteration) + '_vanilla.h5'
        if load_model:
            vanilla.load_weights(vanilla_file)
        else:
            print("Training vanilla")
            vanillaCheckPoint = ModelCheckpoint(vanilla_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            vanilla.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [vanillaCheckPoint],validation_data=(val_data,val_labels))
            # load weights from epoch with the highest val acc
            vanilla.load_weights(vanilla_file)
 
        # test models
        for survivability_setting in survivability_settings:
         
            # deepFogGuard Plus
            output_list.append('deepFogGuard Plus' + '\n')
            print("deepFogGuard Plus")
            output["deepFogGuard Plus"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuardPlus,survivability_setting,output_list,training_labels,test_data,test_labels)

            # deepFogGuard
            output_list.append('deepFogGuard' + '\n')
            print("deepFogGuard")
            output["deepFogGuard"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuard,survivability_setting,output_list,training_labels,test_data,test_labels)

            # vanilla
            output_list.append('Vanilla' + '\n')                    
            print("Vanilla")
            output["Vanilla"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(vanilla,survivability_setting,output_list,training_labels,test_data,test_labels)

        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del deepFogGuard
        del deepFogGuardPlus
        del vanilla
   # calculate average accuracies from all expected accuracies
    for survivability_setting in survivability_settings:
        deepfogGuardPlus_acc = average(output["deepFogGuard Plus"][str(survivability_setting)])
        deepFogGuard_acc = average(output["deepFogGuard"][str(survivability_setting)])
        vanilla_acc = average(output["Vanilla"][str(survivability_setting)])

        output_list.append(str(survivability_setting) + " deepFogGuard Plus Accuracy: " + str(deepfogGuardPlus_acc) + '\n')
        output_list.append(str(survivability_setting) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
        output_list.append(str(survivability_setting) + " Vanilla Accuracy: " + str(vanilla_acc) + '\n')

        print(str(survivability_setting),"deepFogGuard Plus Accuracy:",deepfogGuardPlus_acc)
        print(str(survivability_setting),"deepFogGuard Accuracy:",deepFogGuard_acc)
        print(str(survivability_setting),"Vanilla Accuracy:",vanilla_acc)

    # write experiments output to file
    with open(output_name,'w') as file:
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    # upload file to GCP
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(output_name))
    print(output)
