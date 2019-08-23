
from KerasSingleLaneExperiment.deepFogGuardPlus import define_deepFogGuardPlus, define_adjusted_deepFogGuardPlus
from KerasSingleLaneExperiment.loadData import load_data
from sklearn.model_selection import train_test_split
from KerasSingleLaneExperiment.FailureIteration import calculateExpectedAccuracy
from KerasSingleLaneExperiment.main import average
import keras.backend as K
import gc
import os
from keras.callbacks import ModelCheckpoint

def multiply_dropout_rate(survivability_setting):
    complement_setting = list(map(lambda num : 1 - num,survivability_setting))
    complement_setting = list(map(lambda num : num * 10,complement_setting))
    return list(map(lambda num : 1 - num,complement_setting))
# runs all 3 failure configurations for all 3 models
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
        [.92,.96,.99],
        [.87,.91,.95],
        [.78,.8,.85],
    ]
    hidden_units = 250
    batch_size = 1028
    load_model = False
    num_train_epochs = 25 
    # file name with the experiments accuracy output
    output_name = "results/health_fixed10xvariable_nodewise_dropout.txt"
    num_iterations = 10
    verbose = 2
    # keep track of output so that output is in order
    output_list = []
    
    # convert survivability settings into strings so it can be used in the dictionary as keys
    normal = str(survivability_settings[0])
    poor = str(survivability_settings[1])
    hazardous = str(survivability_settings[2])
    
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

    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        output_list.append('deepFogGuardPlus Node-wise Dropout' + '\n')                  
        for survivability_setting in survivability_settings:
            # variable node-wise dropout
            deepFogGuardPlus_variable_nodewise_dropout_file = str(iteration) + " " + str(survivability_setting) + 'health_variable_nodewise_dropout.h5'
            deepFogGuardPlus_variable_nodewise_dropout = define_deepFogGuardPlus(num_vars,num_classes,hidden_units,survivability_setting)

            # 10x variable node_wise dropout
            deepFogGuardPlus_variable_10x_nodewise_dropout_file = str(iteration) + " " + str(survivability_setting) + 'health_fixed_10xvariable_nodewise_dropout.h5'
            # multiply the dropout rate by 10
            survivability_setting_10x = multiply_dropout_rate(survivability_setting)
            deepFogGuardPlus_variable_10x_nodewise_dropout = define_deepFogGuardPlus(num_vars,num_classes,hidden_units,survivability_setting_10x)
            if load_model:
                deepFogGuardPlus_variable_nodewise_dropout.load_weights(deepFogGuardPlus_variable_nodewise_dropout_file)
                deepFogGuardPlus_variable_10x_nodewise_dropout.load_weights(deepFogGuardPlus_variable_10x_nodewise_dropout_file)
            else:
                print("Training deepFogGuardPlus Variable Node-wise Dropout")
                print(str(survivability_setting))
                # node-wise dropout
                deepFogGuardPlus_variable_nodewise_dropout_CheckPoint = ModelCheckpoint(deepFogGuardPlus_variable_nodewise_dropout_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuardPlus_variable_nodewise_dropout.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [deepFogGuardPlus_variable_nodewise_dropout_CheckPoint],validation_data=(val_data,val_labels))
                deepFogGuardPlus_variable_nodewise_dropout.load_weights(deepFogGuardPlus_variable_nodewise_dropout_file)
                # 10x Variable Dropout
                print("deepFogGuardPlus Node-wise 10x Variable Dropout")
                deepFogGuardPlus_variable_10x_nodewise_dropout_CheckPoint = ModelCheckpoint(deepFogGuardPlus_variable_10x_nodewise_dropout_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                deepFogGuardPlus_variable_10x_nodewise_dropout.fit(training_data,training_labels,epochs=num_train_epochs, batch_size=batch_size,verbose=verbose,shuffle = True, callbacks = [deepFogGuardPlus_variable_10x_nodewise_dropout_CheckPoint],validation_data=(val_data,val_labels))
                deepFogGuardPlus_variable_10x_nodewise_dropout.load_weights(deepFogGuardPlus_variable_10x_nodewise_dropout_file)

                output["deepFogGuardPlus Node-wise Variable Dropout"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuardPlus_variable_nodewise_dropout,survivability_setting,output_list,training_labels,test_data,test_labels)
                output["deepFogGuardPlus Node-wise 10x Variable Dropout"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuardPlus_variable_10x_nodewise_dropout,survivability_setting,output_list,training_labels,test_data,test_labels)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del deepFogGuardPlus_variable_nodewise_dropout
            del deepFogGuardPlus_variable_10x_nodewise_dropout
    # calculate average accuracies for deepFogGuardPlus Node-wise Dropout
    for survivability_setting in survivability_settings:
        deepFogGuardPlus_variable_nodewise_dropout_acc = average(output["deepFogGuardPlus Node-wise Variable Dropout"][str(survivability_setting)])
        output_list.append(str(survivability_setting) + " deepFogGuardPlus Node-wise Variable Dropout: " + '\n')
        print(survivability_setting,"deepFogGuardPlus Node-wise Variable Dropout:",deepFogGuardPlus_variable_nodewise_dropout_acc)  

        deepFogGuardPlus_variable_10x_nodewise_dropout_acc = average(output["deepFogGuardPlus Node-wise 10x Variable Dropout"][str(survivability_setting)])
        output_list.append(str(survivability_setting) + " deepFogGuardPlus Node-wise Dropout: " + str(deepFogGuardPlus_variable_10x_nodewise_dropout_acc) + '\n')
        print(survivability_setting,"deepFogGuardPlus Node-wise Dropout:",deepFogGuardPlus_variable_10x_nodewise_dropout_acc)  


    # write experiments output to file
    with open(output_name,'w') as file:
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(output_name))
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
    print(output)
