
from KerasSingleLaneExperiment.deepFogGuardPlus import define_deepFogGuardPlus, define_adjusted_deepFogGuardPlus
from KerasSingleLaneExperiment.loadData import load_data
from sklearn.model_selection import train_test_split
from KerasSingleLaneExperiment.FailureIteration import calculateExpectedAccuracy
from KerasSingleLaneExperiment.main import average
import keras.backend as K
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
    # nodewise survival rates for deepFogGuardPlus
    # elements of the vector are 1 - node-wise_dropout_rate
    nodewise_survival_rates = [
        [.95,.95,.95],
        [.9,.9,.9],
        [.7,.7,.7],
        [.5,.5,.5],
    ]
    hidden_units = 250
    batch_size = 1028
    load_model = False
    num_train_epochs = 25 
    # file name with the experiments accuracy output
    output_name = "results/health_nodewise_dropout.txt"
    num_iterations = 10
    verbose = 2
    # keep track of output so that output is in order
    output_list = []
    
    # convert survivability settings into strings so it can be used in the dictionary as keys
    no_failure = str(survivability_settings[0])
    normal = str(survivability_settings[1])
    poor = str(survivability_settings[2])
    hazardous = str(survivability_settings[3])
    
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
            deepFogGuardPlus_nodewise_dropout = define_deepFogGuardPlus(num_vars,num_classes,hidden_units,nodewise_survival_rate)
            # adjusted node_wise dropout
            deepFogGuardPlus_adjusted_nodewise_dropout_file = str(iteration) + " " + str(nodewise_survival_rate) + 'health_nodewise_dropout.h5'
            deepFogGuardPlus_adjusted_nodewise_dropout = define_adjusted_deepFogGuardPlus(num_vars,num_classes,hidden_units,nodewise_survival_rate)
            if load_model:
                deepFogGuardPlus_nodewise_dropout.load_weights(deepFogGuardPlus_nodewise_dropout_file)
                deepFogGuardPlus_adjusted_nodewise_dropout.load_weights(deepFogGuardPlus_nodewise_dropout_file)
            else:
                print("Training deepFogGuardPlus Node-wise Dropout")
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
            del deepFogGuardPlus_nodewise_dropout

    # calculate average accuracies for deepFogGuardPlus Node-wise Dropout
    for nodewise_survival_rate in nodewise_survival_rates:
        print(nodewise_survival_rate)
        for survivability_setting in survivability_settings:
            deepFogGuardPlus_nodewise_dropout_acc = average(output["deepFogGuardPlus Node-wise Dropout"][str(nodewise_survival_rate)][str(survivability_setting)])
            output_list.append(str(nodewise_survival_rate) + str(survivability_setting) + " deepFogGuardPlus Node-wise Dropout: " + str(deepFogGuardPlus_nodewise_dropout_acc) + '\n')
            print(nodewise_survival_rate,survivability_setting,"deepFogGuardPlus Node-wise Dropout:",deepFogGuardPlus_nodewise_dropout_acc)  

            deepFogGuardPlus_adjusted_nodewise_dropout_acc = average(output["deepFogGuardPlus Adjusted Node-wise Dropout"][str(nodewise_survival_rate)][str(survivability_setting)])
            output_list.append(str(nodewise_survival_rate) + str(survivability_setting) + " deepFogGuardPlus Adjusted Node-wise Dropout: " + str(deepFogGuardPlus_nodewise_dropout_acc) + '\n')
            print(nodewise_survival_rate,survivability_setting,"deepFogGuardPlus Adjusted Node-wise Dropout:",deepFogGuardPlus_nodewise_dropout_acc)  
    # write experiments output to file
    with open(output_name,'w') as file:
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(output_name))
    print(output)
