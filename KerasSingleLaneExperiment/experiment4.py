from KerasSingleLaneExperiment.deepFogGuardPlus import define_deepFogGuardPlus
from KerasSingleLaneExperiment.deepFogGuard import define_deepFogGuard
from KerasSingleLaneExperiment.loadData import load_data
from sklearn.model_selection import train_test_split
from KerasSingleLaneExperiment.FailureIteration import run
from KerasSingleLaneExperiment.experiment2 import average
import keras.backend as K
import datetime
import os
import gc

# experiment with new active guard 
# do active guard results for everything (table 1, table 5, extra table at the end)

def normal_experiment():
    use_GCP = True
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/mHealth_complete.log ./')
    data,labels= load_data('mHealth_complete.log')
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .2, shuffle = True, stratify = labels)
    input_size = len(training_data[0])
    num_classes = 13
    hidden_units = 250
    batch_size = 1028
    verbose = 2
    output_list = []

    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    file_name = 'results/' + date + '/experiment4_normalexperiment_results.txt'
    survive_configurations = [
        [.78,.8,.85],
        [.87,.91,.95],
        [.92,.96,.99],
    ]
    num_iterations = 10
    output = {
        "Active Guard":
        {
            "[0.78, 0.8, 0.85]": [0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]": [0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        },
    }
    for iteration in range(1,num_iterations+1):
        for survive_configuration in survive_configurations:
            model = define_active_guard_model_with_connections_hyperconnectionweight1(input_size,num_classes,hidden_units,0,survive_configuration,[1,1,1])
            model.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
            output["Active Guard"][str(survive_configuration)][iteration-1] = run(" ",model,survive_configuration,output_list,training_labels,test_data,test_labels)
            #active_guard_file = str(iteration) + " " + str(survive_configuration) + ' active_guard.h5'
            #model.save_weights(active_guard_file)
            #os.system('gsutil -m -q cp -r %s gs://anrl-storage/models/fixed_activeguard' % active_guard_file)
            # clear session to remove old graphs from memory so that subsequent training is not slower
            K.clear_session()
            gc.collect()
            del model
        # no failure 
        # used dropout of .1
        model = define_active_guard_model_with_connections_hyperconnectionweight1(input_size,num_classes,hidden_units,0,[.9,.9,.9],[1,1,1])
        model.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
        output["Active Guard"][str([1,1,1])][iteration-1] = run(" ",model,survive_configuration,output_list,training_labels,test_data,test_labels)
        #active_guard_file = str(iteration) + " [1,1,1]" + ' baseline_active_guard.h5'
        #model.save_weights(active_guard_file)
        #os.system('gsutil -m -q cp -r %s gs://anrl-storage/models/fixed_activeguard' % active_guard_file)
     # write average accuracies to a file 
    with open(file_name,'a+') as file:
        for survive_configuration in survive_configurations:
            output_list.append(str(survive_configuration) + '\n')
            active_guard_acc = average(output["Active Guard"][str(survive_configuration)])
            output_list.append(str(survive_configuration) + " ActiveGuard Accuracy: " + str(active_guard_acc) + '\n')
            print(str(survive_configuration),"ActiveGuard Accuracy:",active_guard_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))
def baseline_experiment():
    use_GCP = True
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/mHealth_complete.log ./')
   
    data,labels= load_data('mHealth_complete.log')
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .2, shuffle = True, stratify = labels)
    input_size = len(training_data[0])
    num_classes = 13
    hidden_units = 250
    survive_rates = [.92,.96,.99]
    model = define_active_guard_model_with_connections_hyperconnectionweight1(input_size,num_classes,hidden_units,0,survive_rates,[1,1,1])
    batch_size = 1028
    verbose = 2
    output_list = []

    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    file_name = 'results/' + date + '/experiment4_baselineexperiment_nofailure_results.txt'
    # make folder for outputs 
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    survive_configurations = [
        [.78,.8,.85],
        [.87,.91,.95],
        [.92,.96,.99],
        [1,1,1]
    ]
    baseline = [
        [.9,.9,.9],
        [.7,.7,.7],
        [.5,.5,.5]
    ]
    num_iterations = 10
    output = {
        "Baseline Active Guard":
        {
            "[0.9, 0.9, 0.9]":
            {
                "[0.78, 0.8, 0.85]":[0] * num_iterations,
                "[0.87, 0.91, 0.95]":[0] * num_iterations,
                "[0.92, 0.96, 0.99]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations,
            },
            "[0.7, 0.7, 0.7]":
            {
                "[0.78, 0.8, 0.85]":[0] * num_iterations,
                "[0.87, 0.91, 0.95]":[0] * num_iterations,
                "[0.92, 0.96, 0.99]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations,
            },
            "[0.5, 0.5, 0.5]":
            {
                "[0.78, 0.8, 0.85]":[0] * num_iterations,
                "[0.87, 0.91, 0.95]":[0] * num_iterations,
                "[0.92, 0.96, 0.99]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations,
            },
        },
    }
    for iteration in range(1,num_iterations+1):
        for survive_configuration in baseline:
            model = define_active_guard_model_with_connections_hyperconnectionweight1(input_size,num_classes,hidden_units,0,survive_configuration,[1,1,1])
            model.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
            for normal_survival_config in survive_configurations:
                output["Baseline Active Guard"][str(survive_configuration)][str(normal_survival_config)][iteration-1] = run(" ",model,normal_survival_config,output_list,training_labels,test_data,test_labels)
                active_guard_file = str(iteration) + " " + str(survive_configuration) + str(normal_survival_config) + ' baselineactive_guard.h5'
                model.save_weights(active_guard_file)
                #os.system('gsutil -m -q cp -r %s gs://anrl-storage/models/fixed_activeguard' % active_guard_file)
            # clear session to remove old graphs from memory so that subsequent training is not slower
            K.clear_session()
            gc.collect()
            del model
    
     # write average accuracies to a file 
    with open(file_name,'a+') as file:
        for survive_config in baseline:
            print(survive_config)
            #file.write(str(survive_config) + '\n')  
            output_list.append(str(survive_config) + '\n')
            for original_survive_config in survive_configurations:
                #file.write(str(original_survive_config) + '\n')  
                baseline_active_guard_acc = average(output["Baseline Active Guard"][str(survive_config)][str(original_survive_config)])
                #file.write(str(baseline_active_guard_acc) + '\n')  
                output_list.append(str(survive_config) + str(original_survive_config) + " Baseline ActiveGuard Accuracy: " + str(baseline_active_guard_acc) + '\n')
                print(survive_config,original_survive_config,"Baseline ActiveGuard Accuracy:",baseline_active_guard_acc)  
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))
if __name__ == "__main__":
    normal_experiment()
    #baseline_experiment()