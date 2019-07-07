
from experiment.ActiveGuard import define_active_guard_model_with_connections
from experiment.FixedGuard import define_model_with_nofogbatchnorm_connections_extrainput,define_fixed_guard_baseline_model
from experiment.Baseline import define_baseline_functional_model
from experiment.loadData import load_data
from sklearn.model_selection import train_test_split
from experiment.FailureIteration import run
import keras.backend as K
import datetime
import os

# function to return average of a list 
def average(list):
    return sum(list) / len(list)
# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    use_GCP = True
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/mHealth_complete.log ./')
        os.mkdir('models/')
        os.mkdir('models/no he_normal')
    data,labels= load_data('mHealth_complete.log')
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .2, shuffle = True,stratify = labels)
    num_vars = len(training_data[0])
    num_classes = 13
    survive_configurations = [
        [.78,.8,.85],
        [.87,.91,.95],
        [.92,.96,.99],
        [1,1,1]
    ]
    # survibility configurations for active guard basleline
    activeguard_baseline_surviveconfigs = [
        [.9,.9,.9],
        [.7,.7,.7],
        [.5,.5,.5],
    ]
    hidden_units = 250
    batch_size = 1028
    load_model = False
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    file_name = 'results/' + date + '/experiment_no_failure_results10.txt'
    output_name = "experiment_no_failure_output10.txt"
    num_iterations = 1
    verbose = 0
    # keep track of output so that output is in order
    output_list = []
    # dictionary to store all the results
    output = {
        "Active Guard":
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        }, 
        "Fixed Guard":
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        },
        "Baseline": 
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        },
        "Baseline Fixed Guard": 
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        },
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
        }
    }
    # make folder for outputs 
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    for iteration in range(1,num_iterations+1):   
        with open(file_name,'a+') as file:
            output_list.append('ITERATION ' + str(iteration) +  '\n')
            file.write('ITERATION ' + str(iteration) +  '\n')
            print("ITERATION ", iteration)
        for survive_configuration in survive_configurations:
            K.set_learning_phase(1)
            # create models

            # active guard
            active_guard = define_active_guard_model_with_connections(num_vars,num_classes,hidden_units,0,survive_configuration)
            active_guard_file = str(iteration) + " " + str(survive_configuration) + ' active_guard.h5'
            if load_model:
                active_guard.load_weights(active_guard_file)
            else:
                print("Training active guard")
                active_guard.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
                #active_guard.save_weights(active_guard_file)

            # fixed guard
            fixed_guard = define_model_with_nofogbatchnorm_connections_extrainput(num_vars,num_classes,hidden_units,0,survive_configuration)
            fixed_guard_file = str(iteration) + " " +str(survive_configuration) + ' fixed_guard.h5'
            if load_model:
                fixed_guard.load_weights(fixed_guard_file)
            else:
                print("Training fixed guard")
                fixed_guard.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
                #fixed_guard.save_weights(fixed_guard_file)

            # fixed guard baseline
            baseline_fixed_guard = define_fixed_guard_baseline_model(num_vars,num_classes,hidden_units,0,survive_configuration)
            baseline_fixed_guard_file = str(iteration) + " " + str(survive_configuration) + ' baseline_fixed_guard.h5'
            if load_model:
                baseline_fixed_guard.load_weights(baseline_fixed_guard_file)
            else:
                print("Training fixed guard baseline")
                baseline_fixed_guard.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
                #baseline_fixed_guard.save_weights(baseline_fixed_guard_file)

            # baseline model
            baseline = define_baseline_functional_model(num_vars,num_classes,hidden_units,0)
            baseline_file = str(iteration) + " " + str(survive_configuration) + ' baseline.h5'
            if load_model:
                baseline.load_weights(baseline_file)
            else:
                print("Training baseline")
                baseline.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
                #baseline.save_weights(baseline_file)

            # test models
            K.set_learning_phase(0)

            # write results to a file 
            with open(file_name,'a+') as file:
                # survival configurations
                print(survive_configuration)
                file.write(str(survive_configuration) + '\n')

                # active guard
                file.write('ACTIVE GUARD' + '\n')
                output_list.append('ACTIVE GUARD' + '\n')
                print("ACTIVE GUARD")
                output["Active Guard"][str(survive_configuration)][iteration-1] = run(file_name,active_guard,survive_configuration,output_list,training_labels,test_data,test_labels)

                # fixed guard
                file.write('FIXED GUARD' + '\n')
                output_list.append('FIXED GUARD' + '\n')
                print("FIXED GUARD")
                output["Fixed Guard"][str(survive_configuration)][iteration-1] = run(file_name,fixed_guard,survive_configuration,output_list,training_labels,test_data,test_labels)

                # baseline fixed guard
                file.write('BASELINE FIXED GUARD' + '\n')
                output_list.append('BASELINE FIXED GUARD' + '\n')
                print("BASELINE FIXED GUARD")
                output["Baseline Fixed Guard"][str(survive_configuration)][iteration-1] = run(file_name,baseline_fixed_guard,survive_configuration,output_list,training_labels,test_data,test_labels)

                # baseline
                file.write('BASELINE' + '\n')
                output_list.append('BASELINE' + '\n')                    
                print("BASELINE")
                output["Baseline"][str(survive_configuration)][iteration-1] = run(file_name,baseline,survive_configuration,output_list,training_labels,test_data,test_labels)

            # save files to GCP
            # if use_GCP:
            #     #os.system('gsutil -m cp -r *.h5 gs://anrl-storage/models')
            #     # os.system('gsutil -m -q cp -r %s gs://anrl-storage/models' % active_guard_file)
            #     # os.system('gsutil -m -q cp -r %s gs://anrl-storage/models' % fixed_guard_file)
            #     # os.system('gsutil -m -q cp -r %s gs://anrl-storage/models' % baseline_fixed_guard_file)
            #     # os.system('gsutil -m -q cp -r %s gs://anrl-storage/models' % baseline_file)

        # runs baseline for active guard
        with open(file_name,'a+') as file:
            file.write('ACTIVE GUARD BASELINE' + '\n')
        output_list.append('ACTIVE GUARD BASELINE' + '\n')                  
        print("ACTIVE GUARD BASELINE")
        for survive_configuration in activeguard_baseline_surviveconfigs:
            K.set_learning_phase(1)
            baseline_activeguard_file = str(iteration) + " " + str(survive_configuration) + ' baseline_active_guard.h5'
            baseline_active_guard_model = define_active_guard_model_with_connections(num_vars,num_classes,hidden_units,0,survive_configuration)
            if load_model:
                baseline_active_guard_model.load_weights(baseline_activeguard_file)
            else:
                print("Training active guard baseline")
                baseline_active_guard_model.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
                baseline_active_guard_model.save_weights(baseline_activeguard_file)
                # load trained weights and change active guard weights to normal
                print("Test on normal survival rates")
                output_list.append("Test on normal survival rates" + '\n')
                for normal_survival_config in survive_configurations:
                    output_list.append(str(normal_survival_config)+ '\n')
                    output["Baseline Active Guard"][str(survive_configuration)][str(normal_survival_config)][iteration-1] = run(file_name,baseline_active_guard_model,normal_survival_config,output_list,training_labels,test_data,test_labels)
                    # test model 
                # if use_GCP:
                #      os.system('gsutil -om -q cp -r %s gs://anrl-storage/models' % baseline_activeguard_file)
            # write results to a file 
            with open(file_name,'a+') as file:
                print(survive_configuration)
                file.write(str(survive_configuration)+ '\n')
                output_list.append(str(survive_configuration)+ '\n')
                output["Baseline Active Guard"][str(survive_configuration)][iteration-1] = run(file_name,baseline_active_guard_model,survive_configuration,output_list,training_labels,test_data,test_labels)

   # write average accuracies to a file 
    with open(file_name,'a+') as file:
        for survive_configuration in survive_configurations:
            active_guard_acc = average(output["Active Guard"][str(survive_configuration)])
            fixed_guard_acc = average(output["Fixed Guard"][str(survive_configuration)])
            baseline_acc = average(output["Baseline"][str(survive_configuration)])
            baseline_fixed_guard_acc = average(output["Baseline Fixed Guard"][str(survive_configuration)])

            file.write(str(active_guard_acc) + '\n')
            file.write(str(fixed_guard_acc) + '\n')
            file.write(str(baseline_acc) + '\n')         
            file.write(str(baseline_fixed_guard_acc) + '\n')

            output_list.append(str(survive_configuration) + " ActiveGuard Accuracy: " + str(active_guard_acc) + '\n')
            output_list.append(str(survive_configuration) + " FixedGuard Accuracy: " + str(fixed_guard_acc) + '\n')
            output_list.append(str(survive_configuration) + " Baseline Accuracy: " + str(baseline_acc) + '\n')
            output_list.append(str(survive_configuration) + " Baseline FixedGuard Accuracy: " + str(baseline_fixed_guard_acc) + '\n')

            print(str(survive_configuration),"ActiveGuard Accuracy:",active_guard_acc)
            print(str(survive_configuration),"FixedGuard Accuracy:",fixed_guard_acc)
            print(str(survive_configuration),"Baseline Accuracy:",baseline_acc)
            print(str(survive_configuration),"Baseline FixedGuard Accuracy:",baseline_fixed_guard_acc)

        for survive_config in activeguard_baseline_surviveconfigs:
            print(survive_config)
            file.write(str(survive_config) + '\n')  
            for original_survive_config in survive_configurations:
                file.write(str(original_survive_config) + '\n')  
                baseline_active_guard_acc = average(output["Baseline Active Guard"][str(survive_config)][str(original_survive_config)])
                file.write(str(baseline_active_guard_acc) + '\n')  
                output_list.append(str(survive_config) + str(original_survive_config) + " Baseline ActiveGuard Accuracy: " + str(baseline_active_guard_acc) + '\n')
                print(survive_config,original_survive_config,"Baseline ActiveGuard Accuracy:",baseline_active_guard_acc)  
        file.flush()
        os.fsync(file)
    with open(output_name,'w') as file:
        file.write(str(output))
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    print(output)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(output_name))