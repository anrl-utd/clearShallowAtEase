from KerasSingleLaneExperiment.deepFogGuardPlus import define_deepFogGuardPlus
from KerasSingleLaneExperiment.deepFogGuard import define_deepFogGuard
from KerasSingleLaneExperiment.loadData import load_data
from sklearn.model_selection import train_test_split
from KerasSingleLaneExperiment.FailureIteration import run
import keras.backend as K
import datetime
import os
import gc 

# runs all hyperconnection configurations for both fixed and active guard survival configurations
# sensitivity analysis 
def main():
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
    hyperconnections = [
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [1,1,1],
    ]
    hidden_units = 250
    batch_size = 1028
    load_model = False
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    file_name = 'results/' + date + '/experiment2_results16_20.txt'
    num_iterations = 5
    verbose = 0
    # keep track of output so that output is in order
    output_list = []
    # dictionary to store all the results
    output = {
        "Active Guard":
        {
            "[0.78, 0.8, 0.85]":
            {
                "[0, 0, 0]":[0] * num_iterations,
                "[1, 0, 0]":[0] * num_iterations,
                "[0, 1, 0]":[0] * num_iterations,
                "[0, 0, 1]":[0] * num_iterations,
                "[1, 1, 0]":[0] * num_iterations,
                "[1, 0, 1]":[0] * num_iterations,
                "[0, 1, 1]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations
            },
            "[0.87, 0.91, 0.95]":
            {
                "[0, 0, 0]":[0] * num_iterations,
                "[1, 0, 0]":[0] * num_iterations,
                "[0, 1, 0]":[0] * num_iterations,
                "[0, 0, 1]":[0] * num_iterations,
                "[1, 1, 0]":[0] * num_iterations,
                "[1, 0, 1]":[0] * num_iterations,
                "[0, 1, 1]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations
            },
            "[0.92, 0.96, 0.99]":
            {
                "[0, 0, 0]":[0] * num_iterations,
                "[1, 0, 0]":[0] * num_iterations,
                "[0, 1, 0]":[0] * num_iterations,
                "[0, 0, 1]":[0] * num_iterations,
                "[1, 1, 0]":[0] * num_iterations,
                "[1, 0, 1]":[0] * num_iterations,
                "[0, 1, 1]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations
            },
        }, 
        "Fixed Guard":
        {
            "[0.78, 0.8, 0.85]":
            {
                "[0, 0, 0]":[0] * num_iterations,
                "[1, 0, 0]":[0] * num_iterations,
                "[0, 1, 0]":[0] * num_iterations,
                "[0, 0, 1]":[0] * num_iterations,
                "[1, 1, 0]":[0] * num_iterations,
                "[1, 0, 1]":[0] * num_iterations,
                "[0, 1, 1]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations
            },
            "[0.87, 0.91, 0.95]":
            {
                "[0, 0, 0]":[0] * num_iterations,
                "[1, 0, 0]":[0] * num_iterations,
                "[0, 1, 0]":[0] * num_iterations,
                "[0, 0, 1]":[0] * num_iterations,
                "[1, 1, 0]":[0] * num_iterations,
                "[1, 0, 1]":[0] * num_iterations,
                "[0, 1, 1]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations
            },
            "[0.92, 0.96, 0.99]":
            {
                "[0, 0, 0]":[0] * num_iterations,
                "[1, 0, 0]":[0] * num_iterations,
                "[0, 1, 0]":[0] * num_iterations,
                "[0, 0, 1]":[0] * num_iterations,
                "[1, 1, 0]":[0] * num_iterations,
                "[1, 0, 1]":[0] * num_iterations,
                "[0, 1, 1]":[0] * num_iterations,
                "[1, 1, 1]":[0] * num_iterations
            },
        }
    }
    # make folder for outputs 
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    for iteration in range(1,num_iterations+1):   
        # with open(file_name,'a+') as file:
        #     file.write('ITERATION ' + str(iteration) +  '\n')
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        for survive_configuration in survive_configurations:
            K.set_learning_phase(1)
            # create models

            for hyperconnection in hyperconnections:
                # active guard
                active_guard = define_active_guard_model_with_connections_hyperconnectionweight1(num_vars,num_classes,hidden_units,0,survive_configuration,hyperconnection)
                active_guard_file = str(iteration) + " " + str(survive_configuration) + str(hyperconnection) + ' active_guard.h5'
                if load_model:
                    active_guard.load_weights(active_guard_file)
                else:
                    print("Training active guard")
                    active_guard.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)

                # fixed guard
                fixed_guard = define_fixed_guard_model_experiment2(num_vars,num_classes,hidden_units,0,hyperconnection)
                fixed_guard_file = str(iteration) + " " +str(survive_configuration) + str(hyperconnection) +  ' fixed_guard.h5'
                if load_model:
                    fixed_guard.load_weights(fixed_guard_file)
                else:
                    print("Training fixed guard")
                    fixed_guard.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
                    #fixed_guard.save_weights(fixed_guard_file)

                # test models
                K.set_learning_phase(0)

                # write results to a file 
                # with open(file_name,'a+') as file:
                # survival configurations
                print(survive_configuration)
                # file.write(str(survive_configuration) + '\n')
                # file.write(str(hyperconnection) + '\n')

                # active guard
                #file.write('ACTIVE GUARD' + '\n')
                output_list.append('ACTIVE GUARD' + '\n')
                print("ACTIVE GUARD")
                output["Active Guard"][str(survive_configuration)][str(hyperconnection)][iteration-1] = run(file_name,active_guard,survive_configuration,output_list,training_labels,test_data,test_labels)
                    # fixed guard
                #file.write('FIXED GUARD' + '\n')
                output_list.append('FIXED GUARD' + '\n')
                print("FIXED GUARD")
                output["Fixed Guard"][str(survive_configuration)][str(hyperconnection)][iteration-1] = run(file_name,fixed_guard,survive_configuration,output_list,training_labels,test_data,test_labels)

                # clear session to remove old graphs from memory so that subsequent training is not slower
                K.clear_session()
                gc.collect()
                del active_guard
                del fixed_guard

   # write average accuracies to a file 
    with open(file_name,'a+') as file:
        for survive_configuration in survive_configurations:
            output_list.append(str(survive_configuration) + '\n')
            for hyperconnection in hyperconnections:
                output_list.append(str(hyperconnection) + '\n')
                active_guard_acc = average(output["Active Guard"][str(survive_configuration)][str(hyperconnection)])
                fixed_guard_acc = average(output["Fixed Guard"][str(survive_configuration)][str(hyperconnection)])
            

                # file.write(str(active_guard_acc) + '\n')
                # file.write(str(fixed_guard_acc) + '\n')

                output_list.append(str(survive_configuration) + " " + str(hyperconnection) + " ActiveGuard Accuracy: " + str(active_guard_acc) + '\n')
                output_list.append(str(survive_configuration) + " " + str(hyperconnection) + " FixedGuard Accuracy: " + str(fixed_guard_acc) + '\n')

                print(str(survive_configuration),str(hyperconnection),"ActiveGuard Accuracy:",active_guard_acc)
                print(str(survive_configuration),str(hyperconnection),"FixedGuard Accuracy:",fixed_guard_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    print(output)
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))

if __name__ == "__main__":
    main()