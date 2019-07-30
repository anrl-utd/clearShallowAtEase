
from KerasSingleLaneExperiment.deepFogGuardPlus import define_deepFogGuardPlus
from KerasSingleLaneExperiment.deepFogGuard import define_deepFogGuard
from KerasSingleLaneExperiment.Vanilla import define_vanilla_model
from KerasSingleLaneExperiment.loadData import load_data
from sklearn.model_selection import train_test_split
from KerasSingleLaneExperiment.FailureIteration import run
from KerasSingleLaneExperiment.main import average
import keras.backend as K
import datetime
import gc
import os

# TODO: to add dropout abalation to actual model

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
        "deepFogGuard Plus":
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        }, 
        "deepFogGuard":
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        },
        "Vanilla": 
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        },
        "deepFogGuard Weight Ablation": 
        {
            "[0.78, 0.8, 0.85]":[0] * num_iterations,
            "[0.87, 0.91, 0.95]":[0] * num_iterations,
            "[0.92, 0.96, 0.99]":[0] * num_iterations,
            "[1, 1, 1]":[0] * num_iterations,
        },
        "deepFogGuard Plus Ablation": 
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
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        for survive_configuration in survive_configurations:
            K.set_learning_phase(1)
            # create models

            # deepFogGuardPlus
            deepFogGuardPlus = define_deepFogGuardPlus(num_vars,num_classes,hidden_units,survive_configuration)
            deepFogGuardPlus_file = str(iteration) + " " + str(survive_configuration) + ' deepFogGuardPlus.h5'
            if load_model:
                deepFogGuardPlus.load_weights(deepFogGuardPlus_file)
            else:
                print("Training deepFogGuardPlus")
                deepFogGuardPlus.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
                deepFogGuardPlus.save_weights(deepFogGuardPlus_file)

            # deepFogGuard
            deepFogGuard = define_deepFogGuard(num_vars,num_classes,hidden_units,survive_configuration)
            deepFogGuard_file = str(iteration) + " " +str(survive_configuration) + ' deepFogGuard.h5'
            if load_model:
                deepFogGuard.load_weights(deepFogGuard_file)
            else:
                print("Training deepFogGuard")
                deepFogGuard.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
                deepFogGuard.save_weights(deepFogGuard_file)

            # deepFogGuard weight ablation
            deepFogGuard_weight_ablation = deepFogGuard(num_vars,num_classes,hidden_units,survive_configuration,isUnWeighted = False)
            deepFogGuard_weight_ablation_file = str(iteration) + " " + str(survive_configuration) + ' deepFogGuard_weight_ablation.h5'
            if load_model:
                deepFogGuard_weight_ablation.load_weights(deepFogGuard_weight_ablation_file)
            else:
                print("Training deepFogGuard Weight Ablation")
                deepFogGuard_weight_ablation.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
                deepFogGuard_weight_ablation.save_weights(deepFogGuard_weight_ablation_file)

            # vanilla model
            vanilla = define_vanilla_model(num_vars,num_classes,hidden_units)
            vanilla_file = str(iteration) + " " + str(survive_configuration) + ' vanilla.h5'
            if load_model:
                vanilla.load_weights(vanilla_file)
            else:
                print("Training vanilla")
                vanilla.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
                vanilla.save_weights(vanilla_file)

            # test models
            K.set_learning_phase(0)

            # survival configurations
            print(survive_configuration)
            output_list.append(str(survive_configuration) + '\n')

            # deepFogGuard Plus
            output_list.append('deepFogGuard Plus' + '\n')
            print("deepFogGuard Plus")
            output["deepFogGuard Plus"][str(survive_configuration)][iteration-1] = run(deepFogGuardPlus,survive_configuration,output_list,training_labels,test_data,test_labels)

            # deepFogGuard
            output_list.append('deepFogGuard' + '\n')
            print("deepFogGuard")
            output["deepFogGuard"][str(survive_configuration)][iteration-1] = run(deepFogGuard,survive_configuration,output_list,training_labels,test_data,test_labels)

            # deepFogGuard Weight Ablation
            output_list.append('deepFogGuard Weight Ablation' + '\n')
            print("deepFogGuard Weight Ablation")
            output["deepFogGuard Weight Ablation"][str(survive_configuration)][iteration-1] = run(deepFogGuard_weight_ablation,survive_configuration,output_list,training_labels,test_data,test_labels)

            # vanilla
            output_list.append('Vanilla' + '\n')                    
            print("Vanilla")
            output["Vanilla"][str(survive_configuration)][iteration-1] = run(vanilla,survive_configuration,output_list,training_labels,test_data,test_labels)

        # runs deepFogGuard Plus Ablation
        output_list.append('deepFogGuard Plus Ablation' + '\n')                  
        print("deepFogGuard Plus Ablation")
        for survive_configuration in activeguard_baseline_surviveconfigs:
            K.set_learning_phase(1)
            deepFogGuardPlus_Ablation_file = str(iteration) + " " + str(survive_configuration) + ' deepFogGuardPlus_Ablation.h5'
            deepFogGuardPlus_ablation = deepFogGuardPlus(num_vars,num_classes,hidden_units,survive_configuration)
            if load_model:
                deepFogGuardPlus_ablation.load_weights(deepFogGuardPlus_Ablation_file)
            else:
                print("Training deepFogGuard Plus Ablation")
                deepFogGuardPlus_ablation.fit(data,labels,epochs=10, batch_size=batch_size,verbose=verbose,shuffle = True)
                deepFogGuardPlus_ablation.save_weights(deepFogGuardPlus_Ablation_file)
                print("Test on normal survival rates")
                output_list.append("Test on normal survival rates" + '\n')
                for normal_survival_config in survive_configurations:
                    output_list.append(str(normal_survival_config)+ '\n')
                    output["deepFogGuard Plus Ablation"][str(survive_configuration)][str(normal_survival_config)][iteration-1] = run(deepFogGuardPlus_ablation,normal_survival_config,output_list,training_labels,test_data,test_labels)
        
            print(survive_configuration)
            output_list.append(str(survive_configuration)+ '\n')
            output["deepFogGuard Plus Ablation"][str(survive_configuration)][iteration-1] = run(deepFogGuardPlus_ablation,survive_configuration,output_list,training_labels,test_data,test_labels)
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del deepFogGuard
        del deepFogGuard_weight_ablation
        del deepFogGuardPlus
        del deepFogGuardPlus_ablation
        del vanilla
   # calculate average accuracies 
    for survive_configuration in survive_configurations:
        deepfogGuardPlus_acc = average(output["deepFogGuard Plus"][str(survive_configuration)])
        deepFogGuard_acc = average(output["deepFogGuard"][str(survive_configuration)])
        vanilla_acc = average(output["Vanilla"][str(survive_configuration)])
        deepFogGuard_WeightAblation_acc = average(output["deepFogGuard Weight Ablation"][str(survive_configuration)])

        output_list.append(str(survive_configuration) + " deepFogGuard Plus Accuracy: " + str(deepfogGuardPlus_acc) + '\n')
        output_list.append(str(survive_configuration) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
        output_list.append(str(survive_configuration) + " Vanilla Accuracy: " + str(vanilla_acc) + '\n')
        output_list.append(str(survive_configuration) + " deepFogGuard Weight Ablation: " + str(deepFogGuard_WeightAblation_acc) + '\n')

        print(str(survive_configuration),"deepFogGuard Plus Accuracy:",deepfogGuardPlus_acc)
        print(str(survive_configuration),"deepFogGuard Accuracy:",deepFogGuard_acc)
        print(str(survive_configuration),"Vanilla Accuracy:",vanilla_acc)
        print(str(survive_configuration),"deepFogGuard Weight Ablation:",deepFogGuard_WeightAblation_acc)

    # calculate average accuracies for deepFogGuard Plus Ablation
    for survive_config in activeguard_baseline_surviveconfigs:
        print(survive_config)
        for original_survive_config in survive_configurations:
            deepFogGuardPlus_Ablation_acc = average(output["deepFogGuard Plus Ablation"][str(survive_config)][str(original_survive_config)])
            output_list.append(str(survive_config) + str(original_survive_config) + " deepFogGuard Plus Ablation: " + str(deepFogGuardPlus_Ablation_acc) + '\n')
            print(survive_config,original_survive_config,"deepFogGuard Plus Ablation:",deepFogGuardPlus_Ablation_acc)  
    with open(output_name,'w') as file:
        file.write(str(output))
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    print(output)
