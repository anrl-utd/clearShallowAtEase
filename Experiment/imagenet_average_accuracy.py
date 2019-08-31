
from Experiment.common_exp_methods_CNN import define_model
from Experiment.FailureIterationFromImageDataGenerator import calculateExpectedAccuracyFromImageGenerator
from Experiment.utility import average, get_model_weights_MLP_camera
from Experiment.imagenet_common_exp_methods import init_data, init_common_experiment_params
from Experiment.utility import get_model_weights_CNN_imagenet
from Experiment.common_exp_methods import convert_to_string, make_output_dictionary_average_accuracy, make_results_folder,write_n_upload
import keras.backend as K
import datetime
import gc
import os
import numpy as np

def define_and_train(iteration, model_name, load_model, train_generator, val_generator, input_shape, classes, alpha, default_failout_survival_rate,num_train_examples, epochs):
    model, model_file = define_model(iteration, model_name, "imagenet", input_shape, classes, alpha, default_failout_survival_rate)
    get_model_weights_CNN_imagenet(model, model_name, load_model, model_file, train_generator, val_generator,num_train_examples,epochs)
    return model

def calc_accuracy(iteration, model_name, model, survivability_setting, output_list,test_generator, num_test_examples):
    output_list.append(model_name + "\n")
    print(model_name)
    output[model_name][str(survivability_setting)][iteration-1] = calculateExpectedAccuracyFromImageGenerator(model,survivability_setting,output_list,test_generator,num_test_examples = num_test_examples)


# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":
    use_GCP = False
    train_generator, test_generator = init_data(use_GCP) 
    num_iterations,num_train_examples,num_test_examples, survivability_settings, input_shape, num_classes, alpha, epochs = init_common_experiment_params()

    default_failout_survival_rate = [.95,.95,.95]
    load_model = False
    num_iterations = 1
    make_results_folder()
    output_name = 'results' + '/imagenet_average_accuracy_results.txt'
    output_list = []
    
    output = make_output_dictionary_average_accuracy(survivability_settings, num_iterations)

    val_generator = None

    for iteration in range(1,num_iterations+1):   
        output_list.append('ITERATION ' + str(iteration) +  '\n')
        print("ITERATION ", iteration)
        ResiliNet = define_and_train(
            iteration = iteration, 
            model_name = "ResiliNet", 
            load_model = load_model, 
            train_generator = train_generator, 
            val_generator = val_generator, 
            input_shape = input_shape, 
            classes = num_classes, 
            alpha = alpha, 
            default_failout_survival_rate = default_failout_survival_rate,
            num_train_examples = num_train_examples,
            epochs = epochs
            )
        deepFogGuard = define_and_train(
            iteration = iteration, 
            model_name = "deepFogGuard", 
            load_model = load_model, 
            train_generator = train_generator, 
            val_generator = val_generator, 
            input_shape = input_shape, 
            classes = num_classes, 
            alpha = alpha, 
            default_failout_survival_rate = None,
            num_train_examples = num_train_examples,
            epochs = epochs
            )
        Vanilla = define_and_train(
            iteration = iteration, 
            model_name = "Vanilla", 
            load_model = load_model, 
            train_generator = train_generator, 
            val_generator = val_generator, 
            input_shape = input_shape, 
            classes = num_classes, 
            alpha = alpha, 
            default_failout_survival_rate = None,
            num_train_examples = num_train_examples,
            epochs = epochs
            )
 
        # test models
        for survivability_setting in survivability_settings:
            calc_accuracy(iteration, "ResiliNet", ResiliNet, survivability_setting, output_list,test_generator, num_test_examples)
            calc_accuracy(iteration, "deepFogGuard", deepFogGuard, survivability_setting, output_list,test_generator, num_test_examples)
            calc_accuracy(iteration, "Vanilla", Vanilla, survivability_setting, output_list,test_generator, num_test_examples)
            
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del deepFogGuard
        del ResiliNet
        del Vanilla
   # calculate average accuracies from all expected accuracies
    for survivability_setting in survivability_settings:
        ResiliNet_acc = average(output["ResiliNet"][str(survivability_setting)])
        deepFogGuard_acc = average(output["deepFogGuard"][str(survivability_setting)])
        Vanilla_acc = average(output["Vanilla"][str(survivability_setting)])

        ResiliNet_std = np.std(output["ResiliNet"][str(survivability_setting)],ddof=1)
        deepFogGuard_std = np.std(output["deepFogGuard"][str(survivability_setting)],ddof = 1)
        Vanilla_std = np.std(output["Vanilla"][str(survivability_setting)],ddof = 1)

        output_list.append(str(survivability_setting) + " ResiliNet Accuracy: " + str(ResiliNet_acc) + '\n')
        output_list.append(str(survivability_setting) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
        output_list.append(str(survivability_setting) + " Vanilla Accuracy: " + str(Vanilla_acc) + '\n')

        output_list.append(str(survivability_setting) + " ResiliNet STD: " + str(ResiliNet_std) + '\n')
        output_list.append(str(survivability_setting) + " deepFogGuard STD: " + str(deepFogGuard_std) + '\n')
        output_list.append(str(survivability_setting) + " Vanilla STD: " + str(Vanilla_std) + '\n')

        print(str(survivability_setting),"ResiliNet Accuracy:",ResiliNet_acc)
        print(str(survivability_setting),"deepFogGuard Accuracy:",deepFogGuard_acc)
        print(str(survivability_setting),"Vanilla Accuracy:",Vanilla_acc)

        print(str(survivability_setting),"ResiliNet std:",ResiliNet_std)
        print(str(survivability_setting),"deepFogGuard std:",deepFogGuard_std)
        print(str(survivability_setting),"Vanilla std:",Vanilla_std)
        
    write_n_upload(output_name, output_list, use_GCP)
