
from Experiment.common_exp_methods_CNN import define_model
from Experiment.Accuracy import calculateExpectedAccuracy
from Experiment.common_exp_methods_CNN_imagenet import init_data, init_common_experiment_params, get_model_weights_CNN_imagenet
from Experiment.common_exp_methods import average, convert_to_string, make_output_dictionary_average_accuracy, make_results_folder,write_n_upload
import keras.backend as K
import datetime
import gc
import os
import numpy as np

import tensorflow as tf
def define_and_train(iteration, model_name, load_model, train_generator, val_generator, input_shape, classes, alpha,num_train_examples, epochs,num_gpus, strides, num_workers):
    K.set_learning_phase(1)
    model, parallel_model, model_file = define_model(iteration, model_name, "imagenet", input_shape, classes, alpha, strides, num_gpus)
    model = get_model_weights_CNN_imagenet(model, parallel_model, model_name, load_model, model_file, train_generator, val_generator,num_train_examples,epochs, num_gpus, num_workers)
    return model

def calc_accuracy(iteration, model_name, model, reliability_setting, output_list,test_generator, num_test_examples):
    output_list.append(model_name + "\n")
    print(model_name)
    output[model_name][str(reliability_setting)][iteration-1] = calculateExpectedAccuracy(model,reliability_setting,output_list,test_generator= test_generator,num_test_examples = num_test_examples)


# runs all 3 failure configurations for all 3 models
if __name__ == "__main__":

    use_GCP = False
    num_iterations,num_train_examples,num_test_examples, reliability_settings, input_shape, num_classes, alpha, epochs, num_gpus, strides, num_workers = init_common_experiment_params()
    train_generator, test_generator = init_data(use_GCP, num_gpus) 
    
    load_model = False
    num_iterations = 3
    make_results_folder()
    output_name = 'results' + '/imagenet_average_accuracy_results.txt'
    output_list = []
    
    output = make_output_dictionary_average_accuracy(reliability_settings, num_iterations)

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
            num_train_examples = num_train_examples,
            epochs = epochs,
            num_gpus = num_gpus,
            strides = strides,
            num_workers = num_workers
            )
        # deepFogGuard = define_and_train(
        #     iteration = iteration, 
        #     model_name = "deepFogGuard", 
        #     load_model = load_model, 
        #     train_generator = train_generator, 
        #     val_generator = val_generator, 
        #     input_shape = input_shape, 
        #     classes = num_classes, 
        #     alpha = alpha, 
        #     num_train_examples = num_train_examples,
        #     epochs = epochs,
        #     num_gpus = num_gpus,
        #     strides = strides,
        #     num_workers = num_workers
        #     )
        # Vanilla = define_and_train(
        #     iteration = iteration, 
        #     model_name = "Vanilla", 
        #     load_model = load_model, 
        #     train_generator = train_generator, 
        #     val_generator = val_generator, 
        #     input_shape = input_shape, 
        #     classes = num_classes, 
        #     alpha = alpha, 
        #     num_train_examples = num_train_examples,
        #     epochs = epochs,
        #     num_gpus = num_gpus,
        #     strides = strides,
        #     num_workers = num_workers
        #     )
        # test models
        for reliability_setting in reliability_settings:
            calc_accuracy(iteration, "ResiliNet", ResiliNet, reliability_setting, output_list,test_generator, num_test_examples)
            # calc_accuracy(iteration, "deepFogGuard", deepFogGuard, reliability_setting, output_list,test_generator, num_test_examples)
            # calc_accuracy(iteration, "Vanilla", Vanilla, reliability_setting, output_list,test_generator, num_test_examples)
            
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        # del deepFogGuard
        del ResiliNet
        # del Vanilla
   # calculate average accuracies from all expected accuracies
    for reliability_setting in reliability_settings:
        ResiliNet_acc = average(output["ResiliNet"][str(reliability_setting)])
        # deepFogGuard_acc = average(output["deepFogGuard"][str(reliability_setting)])
        # Vanilla_acc = average(output["Vanilla"][str(reliability_setting)])

        ResiliNet_std = np.std(output["ResiliNet"][str(reliability_setting)],ddof=1)
        # deepFogGuard_std = np.std(output["deepFogGuard"][str(reliability_setting)],ddof = 1)
        # Vanilla_std = np.std(output["Vanilla"][str(reliability_setting)],ddof = 1)

        output_list.append(str(reliability_setting) + " ResiliNet Accuracy: " + str(ResiliNet_acc) + '\n')
        # output_list.append(str(reliability_setting) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
        # output_list.append(str(reliability_setting) + " Vanilla Accuracy: " + str(Vanilla_acc) + '\n')

        output_list.append(str(reliability_setting) + " ResiliNet std: " + str(ResiliNet_std) + '\n')
        # output_list.append(str(reliability_setting) + " deepFogGuard std: " + str(deepFogGuard_std) + '\n')
        # output_list.append(str(reliability_setting) + " Vanilla std: " + str(Vanilla_std) + '\n')

        print(str(reliability_setting),"ResiliNet Accuracy:",ResiliNet_acc)
        # print(str(reliability_setting),"deepFogGuard Accuracy:",deepFogGuard_acc)
        # print(str(reliability_setting),"Vanilla Accuracy:",Vanilla_acc)

        print(str(reliability_setting),"ResiliNet std:",ResiliNet_std)
        # print(str(reliability_setting),"deepFogGuard std:",deepFogGuard_std)
        # print(str(reliability_setting),"Vanilla std:",Vanilla_std)
    write_n_upload(output_name, output_list, use_GCP)
