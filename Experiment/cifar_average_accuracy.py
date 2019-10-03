
from keras.applications.mobilenet import MobileNet
import keras.backend as K
import math
import os 
from Experiment.common_exp_methods_CNN import define_model
from Experiment.Accuracy import calculateExpectedAccuracy
from Experiment.common_exp_methods_CNN_cifar import init_data, init_common_experiment_params, get_model_weights_CNN_cifar 
from Experiment.common_exp_methods import average, make_output_dictionary_average_accuracy, write_n_upload, make_results_folder
import gc
import numpy as np

def define_and_train(iteration, model_name, load_model, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, default_failout_survival_rate, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus):
    K.set_learning_phase(1)
    model, model_file = define_model(iteration, model_name, "cifar", input_shape, classes, alpha, default_failout_survival_rate, strides)
    get_model_weights_CNN_cifar(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
    return model

if __name__ == "__main__":
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data() 

    num_iterations, classes, reliability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, use_GCP, alpha, input_shape, strides, num_gpus = init_common_experiment_params()
    
    default_failout_survival_rate = [.95,.95]
    train_steps_per_epoch = math.ceil(len(training_data) / batch_size)
    val_steps_per_epoch = math.ceil(len(val_data) / batch_size)

    output = make_output_dictionary_average_accuracy(reliability_settings, num_iterations)
    load_model = False
    
    make_results_folder()
    output_name = 'results' + '/cifar_average_accuracy_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        Vanilla = define_and_train(iteration, "Vanilla", load_model, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, default_failout_survival_rate, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
        deepFogGuard = define_and_train(iteration, "deepFogGuard", load_model, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, default_failout_survival_rate, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
        ResiliNet = define_and_train(iteration, "ResiliNet", load_model, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, strides, default_failout_survival_rate, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch, num_gpus)
        
        for reliability_setting in reliability_settings:
            output_list.append(str(reliability_setting) + '\n')
            print(reliability_setting)
            output["Vanilla"][str(reliability_setting)][iteration-1] = calculateExpectedAccuracy(Vanilla, reliability_setting,output_list, training_labels= training_labels, test_data= test_data, test_labels= test_labels)
            output["deepFogGuard"][str(reliability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuard, reliability_setting,output_list, training_labels= training_labels, test_data= test_data, test_labels= test_labels)
            output["ResiliNet"][str(reliability_setting)][iteration-1] = calculateExpectedAccuracy(ResiliNet, reliability_setting,output_list, training_labels= training_labels, test_data= test_data, test_labels= test_labels)
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del Vanilla
        del deepFogGuard
        del ResiliNet
   
    for reliability_setting in reliability_settings:
        output_list.append(str(reliability_setting) + '\n')

        Vanilla_acc = average(output["Vanilla"][str(reliability_setting)])
        deepFogGuard_acc = average(output["deepFogGuard"][str(reliability_setting)])
        ResiliNet_acc = average(output["ResiliNet"][str(reliability_setting)])

        output_list.append(str(reliability_setting) + " Vanilla Accuracy: " + str(Vanilla_acc) + '\n')
        output_list.append(str(reliability_setting) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
        output_list.append(str(reliability_setting) + " ResiliNet Accuracy: " + str(ResiliNet_acc) + '\n')

        print(str(reliability_setting),"Vanilla Accuracy:",Vanilla_acc)
        print(str(reliability_setting),"deepFogGuard Accuracy:",deepFogGuard_acc)
        print(str(reliability_setting),"ResiliNet Accuracy:",ResiliNet_acc)

        Vanilla_std = np.std(output["Vanilla"][str(reliability_setting)],ddof=1)
        deepFogGuard_std = np.std(output["deepFogGuard"][str(reliability_setting)],ddof=1)
        ResiliNet_std = np.std(output["ResiliNet"][str(reliability_setting)],ddof=1)

        output_list.append(str(reliability_setting) + " Vanilla std: " + str(Vanilla_std) + '\n')
        output_list.append(str(reliability_setting) + " deepFogGuard std: " + str(deepFogGuard_std) + '\n')
        output_list.append(str(reliability_setting) + " ResiliNet std: " + str(ResiliNet_std) + '\n')

        print(str(reliability_setting),"Vanilla std:",Vanilla_std)
        print(str(reliability_setting),"deepFogGuard std:",deepFogGuard_std)
        print(str(reliability_setting),"ResiliNet std:",ResiliNet_std)
    
    write_n_upload(output_name, output_list, use_GCP)
    print(output)
