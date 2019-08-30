
from keras.applications.mobilenet import MobileNet
import keras.backend as K
import math
import os 
from Experiment.cnn_Vanilla import define_vanilla_model_CNN
from Experiment.cnn_deepFogGuard import define_deepFogGuard_CNN
from Experiment.cnn_deepFogGuardPlus import define_deepFogGuardPlus_CNN
from Experiment.FailureIteration import calculateExpectedAccuracy
from Experiment.cifar_common_exp_methods import init_data, init_common_experiment_params 
from Experiment.utility import average, get_model_weights_CNN
from Experiment.common_exp_methods import make_output_dictionary_average_accuracy
import gc

def define_and_train(iteration, model_name, load_model, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, default_failout_survival_rate, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch):
    # ResiliNet
    if model_name == "ResiliNet":
        model = define_deepFogGuardPlus_CNN(classes=classes,input_shape = input_shape,alpha = alpha,survivability_setting=default_failout_survival_rate)
        model_file = "deepFogGuardPlus_cifar_average_accuracy" + str(iteration) + ".h5"
    # deepFogGuard
    if model_name == "deepFogGuard":
        model = define_deepFogGuard_CNN(classes=classes,input_shape = input_shape,alpha = alpha)
        model_file = "deepFogGuard_cifar_average_accuracy" + str(iteration) + ".h5"
    # Vanilla model
    if model_name == "Vanilla":
        model = define_vanilla_model_CNN(classes=classes,input_shape = input_shape,alpha = alpha)
        model_file = "vanilla_cifar_average_accuracy" + str(iteration) + ".h5"
    
    get_model_weights_CNN(model, model_name, load_model, model_file, training_data, training_labels, val_data, val_labels, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch)
    return model

if __name__ == "__main__":
    training_data, test_data, training_labels, test_labels, val_data, val_labels = init_data() 

    num_iterations, classes, survivability_settings, train_datagen, batch_size, epochs, progress_verbose, checkpoint_verbose, use_GCP, alpha, input_shape = init_common_experiment_params()
    
    default_failout_survival_rate = [.95,.95]
    train_steps_per_epoch = math.ceil(len(training_data) / batch_size)
    val_steps_per_epoch = math.ceil(len(val_data) / batch_size)

    output = make_output_dictionary_average_accuracy(survivability_settings, num_iterations)
    load_model = False
    
    # make folder for outputs 
    if not os.path.exists('results/' ):
        os.mkdir('results/' )
    if not os.path.exists('models'):      
        os.mkdir('models/')
    file_name = 'results' + '/cifar_average_accuracy_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        Vanilla = define_and_train(iteration, "Vanilla", load_model, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, default_failout_survival_rate, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch)
        deepFogGuard = define_and_train(iteration,"deepFogGuard", model_name, load_model, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, default_failout_survival_rate, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch)
        ResiliNet = define_and_train(iteration, "ResiliNet", load_model, training_data, training_labels, val_data, val_labels, batch_size, classes, input_shape, alpha, default_failout_survival_rate, train_datagen, epochs, progress_verbose, checkpoint_verbose, train_steps_per_epoch, val_steps_per_epoch)
        
        for survivability_setting in survivability_settings:
            output_list.append(str(survivability_setting) + '\n')
            print(survivability_setting)
            output["Vanilla"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(Vanilla, survivability_setting,output_list, training_labels, test_data, test_labels)
            output["deepFogGuard"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuard, survivability_setting,output_list, training_labels, test_data, test_labels)
            output["ResiliNet"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(ResiliNet, survivability_setting,output_list, training_labels, test_data, test_labels)
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del Vanilla
        del deepFogGuard
        del ResiliNet
   
    for survivability_setting in survivability_settings:
        output_list.append(str(survivability_setting) + '\n')

        Vanilla_acc = average(output["Vanilla"][str(survivability_setting)])
        deepFogGuard_acc = average(output["deepFogGuard"][str(survivability_setting)])
        ResiliNet_acc = average(output["ResiliNet"][str(survivability_setting)])

        output_list.append(str(survivability_setting) + " Vanilla Accuracy: " + str(Vanilla_acc) + '\n')
        output_list.append(str(survivability_setting) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
        output_list.append(str(survivability_setting) + " ResiliNet Accuracy: " + str(ResiliNet_acc) + '\n')

        print(str(survivability_setting),"Vanilla Accuracy:",Vanilla_acc)
        print(str(survivability_setting),"deepFogGuard Accuracy:",deepFogGuard_acc)
        print(str(survivability_setting),"ResiliNet Accuracy:",ResiliNet_acc)
    
    write_n_upload(output_name, output_list, use_GCP)
    print(output)
