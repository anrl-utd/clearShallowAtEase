
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, BatchNormalization, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
import os 
from KerasSingleLaneExperiment.cnn import define_Vanilla_CNN, define_deepFogGuard_CNN, define_deepFogGuardPlus_CNN
from KerasSingleLaneExperiment.FailureIteration import run
import numpy as np
from KerasSingleLaneExperiment.experiment import average
import datetime
import gc

# Vanilla, deepFogGuard, and deepFogGuard+ experiments
def cnn_normal_experiments():
    # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )
    survive_configs = [
        [.96,.98],
        [.90,.95],
        [.80,.85],
        [1,1]
    ]
    num_iterations = 10
    output = {
        "deepFogGuard Plus":
        {
            "[0.96, 0.98]": [0] * num_iterations,
            "[0.9, 0.95]":[0] * num_iterations,
            "[0.8, 0.85]":[0] * num_iterations,
            "[1, 1]":[0] * num_iterations,
        },
        "deepFogGuard":
        {
            "[0.96, 0.98]": [0] * num_iterations,
            "[0.9, 0.95]":[0] * num_iterations,
            "[0.8, 0.85]":[0] * num_iterations,
            "[1, 1]":[0] * num_iterations,
        },
          "Vanilla":
        {
            "[0.96, 0.98]": [0] * num_iterations,
            "[0.9, 0.95]":[0] * num_iterations,
            "[0.8, 0.85]":[0] * num_iterations,
            "[1, 1]":[0] * num_iterations,
        },
    }
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + date):
        os.mkdir('results/' + date)
    if not os.path.exists('models'):      
        os.mkdir('models/')
    file_name = 'results/' + date + '/experiment3_baselineexperiment_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        vanilla_name = "vanilla_cnn" + str(iteration) + ".h5"
        deepFogGuard_name = "deepFogGuard_cnn" + str(iteration) + ".h5"
        deepFogGuardPlus_name = "deepFogGuard_cnn" + str(iteration) + ".h5"

        vanilla = define_Vanilla_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        deepFogGuard = define_deepFogGuard_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        deepFogGuardPlus = define_deepFogGuardPlus_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,survive_rates=[.9,.9,.9])

        num_samples = len(x_train)
        batch_size = 128
        steps_per_epoch = math.ceil(num_samples / batch_size)
        vanilla.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2)
        deepFogGuard.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2)
        deepFogGuardPlus.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2)

        vanilla.save_weights(vanilla_name)
        deepFogGuard.save_weights(deepFogGuard_name)
        deepFogGuardPlus.save_weights(deepFogGuardPlus_name)

        for survive_config in survive_configs:
            output_list.append(str(survive_config) + '\n')
            print(survive_config)
            output["Vanilla"][str(survive_config)][iteration-1] = run(vanilla, survive_config,output_list, y_train, x_test, y_test)
            output["deepFogGuard"][str(survive_config)][iteration-1] = run(deepFogGuard, survive_config,output_list, y_train, x_test, y_test)
            output["deepFogGuard Plus"][str(survive_config)][iteration-1] = run(deepFogGuardPlus, survive_config,output_list, y_train, x_test, y_test)
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del vanilla
        del deepFogGuard
        del deepFogGuardPlus
    with open(file_name,'a+') as file:
        for survive_config in survive_configs:
            output_list.append(str(survive_config) + '\n')

            vanilla_acc = average(output["Vanilla"][str(survive_config)])
            deepFogGuard_acc = average(output["deepFogGuard"][str(survive_config)])
            deepFogGuardPlus_acc = average(output["deepFogGuard Plus"][str(survive_config)])

            output_list.append(str(survive_config) + " Vanilla Accuracy: " + str(vanilla_acc) + '\n')
            output_list.append(str(survive_config) + " deepFogGuard Accuracy: " + str(deepFogGuard_acc) + '\n')
            output_list.append(str(survive_config) + " deepFogGuard Plus Accuracy: " + str(deepFogGuardPlus_acc) + '\n')

            print(str(survive_config),"Vanilla Accuracy:",vanilla_acc)
            print(str(survive_config),"deepFogGuard Accuracy:",deepFogGuard_acc)
            print(str(survive_config),"deepFogGuard Plus Accuracy:",deepFogGuardPlus_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)

# deepFogGuard Plus Ablation experiment
def dropout_ablation():
    # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )
    survive_configs = [
        [.96,.98],
        [.90,.95],
        [.80,.85],
        [1,1]
    ]
    num_iterations = 10
    output = {
        "DeepFogGuard Plus Baseline":
        {
            "[0.9, 0.9, 0.9]":
            {
                "[0.96, 0.98]": [0] * num_iterations,
                "[0.9, 0.95]":[0] * num_iterations,
                "[0.8, 0.85]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
            "[0.7, 0.7, 0.7]":
            {
               "[0.96, 0.98]": [0] * num_iterations,
                "[0.9, 0.95]":[0] * num_iterations,
                "[0.8, 0.85]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
            "[0.5, 0.5, 0.5]":
            {
                "[0.96, 0.98]": [0] * num_iterations,
                "[0.9, 0.95]":[0] * num_iterations,
                "[0.8, 0.85]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
            "[0.95, 0.95, 0.95]":
            {
                "[0.96, 0.98]": [0] * num_iterations,
                "[0.9, 0.95]":[0] * num_iterations,
                "[0.8, 0.85]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
        }
    }
    dropout_configs = [
        [.9,.9,.9],
        [.7,.7,.7],
        [.5,.5,.5],
        [.95,.95,.95],
    ]
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + date):
        os.mkdir('results/' + date)
    if not os.path.exists('models'):      
        os.mkdir('models/')
    file_name = 'results/' + date + '/experiment3_dropoutAblation_95_6to10results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        for dropout in dropout_configs:
            model_name = "GitHubANRL_deepFogGuardPlus_dropoutAblation95" + str(dropout) + "6to10" + str(iteration) + ".h5"
            vanilla_model = define_deepFogGuardPlus_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,survive_rates=dropout)
            vanilla_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            num_samples = len(x_train)
            batch_size = 128
            steps_per_epoch = math.ceil(num_samples / batch_size)
            vanilla_model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2)
            vanilla_model.save_weights(model_name)
            for survive_config in survive_configs:
                output_list.append(str(survive_config) + '\n')
                print(survive_config)
                output["DeepFogGuard Plus Baseline"][str(dropout)][str(survive_config)][iteration-1] = run(vanilla_model, survive_config,output_list, y_train, x_test, y_test)
            # clear session so that vanilla_model will recycled back into memory
            K.clear_session()
            gc.collect()
            del vanilla_model
    with open(file_name,'a+') as file:
        for survive_config in survive_configs:
            for dropout in dropout_configs:
                output_list.append(str(survive_config) + '\n')
                active_guard_acc = average(output["DeepFogGuard Plus Baseline"][str(dropout)][str(survive_config)])
                output_list.append(str(survive_config) + str(dropout) + " Dropout Accuracy: " + str(active_guard_acc) + '\n')
                print(str(survive_config), str(dropout), " Dropout Accuracy:",active_guard_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)

# deepFogGuard hyperconnection weight ablation experiment      
def hyperconnection_weight_ablation():
     # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )
    survive_configs = [
        [.96,.98],
        [.90,.95],
        [.80,.85],
        [1,1]
    ]
    num_iterations = 10
    output = {
        "DeepFogGuard Baseline":
        {
            "[0.96, 0.98]": [0] * num_iterations,
            "[0.9, 0.95]":[0] * num_iterations,
            "[0.8, 0.85]":[0] * num_iterations,
            "[1, 1]":[0] * num_iterations,
        },
    }
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
   # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + date):
        os.mkdir('results/' + date)
    if not os.path.exists('models'):      
        os.mkdir('models/')
    file_name = 'results/' + date + '/experiment3_hyperconnection_weight_ablation_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        num_samples = len(x_train)
        batch_size = 128
        steps_per_epoch = math.ceil(num_samples / batch_size)
        for survive_config in survive_configs:
            model_name = "deepFogGuard_hyperconnectionweightablation_" + str(survive_config) + str(iteration) + ".h5"
            vanilla_model = define_deepFogGuard_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,hyperconnection_weights=survive_config)
            vanilla_model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2)
            vanilla_model.save_weights(model_name)
            output_list.append(str(survive_config) + '\n')
            print(survive_config)
            output["DeepFogGuard Baseline"][str(survive_config)][iteration-1] = run(vanilla_model, survive_config,output_list, y_train, x_test, y_test)
            # clear session so that vanilla_model will recycled back into memory
            K.clear_session()
            gc.collect()
            del vanilla_model
    with open(file_name,'a+') as file:
        for survive_config in survive_configs:
            output_list.append(str(survive_config) + '\n')
            active_guard_acc = average(output["DeepFogGuard Baseline"][str(survive_config)])
            output_list.append(str(survive_config) + str(active_guard_acc) + '\n')
            print(str(survive_config),active_guard_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
  
# deepFogGuard hyperconnection failure configuration ablation experiment
def hyperconnection_sensitivity_ablation():
    # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )
    survive_configs = [
        [.96,.98],
        [.90,.95],
        [.80,.85],
    ]
    num_iterations = 20
    hyperconnections = [
        [0,0],
        [1,0],
        [0,1],
        [1,1],
    ]
    output = {
        "DeepFogGuard Hyperconnection Sensitivity":
        {
            "[0.96, 0.98]":      
            {  
                "[0, 0]":[0] * num_iterations,
                "[1, 0]":[0] * num_iterations,
                "[0, 1]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
            "[0.9, 0.95]":
            {
                "[0, 0]":[0] * num_iterations,
                "[1, 0]":[0] * num_iterations,
                "[0, 1]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
            "[0.8, 0.85]":
            {
                "[0, 0]":[0] * num_iterations,
                "[1, 0]":[0] * num_iterations,
                "[0, 1]":[0] * num_iterations,
                "[1, 1]":[0] * num_iterations,
            },
        },
    }
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + date):
        os.mkdir('results/' + date)
    if not os.path.exists('models'):      
        os.mkdir('models/')
    file_name = 'results/' + date + '/experiment3_hyperconnection_sensitivityablation_results3.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        num_samples = len(x_train)
        batch_size = 128
        steps_per_epoch = math.ceil(num_samples / batch_size)
        for hyperconnection in hyperconnections:
            model_name = "deepFogGuard_hyperconnectionsensitvityablation3" + str(hyperconnection) + str(iteration) + ".h5"
            vanilla_model = define_deepFogGuard_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,hyperconnections = hyperconnection)
            num_samples = len(x_train)
            batch_size = 128
            steps_per_epoch = math.ceil(num_samples / batch_size)
            vanilla_model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2)
            vanilla_model.save_weights(model_name)
            for survive_config in survive_configs:
                output_list.append(str(survive_config) + '\n')
                print(survive_config)
                output["DeepFogGuard Hyperconnection Sensitivity"][str(survive_config)][str(hyperconnection)][iteration-1] = run(vanilla_model, survive_config,output_list, y_train, x_test, y_test)
            # clear session so that vanilla_model will recycled back into memory
            K.clear_session()
            gc.collect()
            del vanilla_model
    with open(file_name,'a+') as file:
        for survive_config in survive_configs:
            for hyperconnection in hyperconnections:
                output_list.append(str(survive_config) + '\n')
                active_guard_acc = average(output["DeepFogGuard Hyperconnection Sensitivity"][str(survive_config)][str(hyperconnection)])
                acc_std = np.std(output["DeepFogGuard Hyperconnection Sensitivity"][str(survive_config)][str(hyperconnection)],ddof=1)
                output_list.append(str(survive_config) + str(hyperconnection) + str(active_guard_acc) + '\n')
                output_list.append(str(survive_config) + str(hyperconnection) + str(acc_std) + '\n')
                print(str(survive_config),active_guard_acc)
                print(str(survive_config), acc_std)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)

# cnn experiment 
if __name__ == "__main__":
    cnn_normal_experiments()
    dropout_ablation()
    hyperconnection_weight_ablation()
    hyperconnection_sensitivity_ablation()