
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

# normal experiments
def main():
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
        "Active Guard":
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
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    file_name = 'results/' + date + '/experiment3_baselineexperiment_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        model_name = "GitHubANRL_cnn_baseline_weights_alpha050_fixedstrides_dataaugmentation" + str(iteration) + ".h5"
        checkpoint = ModelCheckpoint(model_name,verbose=1,save_best_only=True,save_weights_only = True)
        model = define_Vanilla_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        #model = skipconnections_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        # dropout = .1
        #model = skipconnections_dropout_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,survive_rates=[.9,.9,.9])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        num_samples = len(x_train)
        batch_size = 128
        steps_per_epoch = math.ceil(num_samples / batch_size)
        model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2,callbacks = [checkpoint])
        for survive_config in survive_configs:
            output_list.append(str(survive_config) + '\n')
            print(survive_config)
            output["Active Guard"][str(survive_config)][iteration-1] = run(model, survive_config,output_list, y_train, x_test, y_test)
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del model
    with open(file_name,'a+') as file:
        for survive_config in survive_configs:
            output_list.append(str(survive_config) + '\n')
            active_guard_acc = average(output["Active Guard"][str(survive_config)])
            output_list.append(str(survive_config) + " .1 Dropout Accuracy: " + str(active_guard_acc) + '\n')
            print(str(survive_config),".1 Dropout Accuracy:",active_guard_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    use_GCP = True
    if use_GCP:
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))

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
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    file_name = 'results/' + date + '/experiment3_dropoutAblation_95_6to10results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        for dropout in dropout_configs:
            model_name = "GitHubANRL_deepFogGuardPlus_dropoutAblation95" + str(dropout) + "6to10" + str(iteration) + ".h5"
            model = define_deepFogGuardPlus_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,survive_rates=dropout)
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            num_samples = len(x_train)
            batch_size = 128
            steps_per_epoch = math.ceil(num_samples / batch_size)
            model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2)
            model.save_weights(model_name)
            for survive_config in survive_configs:
                output_list.append(str(survive_config) + '\n')
                print(survive_config)
                output["DeepFogGuard Plus Baseline"][str(dropout)][str(survive_config)][iteration-1] = run(model, survive_config,output_list, y_train, x_test, y_test)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del model
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
    use_GCP = True
    if use_GCP:
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))

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
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    file_name = 'results/' + date + '/experiment3_hyperconnection_weight_ablation_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        num_samples = len(x_train)
        batch_size = 128
        steps_per_epoch = math.ceil(num_samples / batch_size)
        for survive_config in survive_configs:
            model_name = "GitHubANRL_deepFogGuard_hyperconnectionweightablation_" + str(survive_config) + "_weights_alpha050_fixedstrides_dataaugmentation" + str(iteration) + ".h5"
            model = define_deepFogGuard_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,hyperconnection_weights=survive_config)
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2)
            model.save_weights(model_name)
            output_list.append(str(survive_config) + '\n')
            print(survive_config)
            output["DeepFogGuard Baseline"][str(survive_config)][iteration-1] = run(model, survive_config,output_list, y_train, x_test, y_test)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del model
    with open(file_name,'a+') as file:
        for survive_config in survive_configs:
            output_list.append(str(survive_config) + '\n')
            active_guard_acc = average(output["DeepFogGuard Baseline"][str(survive_config)])
            output_list.append(str(survive_config) + str(active_guard_acc) + '\n')
            print(str(survive_config),active_guard_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    use_GCP = True
    if use_GCP:
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))

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
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    file_name = 'results/' + date + '/experiment3_hyperconnection_sensitivityablation_results3.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        num_samples = len(x_train)
        batch_size = 128
        steps_per_epoch = math.ceil(num_samples / batch_size)
        for hyperconnection in hyperconnections:
            model_name = "GitHubANRL_deepFogGuardPlus_hyperconnectionsensitvityablation3" + str(hyperconnection) + "_weights_alpha050_fixedstrides_dataaugmentation" + str(iteration) + ".h5"
            model = define_deepFogGuard_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,hyperconnections = hyperconnection)
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            num_samples = len(x_train)
            batch_size = 128
            steps_per_epoch = math.ceil(num_samples / batch_size)
            model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2)
            model.save_weights(model_name)
            for survive_config in survive_configs:
                output_list.append(str(survive_config) + '\n')
                print(survive_config)
                output["DeepFogGuard Hyperconnection Sensitivity"][str(survive_config)][str(hyperconnection)][iteration-1] = run(model, survive_config,output_list, y_train, x_test, y_test)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del model
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
    use_GCP = True
    if use_GCP:
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))

# cnn experiment 
if __name__ == "__main__":
    #main()
    dropout_ablation()
    #hyperconnection_weight_ablation()
    # hyperconnection_sensitivity_ablation()