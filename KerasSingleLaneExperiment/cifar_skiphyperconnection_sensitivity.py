
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
import os 
from KerasSingleLaneExperiment.cnn_deepFogGuard import define_deepFogGuard_CNN
from KerasSingleLaneExperiment.FailureIteration import calculateExpectedAccuracy
import numpy as np
from KerasSingleLaneExperiment.main import average
import datetime
import gc
from sklearn.model_selection import train_test_split


# deepFogGuard hyperconnection failure configuration ablation experiment
if __name__ == "__main__":
    # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    # Concatenate train and test images
    x = np.concatenate((x_train,x_test))
    y = np.concatenate((y_train,y_test))

    # split data into train, validation, and holdout set (80/10/10)
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 42, test_size = .20, shuffle = True)
    x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,random_state = 42, test_size = .50, shuffle = True)
    train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )
    survivability_settings = [
        [.96,.98],
        [.90,.95],
        [.80,.85],
    ]
    # convert survivability settings into strings so it can be used in the dictionary as keys
    normal = str(survivability_settings[0])
    poor = str(survivability_settings[1])
    hazardous = str(survivability_settings[2])
    num_iterations = 20
    skip_hyperconnection_configurations = [
        [0,0],
        [1,0],
        [0,1],
        [1,1],
    ]
    # convert hyperconnection configuration into strings to be used as keys for dictionary
    config_1 = str(skip_hyperconnection_configurations[0])
    config_2 = str(skip_hyperconnection_configurations[1])
    config_3 = str(skip_hyperconnection_configurations[2])
    config_4 = str(skip_hyperconnection_configurations[3])
    batch_size = 128
    epochs = 75
    progress_verbose = 2
    checkpoint_verbose = 1
    use_GCP = True
    alpha = .5
    input_shape = (32,32,3)
    classes = 10
    train_steps_per_epoch = math.ceil(len(x_train) / batch_size)
    val_steps_per_epoch = math.ceil(len(x_val) / batch_size)
    output = {
        "DeepFogGuard Hyperconnection Sensitivity":
        {
            normal:      
            {  
                config_1:[0] * num_iterations,
                config_2:[0] * num_iterations,
                config_3:[0] * num_iterations,
                config_4:[0] * num_iterations,
            },
            poor:
            {
                config_1:[0] * num_iterations,
                config_2:[0] * num_iterations,
                config_3:[0] * num_iterations,
                config_4:[0] * num_iterations,
            },
            hazardous:
            {
                config_1:[0] * num_iterations,
                config_2:[0] * num_iterations,
                config_3:[0] * num_iterations,
                config_4:[0] * num_iterations,
            },
        },
    }
    
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    file_name = 'results/' + date + '/cifar_skiphyperconnection_sensitivity_results.txt'
    output_list = []
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
            model_name = "cifar_skiphyperconnection_sensitivity_results_" + str(skip_hyperconnection_configuration) + str(iteration) + ".h5"
            model = define_deepFogGuard_CNN(classes=classes,input_shape = input_shape,alpha = alpha,hyperconnections = skip_hyperconnection_configuration)
            modelCheckPoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=checkpoint_verbose, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            model.fit_generator(train_datagen.flow(x_train,y_train,batch_size = batch_size),
            epochs = epochs,
            validation_data = (x_val,y_val), 
            steps_per_epoch = train_steps_per_epoch, 
            verbose = progress_verbose, 
            validation_steps = val_steps_per_epoch,
            callbacks = [modelCheckPoint])
            # load weights with the highest validaton acc
            model.load_weights(model_name)
            for survivability_settings in survivability_settings:
                output_list.append(str(survivability_settings) + '\n')
                print(survivability_settings)
                output["DeepFogGuard Hyperconnection Sensitivity"][str(survivability_settings)][str(skip_hyperconnection_configuration)][iteration-1] = calculateExpectedAccuracy(model, survivability_settings,output_list, y_train, x_test, y_test)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del model
    with open(file_name,'a+') as file:
        for survivability_settings in survivability_settings:
            for skip_hyperconnection_configuration in skip_hyperconnection_configurations:
                output_list.append(str(survivability_settings) + '\n')
                deepFogGuard_acc = average(output["DeepFogGuard Hyperconnection Sensitivity"][str(survivability_settings)][str(skip_hyperconnection_configuration)])
                acc_std = np.std(output["DeepFogGuard Hyperconnection Sensitivity"][str(survivability_settings)][str(skip_hyperconnection_configuration)],ddof=1)
                output_list.append(str(survivability_settings) + str(skip_hyperconnection_configuration) + str(deepFogGuard_acc) + '\n')
                output_list.append(str(survivability_settings) + str(skip_hyperconnection_configuration) + str(acc_std) + '\n')
                print(str(survivability_settings),deepFogGuard_acc)
                print(str(survivability_settings), acc_std)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))