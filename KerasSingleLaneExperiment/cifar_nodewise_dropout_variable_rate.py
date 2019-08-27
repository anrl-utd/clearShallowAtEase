
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
import os 
from KerasSingleLaneExperiment.cnn_deepFogGuardPlus import define_deepFogGuardPlus_CNN
from KerasSingleLaneExperiment.FailureIteration import calculateExpectedAccuracy
import numpy as np
from KerasSingleLaneExperiment.main import average
import datetime
import gc
from sklearn.model_selection import train_test_split

# deepFogGuard Plus variable dropout experiment
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
 
    num_iterations = 10
    output_list = []
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
       # dictionary to store all the results
    output = {
         "deepFogGuardPlus Node-wise Variable Dropout": 
        {
            hazardous:[0] * num_iterations,
            poor :[0] * num_iterations,
            normal:[0] * num_iterations,
        }
    }
  
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + date):
        os.mkdir('results/' + date)
    file_name = 'results/' + date + '/cifar_nodewise_dropout_results.txt'
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        for survivability_setting in survivability_settings:
            # # node-wise dropout
            deepFogGuardPlus_variable_nodewise_dropout_file = "cifar_nodewise_variable_dropout_" + str(iteration) + ".h5"
            deepFogGuardPlus_variable_nodewise_dropout = define_deepFogGuardPlus_CNN(classes=classes,input_shape = input_shape,alpha = alpha,survivability_setting=survivability_setting)
            deepFogGuardPlus_variable_nodewise_dropout_Checkpoint = ModelCheckpoint(deepFogGuardPlus_variable_nodewise_dropout_file, monitor='val_acc', verbose=checkpoint_verbose, save_best_only=True, save_weights_only=True, mode='auto', period=1)
            deepFogGuardPlus_variable_nodewise_dropout.fit_generator(train_datagen.flow(x_train,y_train,batch_size = batch_size),
            epochs = epochs,
            validation_data = (x_val,y_val), 
            steps_per_epoch = train_steps_per_epoch, 
            verbose = progress_verbose, 
            validation_steps = val_steps_per_epoch,
            callbacks = [deepFogGuardPlus_variable_nodewise_dropout_Checkpoint])
            # load weights with the highest validaton acc
            deepFogGuardPlus_variable_nodewise_dropout.load_weights(deepFogGuardPlus_variable_nodewise_dropout_file)
            output_list.append(str(survivability_setting) + '\n')
            print(survivability_setting)
            output["deepFogGuardPlus Node-wise Variable Dropout"][str(survivability_setting)][iteration-1] = calculateExpectedAccuracy(deepFogGuardPlus_variable_nodewise_dropout, survivability_setting,output_list, y_train, x_test, y_test)
            # clear session so that model will recycled back into memory
            K.clear_session()
            gc.collect()
            del deepFogGuardPlus_variable_nodewise_dropout
    with open(file_name,'a+') as file:
        for survivability_setting in survivability_settings:
                output_list.append(str(survivability_setting) + '\n')
                
                deepGuardPlus_acc = average(output["deepFogGuardPlus Node-wise Variable Dropout"][str(survivability_setting)])
                output_list.append(str(survivability_setting) + " nodewise_survival_rate Accuracy: " + str(deepGuardPlus_acc) + '\n')
                print(str(survivability_setting), " variable nodewise_survival_rate Accuracy:",deepGuardPlus_acc)

                deepGuardPlus_std = np.std(output["deepFogGuardPlus Node-wise Variable Dropout"][str(survivability_setting)],ddof=1)
                output_list.append(str(survivability_setting) + " nodewise_survival_rate std: " + str(deepGuardPlus_std) + '\n')
                print(str(survivability_setting), " variable nodewise_survival_rate std:",deepGuardPlus_std)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    if use_GCP:
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(file_name))