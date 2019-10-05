from Experiment.cnn_Vanilla import define_vanilla_model_CNN
from Experiment.cnn_deepFogGuard import define_deepFogGuard_CNN
from Experiment.cnn_ResiliNet import define_ResiliNet_CNN
from keras.utils import multi_gpu_model
import keras
import tensorflow as tf

def define_model(iteration, model_name, dataset_name, input_shape, classes, alpha, default_failout_survival_rate, strides, num_gpus):
    # ResiliNet
    if model_name == "ResiliNet":
        model, parallel_model = define_ResiliNet_CNN(classes=classes,input_shape = input_shape,alpha = alpha, strides = strides, failout_survival_setting=default_failout_survival_rate, num_gpus=num_gpus)
        model_file = "models/" + dataset_name + str(iteration) + 'average_accuracy_ResiliNet.h5'
    # deepFogGuard
    if model_name == "deepFogGuard":
        model, parallel_model = define_deepFogGuard_CNN(classes=classes,input_shape = input_shape,alpha = alpha, strides = strides, num_gpus=num_gpus)
        model_file =  "models/"+ dataset_name  + str(iteration) + 'average_accuracy_deepFogGuard.h5'
    # Vanilla model
    if model_name == "Vanilla":
        model, parallel_model = define_vanilla_model_CNN(classes=classes,input_shape = input_shape,alpha = alpha, strides = strides, num_gpus=num_gpus)
        model_file = "models/" + dataset_name  + str(iteration) + 'average_accuracy_vanilla.h5'
    
    return model, parallel_model, model_file

def compile_keras_parallel_model(img_input, cloud_output, num_gpus, name='ANRL_mobilenet'):
    # Create model.
    with tf.device('/cpu:0'):
        model = keras.Model(img_input, cloud_output, name)

    parallel_model = multi_gpu_model(model, gpus = num_gpus)
    parallel_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model, parallel_model