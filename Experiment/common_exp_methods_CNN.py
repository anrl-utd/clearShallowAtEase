from Experiment.cnn_Vanilla import define_vanilla_model_CNN
from Experiment.cnn_deepFogGuard import define_deepFogGuard_CNN
from Experiment.cnn_ResiliNet import define_ResiliNet_CNN

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
