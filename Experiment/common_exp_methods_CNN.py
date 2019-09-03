from Experiment.cnn_Vanilla import define_vanilla_model_CNN
from Experiment.cnn_deepFogGuard import define_deepFogGuard_CNN
from Experiment.cnn_deepFogGuardPlus import define_deepFogGuardPlus_CNN

def define_model(iteration, model_name, dataset_name, input_shape, classes, alpha, default_failout_survival_rate, strides):
    # ResiliNet
    if model_name == "ResiliNet":
        model = define_deepFogGuardPlus_CNN(classes=classes,input_shape = input_shape,alpha = alpha, strides = strides, failout_survival_setting=default_failout_survival_rate)
        model_file = "ResiliNet_"+dataset_name+"_average_accuracy" + str(iteration) + ".h5"
    # deepFogGuard
    if model_name == "deepFogGuard":
        model = define_deepFogGuard_CNN(classes=classes,input_shape = input_shape,alpha = alpha, strides = strides)
        model_file = "deepFogGuard_"+dataset_name+"_average_accuracy" + str(iteration) + ".h5"
    # Vanilla model
    if model_name == "Vanilla":
        model = define_vanilla_model_CNN(classes=classes,input_shape = input_shape,alpha = alpha, strides = strides)
        model_file = "vanilla_cifar_"+dataset_name+"_accuracy" + str(iteration) + ".h5"
    
    return model, model_file