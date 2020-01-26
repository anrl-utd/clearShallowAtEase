import os
from Experiment.data_handler_health import load_data
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import ModelCheckpoint

def init_data(use_GCP):
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/mHealth_complete.log ./')
        if not os.path.exists('models/'):
            os.mkdir('models/')
    data,labels= load_data('mHealth_complete.log')
    # split data into train, val, and test
    # 80/10/10 split
    train_data, test_data, train_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .20, shuffle = True)
    val_data, test_data, val_labels, test_labels = train_test_split(test_data,test_labels,random_state = 42, test_size = .50, shuffle = True)
    return  train_data, val_data, test_data, train_labels, val_labels, test_labels

def init_common_experiment_params(train_data):
    num_vars = len(train_data[0])
    num_classes = 13
    reliability_settings = [
        [1,1,1],
        [.99,.96,.92],
        [.95,.91,.87],
        [.85,.8,.78],
    ]
    num_train_epochs = 50
    hidden_units = 250
    batch_size = 1024
    num_iterations = 10
    return num_iterations, num_vars, num_classes, reliability_settings, num_train_epochs, hidden_units, batch_size

def get_model_weights_MLP_health(model, model_name, load_for_inference, model_file, training_data, training_labels, val_data, val_labels, num_train_epochs, batch_size, verbose):
    if load_for_inference:
        model.load_weights(model_file)
    else:
        print(model_name)
        modelCheckPoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        model.fit(
            x = training_data,
            y = training_labels,
            batch_size = batch_size,
            validation_data = (val_data,val_labels),
            callbacks = [modelCheckPoint],
            verbose = verbose,
            epochs = num_train_epochs,
            shuffle = True
        )
        # load weights from epoch with the highest val acc
        model.load_weights(model_file)

