import os
from Experiment.data_handler_camera import load_dataset
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight

def init_data(use_GCP):
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/multiview-dataset ./')
    if not os.path.exists('models/'):
        os.mkdir('models/')
    train_dir = "multiview-dataset/train_dir"
    val_dir = "multiview-dataset/test_dir"
    test_dir = "multiview-dataset/holdout_dir"
    img_size = (32,32,3)
    classes = ['person_images', 'car_images', 'bus_images']
    training_data, training_labels,_,_ = load_dataset(train_dir,img_size,classes)
    val_data, val_labels,_,_ = load_dataset(val_dir,img_size,classes)
    test_data, test_labels,_,_ = load_dataset(test_dir,img_size,classes)

    training_data = np.array(training_data)
    val_data = np.array(val_data)
    test_data = np.array(test_data)

    # convert one-hot to integer encoding
    training_labels = np.array([np.where(r==1)[0][0] for r in training_labels])
    val_labels = np.array([np.where(r==1)[0][0] for r in val_labels])
    test_labels = np.array([np.where(r==1)[0][0] for r in test_labels])
    # format images correctly to be used for MLP
    training_data = [training_data[:,0],training_data[:,1],training_data[:,2],training_data[:,3],training_data[:,4],training_data[:,5]]
    val_data = [val_data[:,0],val_data[:,1],val_data[:,2],val_data[:,3],val_data[:,4],val_data[:,5]]
    test_data = [test_data[:,0],test_data[:,1],test_data[:,2],test_data[:,3],test_data[:,4],test_data[:,5]]
    return training_data,val_data, test_data, training_labels,val_labels,test_labels

def init_common_experiment_params():
    input_shape = (32,32,3)
    # need to change this to be accurate
    reliability_settings = [
        [1,1,1,1,1,1,1,1],
        [.99,.99,.99,.99,.98,.98,.98,.98],
        [.95,.95,.9,.9,.8,.8,.75,.75],
        [.8,.8,.75,.75,.7,.7,.65,.65]
    ]
    num_classes = 3
    hidden_units = 32
    batch_size = 64
    num_train_epochs = 50
    num_iterations = 10
    return reliability_settings, input_shape, num_classes, hidden_units, batch_size, num_train_epochs, num_iterations

def get_model_weights_MLP_camera(model, model_name, load_for_inference, model_file, training_data, training_labels, val_data, val_labels,num_train_epochs, batch_size, verbose):
    if load_for_inference:
        model.load_weights(model_file)
    else:
        print(model_name)
        modelCheckPoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        class_weights = class_weight.compute_class_weight('balanced',np.unique(training_labels),training_labels)
        model.fit(
            x = training_data,
            y = training_labels,
            batch_size = batch_size,
            validation_data = (val_data,val_labels),
            callbacks = [modelCheckPoint],
            verbose = verbose,
            epochs = num_train_epochs,
            shuffle = True,
            class_weight = class_weights
        )
        # load weights from epoch with the highest val acc
        model.load_weights(model_file)
