import os
from Experiment.data_handler_camera import load_dataset
import numpy as np
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
    train_data, train_labels,_,_ = load_dataset(train_dir,img_size,classes)
    val_data, val_labels,_,_ = load_dataset(val_dir,img_size,classes)
    test_data, test_labels,_,_ = load_dataset(test_dir,img_size,classes)

    train_data = np.array(train_data)
    val_data = np.array(val_data)
    test_data = np.array(test_data)

    # convert one-hot to integer encoding
    train_labels = np.array([np.where(r==1)[0][0] for r in train_labels])
    val_labels = np.array([np.where(r==1)[0][0] for r in val_labels])
    test_labels = np.array([np.where(r==1)[0][0] for r in test_labels])
    # format images correctly to be used for MLP
    train_data = [train_data[:,0],train_data[:,1],train_data[:,2],train_data[:,3],train_data[:,4],train_data[:,5]]
    val_data = [val_data[:,0],val_data[:,1],val_data[:,2],val_data[:,3],val_data[:,4],val_data[:,5]]
    test_data = [test_data[:,0],test_data[:,1],test_data[:,2],test_data[:,3],test_data[:,4],test_data[:,5]]
    return train_data,train_labels,val_data,val_labels,test_data,test_labels

def init_common_experiment_params():
    input_shape = (32,32,3)
    # need to change this to be accurate
    survivability_settings = [
        [1,1,1,1,1,1,1,1],
        [.97,.98,.985,.985,.99,.995,.998,.999],
        [.66,.7,.6,.7,.8,.8,.9,.9],
        [.6,.6,.65,.65,.7,.75,.8,.8]
    ]
    num_classes = 3
    hidden_units = 32
    batch_size = 64
    epochs = 24
    return survivability_settings, input_shape, num_classes, hidden_units, batch_size, epochs

def get_model_weights_MLP_camera(model, model_name, load_model, model_file, train_data, train_labels, val_data, val_labels,num_train_epochs, batch_size, verbose):
    if load_model:
        model.load_weights(model_file)
    else:
        print(model_name)
        modelCheckPoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        class_weights = class_weight.compute_class_weight('balanced',np.unique(train_labels),train_labels)
        model.fit(
            x = train_data,
            y = train_labels,
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