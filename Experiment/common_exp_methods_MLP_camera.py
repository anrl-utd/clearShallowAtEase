import os
from Experiment.camera_data_handler import load_dataset
def init_data(use_GCP):
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/multiview-dataset ./')
    if not os.path.exists('models/'):
        os.mkdir('models/')
    train_dir = "multiview_dataset/train_dir"
    val_dir = "multiview_dataset/test_dir"
    test_dir = "multiview_dataset/holdout_dir"
    img_size = (32,32,3)
    classes = ['person_images', 'car_images', 'bus_images']
    train_data, train_labels,_,_ = load_dataset(train_dir,img_size,classes)
    val_data, val_labels,_,_ = load_dataset(val_dir,img_size,classes)
    test_data, test_labels,_,_ = load_dataset(test_dir,img_size,classes)
    return train_data,train_labels,val_data,val_labels,test_data,test_labels

def init_common_experiment_params():
    input_shape = (32,32,3)
    # need to change this to be accurate
    survivability_settings = [
        [1,1],
        [.96,.98],
        [.90,.95],
        [.80,.85],
    ]
    num_classes = 3
    hidden_units = 32
    batch_size = 64
    epochs = 240
    return survivability_settings, input_shape, num_classes, hidden_units, batch_size, epochs
