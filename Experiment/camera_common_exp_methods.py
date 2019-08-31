import os
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing.image import load_img
def init_data(use_GCP):
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/multiview-dataset ./')
    if not os.path.exists('models/'):
        os.mkdir('models/')
    train_dir = "multiview_dataset/train_dir"
    val_dir = "multiview_dataset/test_dir"
    test_dir = "multiview_dataset/holdout_dir"
    input_shape = (32,32)
    batch_size = 64
    datagen = ImageDataGenerator(
        rescale = 1./255
    )
    train_generator = datagen.flow_from_directory(
        directory = train_dir,
        target_size = input_shape,
        batch_size = batch_size,
    )
    val_generator = datagen.flow_from_directory(
        directory = val_dir ,
        target_size = input_shape,
        batch_size = batch_size,
    )
    test_generator = datagen.flow_from_directory(
        directory = test_dir,
        target_size = input_shape,
        batch_size = batch_size,
    )
    return train_generator, val_generator, test_generator

def init_common_experiment_params():
    num_train_examples = 1000
    num_val_examples = 1000
    num_test_examples = 1000
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
    return num_train_examples,num_val_examples,num_test_examples, survivability_settings, input_shape, num_classes, hidden_units
