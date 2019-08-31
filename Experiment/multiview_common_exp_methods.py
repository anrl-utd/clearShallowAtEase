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
    pass
