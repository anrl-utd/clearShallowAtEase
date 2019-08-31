import os
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing.image import load_img
def init_data(use_GCP):
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/multiview-dataset ./')
    if not os.path.exists('models/'):
        os.mkdir('models/')
    train_dir = "/home/ubuntu/imagenet/train"
    test_dir = "/home/ubuntu/imagenet/val"
    input_shape = (256,256)
    batch_size = 64
    datagen = ImageDataGenerator(
        rescale = 1./255
    )
    train_generator = datagen.flow_from_directory(
        directory = train_dir,
        target_size = input_shape,
        batch_size = batch_size,
    )
    test_generator = datagen.flow_from_directory(
        directory = test_dir,
        target_size = input_shape,
        batch_size = batch_size,
    )
    return train_generator, test_generator

def init_common_experiment_params():
    num_train_examples = 1300000
    num_test_examples = 50000
    input_shape = (256,256,3)
    alpha = .5
    num_iterations = 10
    # need to change this to be accurate
    survivability_settings = [
        [1,1],
        [.96,.98],
        [.90,.95],
        [.80,.85],
    ]
    num_classes = 10
    epochs = 100
    return num_iterations, num_train_examples,num_test_examples, survivability_settings, input_shape, num_classes, alpha, epochs
