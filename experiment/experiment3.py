import keras
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, BatchNormalization, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import math
import os 
from experiment.cnn import baseline_ANRL_MobileNet, skipconnections_ANRL_MobileNet, skipconnections_dropout_ANRL_MobileNet
import numpy as np
def view_model():
    model = MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0)
    model.summary()
def main():
    # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )
    survive_configs = [
        [.96,98],
        [.90,.95],
        [.80,.85]
    ]
    file_name = "GitHubANRL_cnn_fullskiphyperconnection_weights_alpha050_fixedstrides_dataaugmentation.h5"
    checkpoint = ModelCheckpoint(file_name,verbose=1,save_best_only=True,save_weights_only = True)
    #model = baseline_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
    #model = skipconnections_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
    model = skipconnections_dropout_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,survive_rates=survive_configs[0])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    num_samples = len(x_train)
    batch_size = 128
    steps_per_epoch = math.ceil(num_samples / batch_size)
    model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2,callbacks = [checkpoint])
    print(model.evaluate(x_test,y_test))
    model.save_weights(file_name)
    os.system('gsutil -m -q cp -r %s gs://anrl-storage/models' % file_name)

def fail_cnn_node():
        # get cifar10 data 
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test / 255
        file_name = "models_GitHubANRL_cnn_fullskiphyperconnection_weights_alpha050_fixedstrides_dataaugmentation.h5"
        model = skipconnections_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        model.load_weights(file_name)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        failed_layers = ["conv_pw_8","conv_pw_3"]
        for layer_name in failed_layers:
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            # make new weights for the connections
            new_weights = np.zeros(layer_weights[0].shape)
            #new_bias_weights[:] = np.nan # set weights to nan
            layer.set_weights([new_weights])
            print(layer_name, "was failed")
        print(model.evaluate(x_test,y_test))
def fail_hyperconnection():
       # get cifar10 data 
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test / 255
        file_name = "models_GitHubANRL_cnn_skiphyperconnection_weights_alpha050_fixedstrides_dataaugmentation.h5"
        model = skipconnections_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        model.load_weights(file_name)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        layer_name = "skip_hyperconnection_edgecloud"
        layer = model.get_layer(name=layer_name)
        layer_weights = layer.get_weights()
        # make new weights for the connections
        new_weights = np.zeros(layer_weights[0].shape)
        #new_bias_weights[:] = np.nan # set weights to nan
        layer.set_weights([new_weights])
        print(layer_name, "was failed")
        print(model.evaluate(x_test,y_test))
# cnn experiment 
if __name__ == "__main__":
    main()
    #fail_cnn_node()
    #fail_hyperconnection()
    #view_model()