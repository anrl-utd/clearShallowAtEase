
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, BatchNormalization, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import math
import os 
from experiment.cnn import baseline_ANRL_MobileNet, skipconnections_ANRL_MobileNet, skipconnections_dropout_ANRL_MobileNet
from experiment.FailureIteration import run
import numpy as np
from experiment.experiment import average
import datetime
import gc

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
        [.96,.98],
        [.90,.95],
        [.80,.85],
        [1,1]
    ]
    num_iterations = 10
    output = {
        "Active Guard":
        {
            "[0.96, 0.98]": [0] * num_iterations,
            "[0.9, 0.95]":[0] * num_iterations,
            "[0.8, 0.85]":[0] * num_iterations,
            "[1, 1]":[0] * num_iterations,
        },
    }
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    # make folder for outputs 
    if not os.path.exists('results/' + date):
        os.mkdir('results/')
        os.mkdir('results/' + date)
    file_name = 'results/' + date + '/experiment3_baselineexperiment_results.txt'
    for iteration in range(1,num_iterations+1):
        print("iteration:",iteration)
        model_name = "GitHubANRL_cnn_baseline_weights_alpha050_fixedstrides_dataaugmentation" + str(iteration) + ".h5"
        checkpoint = ModelCheckpoint(model_name,verbose=1,save_best_only=True,save_weights_only = True)
        model = baseline_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        #model = skipconnections_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        # dropout = .1
        #model = skipconnections_dropout_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5,survive_rates=[.9,.9,.9])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        num_samples = len(x_train)
        batch_size = 128
        steps_per_epoch = math.ceil(num_samples / batch_size)
        model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2,callbacks = [checkpoint])
        output_list = []
        for survive_config in survive_configs:
            output_list.append(str(survive_config) + '\n')
            print(survive_config)
            output["Active Guard"][str(survive_config)][iteration-1] = run(" ",model, survive_config,output_list, y_train, x_test, y_test)
        # clear session so that model will recycled back into memory
        K.clear_session()
        gc.collect()
        del model
    with open(file_name,'a+') as file:
        for survive_config in survive_configs:
            output_list.append(str(survive_config) + '\n')
            active_guard_acc = average(output["Active Guard"][str(survive_config)])
            output_list.append(str(survive_config) + " .1 Dropout Accuracy: " + str(active_guard_acc) + '\n')
            print(str(survive_config),".1 Dropout Accuracy:",active_guard_acc)
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    use_GCP = True
    if use_GCP:
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')

def fail_cnn_node():
        # get cifar10 data 
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test / 255
        model_name = "models_GitHubANRL_cnn_fullskiphyperconnectiondropout_lowconfig_weights_alpha050_fixedstrides_dataaugmentation.h5"
        model = skipconnections_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        model.load_weights(model_name)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #"conv_pw_8"
        #"conv_pw_3"
        failed_layers = ["conv_pw_3"]
        for layer_name in failed_layers:
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            # make new weights for the connections
            new_weights = np.zeros(layer_weights[0].shape)
            #new_bias_weights[:] = np.nan # set weights to nan
            layer.set_weights([new_weights])
            print(layer_name, "was failed")
        print(model.evaluate(x_test,y_test))

# boost other available hyperconnection when node fails
def fail_cnn_node_experiment2():
        # get cifar10 data 
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test / 255
        model_name = "models_GitHubANRL_cnn_fullskiphyperconnection_weights_alpha050_fixedstrides_dataaugmentation.h5"
        model = skipconnections_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        model.load_weights(model_name)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #"conv_pw_8"
        #"conv_pw_3"
        failed_layers = ["conv_pw_3"]
        for layer_name in failed_layers:
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            # make new weights for the connections
            new_weights = np.zeros(layer_weights[0].shape)
            #new_bias_weights[:] = np.nan # set weights to nan
            layer.set_weights([new_weights])
            print(layer_name, "was failed")
        #boosted_hyperconnection = model.get_layer("skip_hyperconnection_iotfog")
        boosted_hyperconnection = model.get_layer("skip_hyperconnection_edgecloud")
        boosted_hyperconnection.set_weights(np.array(boosted_hyperconnection.get_weights()) * 5)
        print(model.evaluate(x_test,y_test))

def fail_hyperconnection():
       # get cifar10 data 
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test / 255
        model_name = "models_GitHubANRL_cnn_skiphyperconnection_weights_alpha050_fixedstrides_dataaugmentation.h5"
        model = skipconnections_ANRL_MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
        model.load_weights(model_name)
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
    #fail_cnn_node_experiment2()
    #fail_hyperconnection()
    #view_model()