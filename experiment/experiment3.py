import keras
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, BatchNormalization, Input, Dropout
from keras.models import Model
import math
import os 
# cnn experiment 
if __name__ == "__main__":
    # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )
    model = MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .25)
    # transfer learning 
    input_tensor = Input(shape=(32, 32, 3))
   # original_model = MobileNet(weights = 'imagenet',input_tensor = input_tensor, include_top = False, pooling = 'avg',dropout = .25)
    # freeze first 25 layers
    # for layer in original_model.layers[:25]:
    #     layer.trainable=False
    # new_model = Dense(50, activation = 'relu')(original_model.output)
    # new_model = BatchNormalization()(new_model)
    # new_model = Dense(50, activation = 'relu')(new_model)
    # new_model = BatchNormalization()(new_model)
    # new_model = Dense(50, activation = 'relu')(new_model)
    # new_model = BatchNormalization()(new_model)
    # output = Dense(10, activation = 'softmax')(new_model)
    # model= Model(inputs=original_model.input,outputs=output)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    num_samples = len(x_train)
    batch_size = 128
    steps_per_epoch = math.ceil(num_samples / batch_size)
    model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 75,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch, verbose = 2)
    print(model.evaluate(x_test,y_test))
    file_name = "cnn_weights.h5"
    model.save_weights(file_name)
    os.system('gsutil -m -q cp -r %s gs://anrl-storage/models' % file_name)