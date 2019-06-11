from keras.models import Sequential
from keras.layers import Dense,Input,add,multiply,Lambda, BatchNormalization, Activation, Dropout
import keras.backend as K
import tensorflow as tf
from experiment.LambdaLayers import add_first_node_layers,add_node_layers
from keras import regularizers
from keras import optimizers
from keras.models import Model

def define_baseline_functional_model(num_vars,num_classes,hidden_units,regularization):
 # calculate connection weights
    # naming convention:
    # ex: f1f2 = connection between fog node 1 and fog node 2
    # ex: f2c = connection between fog node 2 and cloud node

    connection_weight_f1f2 = 1
    connection_weight_f2f3 = 1
    connection_weight_f3c = 1


    # define lambdas for multiplying node weights by connection weight
    multiply_weight_layer_f1f2 = Lambda((lambda x: x * connection_weight_f1f2), name = "connection_weight_f1f2")
    multiply_weight_layer_f2f3 = Lambda((lambda x: x * connection_weight_f2f3), name = "connection_weight_f2f3")
    multiply_weight_layer_f3c = Lambda((lambda x: x * connection_weight_f3c), name = "connection_weight_f3c")

    # one input layer
    input_layer = Input(shape = (num_vars,))

    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog1_output_layer")(input_layer)
    f1 = Activation(activation='relu')(f1)
    f1f2 = multiply_weight_layer_f1f2(f1)
    connection_f2 = Lambda(add_first_node_layers,name="F1_F2")(f1f2)

    # second fog node
    f2 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog2_input_layer")(connection_f2)
    f2 = Activation(activation='relu')(f2)
    f2 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog2_output_layer")(f2)
    f2 = Activation(activation='relu')(f2)
    f2f3 = multiply_weight_layer_f2f3(f2)
    connection_f3 = Lambda(add_first_node_layers,name="F1F2_F3")(f2f3)

    # third fog node
    f3 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog3_input_layer")(connection_f3)
    f3 = Activation(activation='relu')(f3)
    f3 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog3_layer_1")(f3)
    f3 = Activation(activation='relu')(f3)
    f3 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog3_output_layer")(f3)
    f3 = Activation(activation='relu')(f3)
    f3c = multiply_weight_layer_f3c(f3)
    connection_cloud = Lambda(add_first_node_layers,name="F2F3_FC")(f3c)

    # cloud node
    cloud = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="cloud_input_layer")(connection_cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="cloud_layer_1")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="cloud_layer_2")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="cloud_layer_3")(cloud)
    cloud = Activation(activation='relu')(cloud)
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    # TODO: make a lambda function that checks if there is no data connection flow and does smart random guessing
    model = Model(inputs=input_layer, outputs=output_layer)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model