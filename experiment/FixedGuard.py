from keras.models import Sequential
from keras.layers import Dense,Input,add,multiply,Lambda, BatchNormalization, Activation, Dropout
import keras.backend as K
from experiment.LambdaLayers import add_first_node_layers,add_node_layers
from keras import regularizers,optimizers
from keras.models import Model


# returns fixed guard model with 10 hidden layers
# f1 = fog node 2 = 1st hidden layer
# f2 = fog node 2 = 2nd and 3rd hidden layer
# f3 = fog node 3 = 4th-6th hidden layers
# c = cloud node = 7th-10th hidden layer and output layer 
def define_model_with_connections(num_vars,num_classes,hidden_units,regularization,survive_rates):
    # calculate connection weights
    # naming convention:
    # ex: f1f2 = connection between fog node 1 and fog node 2
    # ex: f2c = connection between fog node 2 and cloud node

    connection_weight_f1f2 = 1

    #normal weights
    connection_weight_f1f3 = survive_rates[0] / (survive_rates[0] + survive_rates[1])
    connection_weight_f2f3 = survive_rates[1] / (survive_rates[0] + survive_rates[1])
    connection_weight_f2c = survive_rates[1] / (survive_rates[1] + survive_rates[2])
    connection_weight_f3c = survive_rates[2] / (survive_rates[1] + survive_rates[2])

    # inverted weights
    # connection_weight_f1f3 =  (survive_rates[0] + survive_rates[1]) / survive_rates[0] 
    # connection_weight_f2f3 = (survive_rates[0] + survive_rates[1]) / survive_rates[1]
    # connection_weight_f2c = (survive_rates[1] + survive_rates[2]) / survive_rates[1] 
    # connection_weight_f3c = (survive_rates[1] + survive_rates[2]) / survive_rates[2]

    # set to constant
    # connection_weight_f1f3 = 1
    # connection_weight_f2f3 = 1
    # connection_weight_f2c = 1
    # connection_weight_f3c = 1

    # define lambdas for multiplying node weights by connection weight
    multiply_weight_layer_f1f2 = Lambda((lambda x: x * connection_weight_f1f2), name = "connection_weight_f1f2")
    multiply_weight_layer_f1f3 = Lambda((lambda x: x * connection_weight_f1f3), name = "connection_weight_f1f3")
    multiply_weight_layer_f2f3 = Lambda((lambda x: x * connection_weight_f2f3), name = "connection_weight_f2f3")
    multiply_weight_layer_f2c = Lambda((lambda x: x * connection_weight_f2c), name = "connection_weight_f2c")
    multiply_weight_layer_f3c = Lambda((lambda x: x * connection_weight_f3c), name = "connection_weight_f3c")

    # one input layer
    input_layer = Input(shape = (num_vars,))

    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog1_output_layer")(input_layer)
    f1 = BatchNormalization()(f1)
    f1 = Activation(activation='relu')(f1)
    f1f2 = multiply_weight_layer_f1f2(f1)
    connection_f2 = Lambda(add_first_node_layers,name="F1_F2")(f1f2)

    # second fog node
    f2 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog2_input_layer")(connection_f2)
    f2 = BatchNormalization()(f2)
    f2 = Activation(activation='relu')(f2)
    f2 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog2_output_layer")(f2)
    f2 = BatchNormalization()(f2)
    f2 = Activation(activation='relu')(f2)
    f1f3 = multiply_weight_layer_f1f3(f1)
    f2f3 = multiply_weight_layer_f2f3(f2)
    connection_f3 = Lambda(add_node_layers,name="F1F2_F3")([f1f3,f2f3])

    # third fog node
    f3 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog3_input_layer")(connection_f3)
    f3 = BatchNormalization()(f3)
    f3 = Activation(activation='relu')(f3)
    f3 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog3_layer_1")(f3)
    f3 = BatchNormalization()(f3)
    f3 = Activation(activation='relu')(f3)
    f3 = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="fog3_output_layer")(f3)
    f3 = BatchNormalization()(f3)
    f3 = Activation(activation='relu')(f3)
    f2c = multiply_weight_layer_f2c(f2)
    f3c = multiply_weight_layer_f3c(f3)
    connection_cloud = Lambda(add_node_layers,name="F2F3_FC")([f2c,f3c])

    # cloud node
    cloud = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="cloud_input_layer")(connection_cloud)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="cloud_layer_1")(cloud)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="cloud_layer_2")(cloud)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activity_regularizer=regularizers.l1(regularization),name="cloud_layer_3")(cloud)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    # TODO: make a lambda functoin that checks if there is no data connection flow and does smart random guessing
    model = Model(inputs=input_layer, outputs=output_layer)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# returns fixed guard model with 10 hidden layers
# f1 = fog node 2 = 1st hidden layer
# f2 = fog node 2 = 2nd and 3rd hidden layer
# f3 = fog node 3 = 4th-6th hidden layers
# c = cloud node = 7th-10th hidden layer and output layer 
def define_model_with_nofogbatchnorm_connections(num_vars,num_classes,hidden_units,regularization,survive_rates):
    # calculate connection weights
    # naming convention:
    # ex: f1f2 = connection between fog node 1 and fog node 2
    # ex: f2c = connection between fog node 2 and cloud node

    connection_weight_f1f2 = 1

    #normal weights
    connection_weight_f1f3 = survive_rates[0] / (survive_rates[0] + survive_rates[1])
    connection_weight_f2f3 = survive_rates[1] / (survive_rates[0] + survive_rates[1])
    connection_weight_f2c = survive_rates[1] / (survive_rates[1] + survive_rates[2])
    connection_weight_f3c = survive_rates[2] / (survive_rates[1] + survive_rates[2])

    # inverted weights
    # connection_weight_f1f3 =  (survive_rates[0] + survive_rates[1]) / survive_rates[1] 
    # connection_weight_f2f3 = (survive_rates[0] + survive_rates[1]) / survive_rates[0]
    # connection_weight_f2c = (survive_rates[1] + survive_rates[2]) / survive_rates[2] 
    # connection_weight_f3c = (survive_rates[1] + survive_rates[2]) / survive_rates[1]

    # set to constant
    # set aux connections to 0
    # connection_weight_f1f3 = 0
    # connection_weight_f2f3 = 1
    # connection_weight_f2c = 0
    # connection_weight_f3c = 1

    # define lambdas for multiplying node weights by connection weight
    multiply_weight_layer_f1f2 = Lambda((lambda x: x * connection_weight_f1f2), name = "connection_weight_f1f2")
    multiply_weight_layer_f1f3 = Lambda((lambda x: x * connection_weight_f1f3), name = "connection_weight_f1f3")
    multiply_weight_layer_f2f3 = Lambda((lambda x: x * connection_weight_f2f3), name = "connection_weight_f2f3")
    multiply_weight_layer_f2c = Lambda((lambda x: x * connection_weight_f2c), name = "connection_weight_f2c")
    multiply_weight_layer_f3c = Lambda((lambda x: x * connection_weight_f3c), name = "connection_weight_f3c")

    # dropout rate for all dropout layers 
    dropout = 0
    # one input layer
    input_layer = Input(shape = (num_vars,))
    #input_layer = Dropout(dropout)(input_layer)
    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,name="fog1_output_layer",activation='relu',kernel_initializer = 'he_normal')(input_layer)
    f1 = Dropout(dropout,seed=7)(f1)
    f1f2 = multiply_weight_layer_f1f2(f1)
    connection_f2 = Lambda(add_first_node_layers,name="F1_F2")(f1f2)

    # second fog node
    f2 = Dense(units=hidden_units,name="fog2_input_layer",activation='relu',kernel_initializer = 'he_normal')(connection_f2)
    f2 = Dropout(dropout,seed=7)(f2)
    f2 = Dense(units=hidden_units,name="fog2_output_layer",activation='relu',kernel_initializer = 'he_normal')(f2)
    f2 = Dropout(dropout,seed=7)(f2)
    f1f3 = multiply_weight_layer_f1f3(f1)
    f2f3 = multiply_weight_layer_f2f3(f2)
    connection_f3 = Lambda(add_node_layers,name="F1F2_F3")([f1f3,f2f3])

    # third fog node
    f3 = Dense(units=hidden_units,name="fog3_input_layer",activation='relu',kernel_initializer = 'he_normal')(connection_f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f3 = Dense(units=hidden_units,name="fog3_layer_1",activation='relu',kernel_initializer = 'he_normal')(f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f3 = Dense(units=hidden_units,name="fog3_output_layer",activation='relu',kernel_initializer = 'he_normal')(f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f2c = multiply_weight_layer_f2c(f2)
    f3c = multiply_weight_layer_f3c(f3)
    connection_cloud = Lambda(add_node_layers,name="F2F3_FC")([f2c,f3c])

    # cloud node
    cloud = Dense(units=hidden_units,name="cloud_input_layer",kernel_initializer = 'he_normal')(connection_cloud)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_1",kernel_initializer = 'he_normal')(cloud)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_2",kernel_initializer = 'he_normal')(cloud)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_3",kernel_initializer = 'he_normal')(cloud)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    # TODO: make a lambda function that checks if there is no data connection flow and does smart random guessing
    model = Model(inputs=input_layer, outputs=output_layer)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_model_with_nofogbatchnorm_connections_extrainput(num_vars,num_classes,hidden_units,regularization,survive_rates):
    # calculate connection weights
    # naming convention:
    # ex: f1f2 = connection between fog node 1 and fog node 2
    # ex: f2c = connection between fog node 2 and cloud node

    connection_weight_f1f2 = 1

    #normal weights
    connection_weight_f1f3 = survive_rates[0] / (survive_rates[0] + survive_rates[1])
    connection_weight_f2f3 = survive_rates[1] / (survive_rates[0] + survive_rates[1])
    connection_weight_f2c = survive_rates[1] / (survive_rates[1] + survive_rates[2])
    connection_weight_f3c = survive_rates[2] / (survive_rates[1] + survive_rates[2])

    # define lambdas for multiplying node weights by connection weight
    multiply_weight_layer_f1f2 = Lambda((lambda x: x * connection_weight_f1f2), name = "connection_weight_f1f2")
    multiply_weight_layer_f1f3 = Lambda((lambda x: x * connection_weight_f1f3), name = "connection_weight_f1f3")
    multiply_weight_layer_f2f3 = Lambda((lambda x: x * connection_weight_f2f3), name = "connection_weight_f2f3")
    multiply_weight_layer_f2c = Lambda((lambda x: x * connection_weight_f2c), name = "connection_weight_f2c")
    multiply_weight_layer_f3c = Lambda((lambda x: x * connection_weight_f3c), name = "connection_weight_f3c")

    # dropout rate for all dropout layers 
    dropout = 0
    # one input layer
    input_layer = Input(shape = (num_vars,))
    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,name="fog1_output_layer",activation='relu')(input_layer)
    f1 = Dropout(dropout,seed=7)(f1)
    f1f2 = multiply_weight_layer_f1f2(f1)
    duplicated_input = Dense(units=hidden_units,name="duplicated_input",activation='relu')(input_layer)
    connection_f2 = Lambda(add_node_layers,name="F1_F2")([f1f2,duplicated_input])
 
    # second fog node
    f2 = Dense(units=hidden_units,name="fog2_input_layer",activation='relu')(connection_f2)
    f2 = Dropout(dropout,seed=7)(f2)
    f2 = Dense(units=hidden_units,name="fog2_output_layer",activation='relu')(f2)
    f2 = Dropout(dropout,seed=7)(f2)
    f1f3 = multiply_weight_layer_f1f3(f1)
    f2f3 = multiply_weight_layer_f2f3(f2)
    connection_f3 = Lambda(add_node_layers,name="F1F2_F3")([f1f3,f2f3])

    # third fog node
    f3 = Dense(units=hidden_units,name="fog3_input_layer",activation='relu')(connection_f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f3 = Dense(units=hidden_units,name="fog3_layer_1",activation='relu')(f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f3 = Dense(units=hidden_units,name="fog3_output_layer",activation='relu')(f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f2c = multiply_weight_layer_f2c(f2)
    f3c = multiply_weight_layer_f3c(f3)
    connection_cloud = Lambda(add_node_layers,name="F2F3_FC")([f2c,f3c])

    # cloud node
    cloud = Dense(units=hidden_units,name="cloud_input_layer")(connection_cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_1")(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_2")(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_3")(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    model = Model(inputs=input_layer, outputs=output_layer)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_fixed_guard_model_experiment2(num_vars,num_classes,hidden_units,regularization,hyperconnections):
    # calculate connection weights
    # naming convention:
    # ex: f1f2 = connection between fog node 1 and fog node 2
    # ex: f2c = connection between fog node 2 and cloud node

    #normal weights
    connection_weight_inputf2 = 1
    connection_weight_f1f2 = 1
    connection_weight_f1f3 = 1
    connection_weight_f2f3 = 1
    connection_weight_f2c = 1
    connection_weight_f3c = 1

     # take away the hyperconnectin if the value in hyperconnections array is 0
    if hyperconnections[0] == 0:
        connection_weight_inputf2 = 0
    if hyperconnections[1] == 0:
        connection_weight_f1f3 = 0
    if hyperconnections[2] == 0:
        connection_weight_f2c = 0

    # define lambdas for multiplying node weights by connection weight
    multiply_weight_layer_inputf2 = Lambda((lambda x: x * connection_weight_inputf2), name = "connection_weight_inputf2")
    multiply_weight_layer_f1f2 = Lambda((lambda x: x * connection_weight_f1f2), name = "connection_weight_f1f2")
    multiply_weight_layer_f1f3 = Lambda((lambda x: x * connection_weight_f1f3), name = "connection_weight_f1f3")
    multiply_weight_layer_f2f3 = Lambda((lambda x: x * connection_weight_f2f3), name = "connection_weight_f2f3")
    multiply_weight_layer_f2c = Lambda((lambda x: x * connection_weight_f2c), name = "connection_weight_f2c")
    multiply_weight_layer_f3c = Lambda((lambda x: x * connection_weight_f3c), name = "connection_weight_f3c")

    # dropout rate for all dropout layers 
    dropout = 0

    # one input layer
    input_layer = Input(shape = (num_vars,))
    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,name="fog1_output_layer",activation='relu')(input_layer)
    f1 = Dropout(dropout,seed=7)(f1)
    f1f2 = multiply_weight_layer_f1f2(f1)
    duplicated_input = Dense(units=hidden_units,name="duplicated_input",activation='relu')(input_layer)
    input_f2 = multiply_weight_layer_inputf2(duplicated_input)
    connection_f2 = Lambda(add_node_layers,name="F1_F2")([f1f2,input_f2])
 
    # second fog node
    f2 = Dense(units=hidden_units,name="fog2_input_layer",activation='relu')(connection_f2)
    f2 = Dropout(dropout,seed=7)(f2)
    f2 = Dense(units=hidden_units,name="fog2_output_layer",activation='relu')(f2)
    f2 = Dropout(dropout,seed=7)(f2)
    f1f3 = multiply_weight_layer_f1f3(f1)
    f2f3 = multiply_weight_layer_f2f3(f2)
    connection_f3 = Lambda(add_node_layers,name="F1F2_F3")([f1f3,f2f3])

    # third fog node
    f3 = Dense(units=hidden_units,name="fog3_input_layer",activation='relu')(connection_f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f3 = Dense(units=hidden_units,name="fog3_layer_1",activation='relu')(f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f3 = Dense(units=hidden_units,name="fog3_output_layer",activation='relu')(f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f2c = multiply_weight_layer_f2c(f2)
    f3c = multiply_weight_layer_f3c(f3)
    connection_cloud = Lambda(add_node_layers,name="F2F3_FC")([f2c,f3c])

    # cloud node
    cloud = Dense(units=hidden_units,name="cloud_input_layer")(connection_cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_1")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_2")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_3")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    model = Model(inputs=input_layer, outputs=output_layer)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_fixed_guard_baseline_model(num_vars,num_classes,hidden_units,regularization,survive_rates):
     # calculate connection weights
    # naming convention:
    # ex: f1f2 = connection between fog node 1 and fog node 2
    # ex: f2c = connection between fog node 2 and cloud node

    connection_weight_f1f2 = 1

    #normal weights
    connection_weight_f1f3 = 1
    connection_weight_f2f3 = 1
    connection_weight_f2c = 1
    connection_weight_f3c = 1


    # define lambdas for multiplying node weights by connection weight
    multiply_weight_layer_f1f2 = Lambda((lambda x: x * connection_weight_f1f2), name = "connection_weight_f1f2")
    multiply_weight_layer_f1f3 = Lambda((lambda x: x * connection_weight_f1f3), name = "connection_weight_f1f3")
    multiply_weight_layer_f2f3 = Lambda((lambda x: x * connection_weight_f2f3), name = "connection_weight_f2f3")
    multiply_weight_layer_f2c = Lambda((lambda x: x * connection_weight_f2c), name = "connection_weight_f2c")
    multiply_weight_layer_f3c = Lambda((lambda x: x * connection_weight_f3c), name = "connection_weight_f3c")

    # dropout rate for all dropout layers 
    dropout = 0
    # one input layer
    input_layer = Input(shape = (num_vars,))
    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,name="fog1_output_layer",activation='relu')(input_layer)
    f1 = Dropout(dropout,seed=7)(f1)
    f1f2 = multiply_weight_layer_f1f2(f1)
    duplicated_input = Dense(units=hidden_units,name="duplicated_input",activation='relu')(input_layer)
    connection_f2 = Lambda(add_node_layers,name="F1_F2")([f1f2,duplicated_input])

    # second fog node
    f2 = Dense(units=hidden_units,name="fog2_input_layer",activation='relu')(connection_f2)
    f2 = Dropout(dropout,seed=7)(f2)
    f2 = Dense(units=hidden_units,name="fog2_output_layer",activation='relu')(f2)
    f2 = Dropout(dropout,seed=7)(f2)
    f1f3 = multiply_weight_layer_f1f3(f1)
    f2f3 = multiply_weight_layer_f2f3(f2)
    connection_f3 = Lambda(add_node_layers,name="F1F2_F3")([f1f3,f2f3])

    # third fog node
    f3 = Dense(units=hidden_units,name="fog3_input_layer",activation='relu')(connection_f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f3 = Dense(units=hidden_units,name="fog3_layer_1",activation='relu')(f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f3 = Dense(units=hidden_units,name="fog3_output_layer",activation='relu')(f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f2c = multiply_weight_layer_f2c(f2)
    f3c = multiply_weight_layer_f3c(f3)
    connection_cloud = Lambda(add_node_layers,name="F2F3_FC")([f2c,f3c])

    # cloud node
    cloud = Dense(units=hidden_units,name="cloud_input_layer")(connection_cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_1")(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_2")(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_3")(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    model = Model(inputs=input_layer, outputs=output_layer)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model