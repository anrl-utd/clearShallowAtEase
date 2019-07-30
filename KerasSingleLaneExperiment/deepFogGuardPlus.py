from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
import keras.backend as K
from KerasSingleLaneExperiment.LambdaLayers import add_node_layers
from keras.models import Model
from keras.backend import constant
import random 

def define_deepFogGuardPlus(num_vars,num_classes,hidden_units,survive_rates,skip_hyperconnections = [1,1,1]):
    """Define a deepFogGuardPlus model.
    ### Naming Convention
        ex: f2f1 = connection between fog node 2 and fog node 1
    ### Arguments
        num_vars (int): specifies number of variables from the data, used to determine input size.
        num_classes (int): specifies number of classes to be outputted by the model
        hidden_units (int): specifies number of hidden units per layer in network
        survive_rates (list): specifies the survival rate of each node in the network
        skip_hyperconnections (list): specifies the alive skip hyperconnections in the network, default value is [1,1,1]
    ### Returns
        Keras Model object
    """
    # weights calculated by survival rates, uncomment if want to weigh by survival rates
    # connection_weight_ef1 = survive_rates[0] / (survive_rates[0] + survive_rates[1])
    # connection_weight_f2f1 = survive_rates[1] / (survive_rates[0] + survive_rates[1])
    # connection_weight_f2c = survive_rates[1] / (survive_rates[1] + survive_rates[2])
    # connection_weight_f1c = survive_rates[2] / (survive_rates[1] + survive_rates[2])

    # all hyperconnection weights are weighted 1
    connection_weight_IoTf2  = 1
    connection_weight_ef2 = 1
    connection_weight_ef1 = 1
    connection_weight_f2f1 = 1
    connection_weight_f2c = 1
    connection_weight_f1c = 1

    # take away the skip hyperconnection if the value in hyperconnections array is 0
    if skip_hyperconnections[0] == 0:
        connection_weight_IoTf2 = 0
    if skip_hyperconnections[1] == 0:
        connection_weight_ef1 = 0
    if skip_hyperconnections[2] == 0:
        connection_weight_f2c = 0
        
    # define lambdas for multiplying node weights by connection weight
    multiply_weight_layer_IoTf2 = Lambda((lambda x: x * connection_weight_IoTf2), name = "connection_weight_IoTf2")
    multiply_weight_layer_ef2 = Lambda((lambda x: x * connection_weight_ef2), name = "connection_weight_ef2")
    multiply_weight_layer_ef1 = Lambda((lambda x: x * connection_weight_ef1), name = "connection_weight_ef1")
    multiply_weight_layer_f2f1 = Lambda((lambda x: x * connection_weight_f2f1), name = "connection_weight_f2f1")
    multiply_weight_layer_f2c = Lambda((lambda x: x * connection_weight_f2c), name = "connection_weight_f2c")
    multiply_weight_layer_f1c = Lambda((lambda x: x * connection_weight_f1c), name = "connection_weight_f1c")

    # variables for active guard 
    e_rand = K.variable(0)
    f2_rand = K.variable(0)
    f1_rand = K.variable(0)
    e_survive_rate = K.variable(survive_rates[0])
    f2_survive_rate = K.variable(survive_rates[1])
    f1_survive_rate = K.variable(survive_rates[2])

    # set training phase to true to enable dropout
    K.set_learning_phase(1)
    if K.learning_phase():
        # seeds so the random_number is different for each fog node 
        e_rand = K.random_uniform(shape=e_rand.shape,seed=7)
        f2_rand = K.random_uniform(shape=f2_rand.shape,seed=11)
        f1_rand = K.random_uniform(shape=f1_rand.shape,seed=42)

    # define lambda for fog failure, failures are only during training
    e_failure_lambda = Lambda(lambda x : K.switch(K.greater(e_rand,e_survive_rate), x * 0, x),name = 'e_failure_lambda')
    f2_failure_lambda = Lambda(lambda x : K.switch(K.greater(f2_rand,f2_survive_rate), x * 0, x),name = 'f2_failure_lambda')
    f1_failure_lambda = Lambda(lambda x : K.switch(K.greater(f1_rand,f1_survive_rate), x * 0, x),name = 'f1_failure_lambda')

    # one input layer
    IoT_node = Input(shape = (num_vars,))

    # edge node
    e = Dense(units=hidden_units,activation='linear',name="edge_output_layer")(IoT_node)
    e = Activation(activation='relu')(e)
    e = e_failure_lambda(e)
    ef2 = multiply_weight_layer_ef2(e)
    # use a linear Dense layer to transform input into the shape needed for the network
    duplicated_input = Dense(units=hidden_units,name="duplicated_input",activation='linear')(IoT_node)
    IoTf2 = multiply_weight_layer_IoTf2(duplicated_input)
    connection_f2 = Lambda(add_node_layers,name="F2_Input")([ef2,IoTf2])

    # fog node 2
    f2 = Dense(units=hidden_units,activation='linear',name="fog2_input_layer")(connection_f2)
    f2 = Activation(activation='relu')(f2)
    f2 = Dense(units=hidden_units,activation='linear',name="fog2_output_layer")(f2)
    f2 = Activation(activation='relu')(f2)
    f2 = f2_failure_lambda(f2)
    ef1 = multiply_weight_layer_ef1(e)
    f2f1 = multiply_weight_layer_f2f1(f2)
    connection_f1 = Lambda(add_node_layers,name="F1_Input")([ef1,f2f1])

    # fog node 1
    f1 = Dense(units=hidden_units,activation='linear',name="fog1_input_layer")(connection_f1)
    f1 = Activation(activation='relu')(f1)
    f1 = Dense(units=hidden_units,activation='linear',name="fog1_layer_1")(f1)
    f1 = Activation(activation='relu')(f1)
    f1 = Dense(units=hidden_units,activation='linear',name="fog1_output_layer")(f1)
    f1 = Activation(activation='relu')(f1)
    f1 = f1_failure_lambda(f1)
    f2c = multiply_weight_layer_f2c(f2)
    f1c = multiply_weight_layer_f1c(f1)
    connection_cloud = Lambda(add_node_layers,name="Cloud_Input")([f2c,f1c])

    # cloud node
    cloud = Dense(units=hidden_units,activation='linear',name="cloud_input_layer")(connection_cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activation='linear',name="cloud_layer_1")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activation='linear',name="cloud_layer_2")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activation='linear',name="cloud_layer_3")(cloud)
    cloud = Activation(activation='relu')(cloud)
    # one output layer
    normal_output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)

    model = Model(inputs=IoT_node, outputs=normal_output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model