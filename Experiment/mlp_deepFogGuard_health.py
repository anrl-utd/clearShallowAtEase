from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
from Experiment.LambdaLayers import add_node_layers
from Experiment.mlp_Vanilla_health import define_MLP_architecture_cloud, define_MLP_architecture_edge, define_MLP_architecture_fog1, define_MLP_architecture_fog2
from keras.models import Model
import random
def define_deepFogGuard_MLP(num_vars,
                            num_classes,
                            hidden_units,
                            reliability_setting = [1.0,1.0,1.0], # reliability of a node between 0 and 1, [f1,f2,e1]
                            skip_hyperconnection_config = [1,1,1], # binary representating if a skip hyperconnection is alive [g1,e1,f2]
                            hyperconnection_weights_scheme = 1):
    """Define a deepFogGuard model.
    ### Naming Convention
        ex: f2f1 = connection between fog node 2 and fog node 1
    ### Arguments
        num_vars (int): specifies number of variables from the data, used to determine input size.
        num_classes (int): specifies number of classes to be outputted by the model
        hidden_units (int): specifies number of hidden units per layer in network
        reliability_setting (list): specifies the reliability of each node in the network
        skip_hyperconnection_config (list): specifies the alive skip hyperconnections in the network, default value is [1,1,1]
        hyperconnection_weights (list): specifies the probability, default value is [1,1,1]
        hyperconnection_weights_scheme (int): determines if the hyperconnections should be based on surivive_rates
    ### Returns
        Keras Model object
    """

    hyperconnection_weight_IoTf2,hyperconnection_weight_ef2,hyperconnection_weight_ef1,hyperconnection_weight_f2f1, hyperconnection_weight_f2c, hyperconnection_weight_f1c = set_hyperconnection_weights(
        hyperconnection_weights_scheme, 
        reliability_setting, 
        skip_hyperconnection_config)
    multiply_hyperconnection_weight_layer_IoTf2, multiply_hyperconnection_weight_layer_ef2, multiply_hyperconnection_weight_layer_ef1, multiply_hyperconnection_weight_layer_f2f1, multiply_hyperconnection_weight_layer_f2c, multiply_hyperconnection_weight_layer_f1c = define_hyperconnection_weight_lambda_layers(
        hyperconnection_weight_IoTf2, 
        hyperconnection_weight_ef2, 
        hyperconnection_weight_ef1, 
        hyperconnection_weight_f2f1, 
        hyperconnection_weight_f2c, 
        hyperconnection_weight_f1c)
   
    # IoT node
    img_input = Input(shape = (num_vars,))
    iot_output = define_MLP_deepFogGuard_architecture_IoT(img_input, hidden_units)

    # edge node
    edge_output = define_MLP_deepFogGuard_architecture_edge(iot_output, hidden_units)
    
    # fog node 2
    fog2_output = define_MLP_deepFogGuard_architecture_fog2(iot_output, edge_output, hidden_units, multiply_hyperconnection_weight_layer_IoTf2, multiply_hyperconnection_weight_layer_ef2)

    # fog node 1
    fog1_output = define_MLP_deepFogGuard_architecture_fog1(edge_output, fog2_output, hidden_units, multiply_hyperconnection_weight_layer_ef1, multiply_hyperconnection_weight_layer_f2f1)

    # cloud node
    cloud_output = define_MLP_deepFogGuard_architecture_cloud(fog2_output, fog1_output, hidden_units, num_classes, multiply_hyperconnection_weight_layer_f1c, multiply_hyperconnection_weight_layer_f2c)

    model = Model(inputs=img_input, outputs=cloud_output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def set_hyperconnection_weights(hyperconnection_weights_scheme, reliability_setting, skip_hyperconnection_config):
    # weighted by 1
    if hyperconnection_weights_scheme == 1: 
        hyperconnection_weight_IoTf2  = 1
        hyperconnection_weight_ef2 = 1
        hyperconnection_weight_ef1 = 1
        hyperconnection_weight_f2f1 = 1
        hyperconnection_weight_f2c = 1
        hyperconnection_weight_f1c = 1
    # normalized reliability
    elif hyperconnection_weights_scheme == 2:   
        hyperconnection_weight_IoTf2  = 1 / (1+reliability_setting[2])
        hyperconnection_weight_ef2 = reliability_setting[2] / (1 + reliability_setting[2])
        hyperconnection_weight_ef1 = reliability_setting[2] / (reliability_setting[2] + reliability_setting[1])
        hyperconnection_weight_f2f1 = reliability_setting[1] / (reliability_setting[2] + reliability_setting[1])
        hyperconnection_weight_f2c = reliability_setting[1] / (reliability_setting[1] + reliability_setting[0])
        hyperconnection_weight_f1c = reliability_setting[0] / (reliability_setting[1] + reliability_setting[0])
    # reliability
    elif hyperconnection_weights_scheme == 3:
        hyperconnection_weight_IoTf2  = 1
        hyperconnection_weight_ef2 = reliability_setting[2] 
        hyperconnection_weight_ef1 = reliability_setting[2] 
        hyperconnection_weight_f2f1 = reliability_setting[1]
        hyperconnection_weight_f2c = reliability_setting[1] 
        hyperconnection_weight_f1c = reliability_setting[0]
    # randomly weighted between 0 and 1
    elif hyperconnection_weights_scheme == 4:
        hyperconnection_weight_IoTf2  = random.uniform(0,1)
        hyperconnection_weight_ef2 = random.uniform(0,1)
        hyperconnection_weight_ef1 = random.uniform(0,1)
        hyperconnection_weight_f2f1 = random.uniform(0,1)
        hyperconnection_weight_f2c = random.uniform(0,1)
        hyperconnection_weight_f1c = random.uniform(0,1)
    # randomly weighted between 0 and 10
    elif hyperconnection_weights_scheme == 5:
        hyperconnection_weight_IoTf2  = random.uniform(0,10)
        hyperconnection_weight_ef2 = random.uniform(0,10)
        hyperconnection_weight_ef1 = random.uniform(0,10)
        hyperconnection_weight_f2f1 = random.uniform(0,10)
        hyperconnection_weight_f2c = random.uniform(0,10)
        hyperconnection_weight_f1c = random.uniform(0,10)
    # randomly weighted by .5
    elif hyperconnection_weights_scheme == 6:
        hyperconnection_weight_IoTf2  = .5
        hyperconnection_weight_ef2 = .5
        hyperconnection_weight_ef1 = .5
        hyperconnection_weight_f2f1 = .5
        hyperconnection_weight_f2c = .5
        hyperconnection_weight_f1c = .5
    else:
        raise ValueError("Incorrect scheme value")
    hyperconnection_weight_IoTf2, hyperconnection_weight_ef1, hyperconnection_weight_f2c = remove_skip_hyperconnection_for_sensitvity_experiment(
        skip_hyperconnection_config, 
        hyperconnection_weight_IoTf2, 
        hyperconnection_weight_ef1, 
        hyperconnection_weight_f2c)
    return (hyperconnection_weight_IoTf2,hyperconnection_weight_ef2,hyperconnection_weight_ef1,hyperconnection_weight_f2f1, hyperconnection_weight_f2c, hyperconnection_weight_f1c)

def remove_skip_hyperconnection_for_sensitvity_experiment(skip_hyperconnection_config, hyperconnection_weight_IoTf2, hyperconnection_weight_ef1, hyperconnection_weight_f2c):
    # take away the skip hyperconnection if the value in hyperconnections array is 0
    # from IoT node to fog node 2
    if skip_hyperconnection_config[0] == 0:
        hyperconnection_weight_IoTf2 = 0
    # from edge node to fog node 1
    if skip_hyperconnection_config[1] == 0:
        hyperconnection_weight_ef1 = 0
    # from fog node 2 to cloud node
    if skip_hyperconnection_config[2] == 0:
        hyperconnection_weight_f2c = 0
    return hyperconnection_weight_IoTf2, hyperconnection_weight_ef1, hyperconnection_weight_f2c
 
def define_hyperconnection_weight_lambda_layers(hyperconnection_weight_IoTf2, hyperconnection_weight_ef2, hyperconnection_weight_ef1, hyperconnection_weight_f2f1, hyperconnection_weight_f2c, hyperconnection_weight_f1c):
    # define lambdas for multiplying node weights by connection weight
    multiply_hyperconnection_weight_layer_IoTf2 = Lambda((lambda x: x * hyperconnection_weight_IoTf2), name = "hyperconnection_weight_IoTf2")
    multiply_hyperconnection_weight_layer_ef2 = Lambda((lambda x: x * hyperconnection_weight_ef2), name = "hyperconnection_weight_ef2")
    multiply_hyperconnection_weight_layer_ef1 = Lambda((lambda x: x * hyperconnection_weight_ef1), name = "hyperconnection_weight_ef1")
    multiply_hyperconnection_weight_layer_f2f1 = Lambda((lambda x: x * hyperconnection_weight_f2f1), name = "hyperconnection_weight_f2f1")
    multiply_hyperconnection_weight_layer_f2c = Lambda((lambda x: x * hyperconnection_weight_f2c), name = "hyperconnection_weight_f2c")
    multiply_hyperconnection_weight_layer_f1c = Lambda((lambda x: x * hyperconnection_weight_f1c), name = "hyperconnection_weight_f1c")
    return multiply_hyperconnection_weight_layer_IoTf2, multiply_hyperconnection_weight_layer_ef2, multiply_hyperconnection_weight_layer_ef1, multiply_hyperconnection_weight_layer_f2f1, multiply_hyperconnection_weight_layer_f2c, multiply_hyperconnection_weight_layer_f1c


def define_MLP_deepFogGuard_architecture_IoT(img_input, hidden_units):
    # use a linear Dense layer to transform input into the shape needed for the network
    iot_output = Dense(units=hidden_units,name="skip_iotfog2",activation='linear')(img_input)
    return iot_output

def define_MLP_deepFogGuard_architecture_edge(iot_output, hidden_units):
    edge_output = define_MLP_architecture_edge(iot_output, hidden_units)
    return edge_output

def define_MLP_deepFogGuard_architecture_fog2(iot_output, edge_output, hidden_units, multiply_hyperconnection_weight_layer_IoTf2 = None, multiply_hyperconnection_weight_layer_ef2 = None):
    if multiply_hyperconnection_weight_layer_IoTf2 == None or multiply_hyperconnection_weight_layer_ef2 == None:
        fog2_input = Lambda(add_node_layers,name="F2_Input")([edge_output,iot_output])
    else:
        fog2_input = Lambda(add_node_layers,name="F2_Input")([multiply_hyperconnection_weight_layer_ef2(edge_output),multiply_hyperconnection_weight_layer_IoTf2(iot_output)])
    fog2_output = define_MLP_architecture_fog2(fog2_input, hidden_units)
    return fog2_output

def define_MLP_deepFogGuard_architecture_fog1(edge_output, fog2_output, hidden_units, multiply_hyperconnection_weight_layer_ef1 = None, multiply_hyperconnection_weight_layer_f2f1 = None):
    if multiply_hyperconnection_weight_layer_ef1 == None or multiply_hyperconnection_weight_layer_f2f1 == None:
        fog1_input = Lambda(add_node_layers,name="F1_Input")([edge_output,fog2_output])
    else:
        fog1_input = Lambda(add_node_layers,name="F1_Input")([multiply_hyperconnection_weight_layer_ef1(edge_output), multiply_hyperconnection_weight_layer_f2f1(fog2_output)])
    fog1_output = define_MLP_architecture_fog1(fog1_input, hidden_units)  
    return fog1_output

def define_MLP_deepFogGuard_architecture_cloud(fog2_output, fog1_output, hidden_units, num_classes, multiply_hyperconnection_weight_layer_f1c = None, multiply_hyperconnection_weight_layer_f2c = None):
    if multiply_hyperconnection_weight_layer_f1c == None or multiply_hyperconnection_weight_layer_f2c == None:
        cloud_input = Lambda(add_node_layers,name="Cloud_Input")([fog1_output,fog2_output])
    else:
        cloud_input = Lambda(add_node_layers,name="Cloud_Input")([multiply_hyperconnection_weight_layer_f1c(fog1_output),multiply_hyperconnection_weight_layer_f2c(fog2_output)])
    cloud_output = define_MLP_architecture_cloud(cloud_input, hidden_units, num_classes)
    return cloud_output