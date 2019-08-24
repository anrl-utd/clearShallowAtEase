from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
from KerasSingleLaneExperiment.LambdaLayers import add_node_layers
from KerasSingleLaneExperiment.mlp_Vanilla_health import define_mlp_architecture_cloud, define_mlp_architecture_edge, define_mlp_architecture_fog1, define_mlp_architecture_fog2
from keras.models import Model
import random
def define_deepFogGuard(num_vars,num_classes,hidden_units,survivability_setting, skip_hyperconnection_config = [1,1,1],hyperconnection_weights_scheme = 1):
    """Define a deepFogGuard model.
    ### Naming Convention
        ex: f2f1 = connection between fog node 2 and fog node 1
    ### Arguments
        num_vars (int): specifies number of variables from the data, used to determine input size.
        num_classes (int): specifies number of classes to be outputted by the model
        hidden_units (int): specifies number of hidden units per layer in network
        survivability_setting (list): specifies the survival rate of each node in the network
        skip_hyperconnection_config (list): specifies the alive skip hyperconnections in the network, default value is [1,1,1]
        hyperconnection_weights (list): specifies the probability, default value is [1,1,1]
        hyperconnection_weights_scheme (int): determines if the hyperconnections should be based on surivive_rates, 1: weighted 1, 2: weighted by weighted survival of multiple nodes, 3: weighted by survival of single node only, 4: weights are randomly weighted from 0-1, 5: weights are randomly weighted from 0-10
    ### Returns
        Keras Model object
    """

    hyperconnection_weight_IoTf2,hyperconnection_weight_ef2,hyperconnection_weight_ef1,hyperconnection_weight_f2f1, hyperconnection_weight_f2c, hyperconnection_weight_f1c = set_hyperconnection_weights(
        hyperconnection_weights_scheme, 
        survivability_setting, 
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
    skip_iotfog2 = define_mlp_deepFogGuard_architecture_IoT(img_input, hidden_units)

    # edge node
   
    # fog node 2
    

    # fog node 1
    f1 = Dense(units=hidden_units,name="fog1_input_layer",activation='relu')(connection_f1)
    f1 = Dense(units=hidden_units,name="fog1_layer_1",activation='relu')(f1)
    f1 = Dense(units=hidden_units,name="fog1_output_layer",activation='relu')(f1)
    f2c = multiply_hyperconnection_weight_layer_f2c(f2)
    f1c = multiply_hyperconnection_weight_layer_f1c(f1)
    connection_cloud = Lambda(add_node_layers,name="Cloud_Input")([f2c,f1c])

    # cloud node
    cloud = Dense(units=hidden_units,name="cloud_input_layer")(connection_cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_1")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_2")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_3")(cloud)
    cloud = Activation(activation='relu')(cloud)
 
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    model = Model(inputs=IoT_node, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def set_hyperconnection_weights(hyperconnection_weights_scheme, survivability_setting, skip_hyperconnection_config):
    # weighted by 1
    if hyperconnection_weights_scheme == 1: 
        hyperconnection_weight_IoTf2  = 1
        hyperconnection_weight_ef2 = 1
        hyperconnection_weight_ef1 = 1
        hyperconnection_weight_f2f1 = 1
        hyperconnection_weight_f2c = 1
        hyperconnection_weight_f1c = 1
    # normalized survivability
    elif hyperconnection_weights_scheme == 2:
        hyperconnection_weight_IoTf2  = 1 / (1+survivability_setting[0])
        hyperconnection_weight_ef2 = survivability_setting[0] / (1 + survivability_setting[0])
        hyperconnection_weight_ef1 = survivability_setting[0] / (survivability_setting[0] + survivability_setting[1])
        hyperconnection_weight_f2f1 = survivability_setting[1] / (survivability_setting[0] + survivability_setting[1])
        hyperconnection_weight_f2c = survivability_setting[1] / (survivability_setting[1] + survivability_setting[2])
        hyperconnection_weight_f1c = survivability_setting[2] / (survivability_setting[1] + survivability_setting[2])
    # survivability
    elif hyperconnection_weights_scheme == 3:
        hyperconnection_weight_IoTf2  = 1
        hyperconnection_weight_ef2 = survivability_setting[0] 
        hyperconnection_weight_ef1 = survivability_setting[0] 
        hyperconnection_weight_f2f1 = survivability_setting[1]
        hyperconnection_weight_f2c = survivability_setting[1] 
        hyperconnection_weight_f1c = survivability_setting[2]
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


def define_mlp_deepFogGuard_architecture_IoT(img_input, hidden_units):
    # use a linear Dense layer to transform input into the shape needed for the network
    skip_iotfog2 = Dense(units=hidden_units,name="skip_iotfog2",activation='linear')(img_input)
    return skip_iotfog2

def define_mlp_deepFogGuard_architecture_edge(iot_output, hidden_units, multiply_dropout_layer_ef = None, multply_dropout_layer_ec = None):
    edge_ouput = define_mlp_architecture_edge(img_input, hidden_units)
    
 
    return edge_output, skip_edgecloud
   
    




def define_mlp_deepFogGuard_architecture_fog2(skip_iotfog2, edge_output, hidden_units, multiply_hyperconnection_weight_layer_IoTf2 = None, multiply_hyperconnection_weight_layer_ef2 = None, multiply_dropout_layer_fc = None):
    if multiply_hyperconnection_weight_layer_IoTf2 == None or multiply_hyperconnection_weight_layer_ef2 == None:
        fog2_input = Lambda(add_node_layers,name="F2_Input")([edge_output,skip_iotfog2])
    else:
        fog2_input = Lambda(add_node_layers,name="F2_Input")([multiply_hyperconnection_weight_layer_ef2(edge_output),multiply_hyperconnection_weight_layer_IoTf2(skip_iotfog2)])
    fog2_output = define_mlp_architecture_fog2(fog2_input, hidden_units)

    
    f1f3 = multiply_hyperconnection_weight_layer_ef1(e)
    f2f3 = multiply_hyperconnection_weight_layer_f2f1(f2)
    connection_f1 = Lambda(add_node_layers,name="F1_Input")([f1f3,f2f3])


    fog = define_cnn_architecture_fog(fog_input,alpha,depth_multiplier)
    # don't need between edge and IoT because 0 will propagate to this node
    # pad from (7,7,256) to (8,8,256)
    fog_output = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)), name = "fogcloud_connection_padding")(fog)
    if(multiply_dropout_layer_fc != None):
        fog_output = multiply_dropout_layer_fc(fog_output)
    return fog_output

def define_mlp_deepFogGuard_architecture_cloud(fog_output, skip_edgecloud, alpha, depth_multiplier, classes, include_top, pooling, multiply_hyperconnection_weight_layer_fc = None, multiply_hyperconnection_weight_layer_ec = None):
    if multiply_hyperconnection_weight_layer_fc == None or multiply_hyperconnection_weight_layer_ec == None:
        cloud_input = layers.add([fog_output, skip_edgecloud], name = "connection_cloud")
    else:
        cloud_input = layers.add([multiply_hyperconnection_weight_layer_fc(fog_output), multiply_hyperconnection_weight_layer_ec(skip_edgecloud)], name = "connection_cloud")
    cloud_output = define_cnn_architecture_cloud(cloud_input,alpha,depth_multiplier,classes,include_top,pooling)
    return cloud_output