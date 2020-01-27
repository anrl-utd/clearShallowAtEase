from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation, add
import keras.backend as K
import keras.layers as layers
from Experiment.LambdaLayers import add_node_layers
from Experiment.mlp_deepFogGuard_camera import define_MLP_deepFogGuard_architecture_edge, define_MLP_deepFogGuard_architecture_fog3, define_MLP_deepFogGuard_architecture_fog4
from Experiment.mlp_Vanilla_camera import define_MLP_architecture_cloud, define_MLP_architecture_fog_with_two_layers
from Experiment.mlp_deepFogGuard_camera import connection_ends, set_hyperconnection_weights, define_hyperconnection_weight_lambda_layers
from Experiment.mlp_deepFogGuard_camera import default_skip_hyperconnection_config

from keras.models import Model
from keras.backend import constant
import random 

from Experiment.Custom_Layers import Failout, InputMux
def define_ResiliNet_MLP(input_shape,
                            num_classes,
                            hidden_units,
                            failout_survival_setting = [.9,.9,.9,.9,.9,.9,.9,.9],
                            reliability_setting = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], 
                            skip_hyperconnection_config = default_skip_hyperconnection_config, 
                            hyperconnection_weights_scheme = 1):
    """Define a ResiliNet model.
    ### Naming Convention
        ex: f2f1 = connection between fog node 2 and fog node 1
    ### Arguments
        num_vars (int): specifies number of variables from the data, used to determine input size.
        num_classes (int): specifies number of classes to be outputted by the model
        hidden_units (int): specifies number of hidden units per layer in network
        failout_survival_setting (list): specifies the failout survival rate of each node in the network
        skip_hyperconnections (list): specifies the alive skip hyperconnections in the network, default value is [1,1,1]
    ### Returns
        Keras Model object
    """

    hyperconnection_weight = {} # define the hyperconnection_weight as dictionary

    hyperconnection_weight = set_hyperconnection_weights(hyperconnection_weight, hyperconnection_weights_scheme, reliability_setting, skip_hyperconnection_config, connection_ends)
    multiply_hyperconnection_weight_layer = define_hyperconnection_weight_lambda_layers(hyperconnection_weight, connection_ends)
   
    # IoT Node (input image)
    img_input_1 = Input(shape = input_shape)    
    img_input_2 = Input(shape = input_shape)
    img_input_3 = Input(shape = input_shape)
    img_input_4 = Input(shape = input_shape)    
    img_input_5 = Input(shape = input_shape) 
    img_input_6 = Input(shape = input_shape) 
    
    input_edge1 = add([img_input_1,img_input_2])
    input_edge2 = img_input_3
    input_edge3 = add([img_input_4,img_input_5])
    input_edge4 = img_input_6
    
    # failout definitions
    edge_failure_lambda, fog_failure_lambda = MLP_failout_definitions(failout_survival_setting)

    # edge nodes
    edge1_output = define_MLP_ResiliNet_architecture_edge(input_edge1, hidden_units, "edge1_output_layer")
    edge1_output = edge_failure_lambda[1](edge1_output)
    edge2_output = define_MLP_ResiliNet_architecture_edge(input_edge2, hidden_units, "edge2_output_layer")
    edge2_output = edge_failure_lambda[2](edge2_output)
    edge3_output = define_MLP_ResiliNet_architecture_edge(input_edge3, hidden_units, "edge3_output_layer")
    edge3_output = edge_failure_lambda[3](edge3_output)
    edge4_output = define_MLP_ResiliNet_architecture_edge(input_edge4, hidden_units, "edge4_output_layer")
    edge4_output = edge_failure_lambda[4](edge4_output)

    # fog node 4
    fog4_output = define_MLP_ResiliNet_architecture_fog4(edge2_output, edge3_output, edge4_output, hidden_units, multiply_hyperconnection_weight_layer)
    fog4_output = fog_failure_lambda[4](fog4_output)

    # fog node 3
    fog3 = Lambda(lambda x: x * 1,name="node4_input")(edge1_output)
    fog3_output = define_MLP_ResiliNet_architecture_fog3(fog3, hidden_units, multiply_hyperconnection_weight_layer)
    fog3_output = fog_failure_lambda[3](fog3_output)

    # fog node 2
    fog2_output = define_MLP_ResiliNet_architecture_fog2(edge1_output, edge2_output, edge3_output, edge4_output, fog3_output, fog4_output, hidden_units, fog_failure_lambda[3], fog_failure_lambda[4], multiply_hyperconnection_weight_layer)
    fog2_output = fog_failure_lambda[2](fog2_output)

    # fog node 1
    fog1_output = define_MLP_ResiliNet_architecture_fog1(fog2_output, fog3_output, fog4_output, hidden_units, fog_failure_lambda[2], multiply_hyperconnection_weight_layer)
    fog1_output = fog_failure_lambda[1](fog1_output)

    # cloud node
    cloud_output = define_MLP_ResiliNet_architecture_cloud(fog2_output, fog1_output, hidden_units, num_classes, fog_failure_lambda[1], multiply_hyperconnection_weight_layer)

    model = Model(inputs=[img_input_1,img_input_2,img_input_3,img_input_4,img_input_5,img_input_6], outputs=cloud_output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def MLP_failout_definitions(failout_survival_setting):
    fog_reliability = [0] * 5

    fog_reliability[1] = failout_survival_setting[0]
    fog_reliability[2] = failout_survival_setting[1]
    fog_reliability[3] = failout_survival_setting[2]
    fog_reliability[4] = failout_survival_setting[3]

    edge_reliability = [0] * 5
    edge_reliability[1] = failout_survival_setting[4]
    edge_reliability[2] = failout_survival_setting[5]
    edge_reliability[3] = failout_survival_setting[6]
    edge_reliability[4] = failout_survival_setting[7]
    
    edge_failure_lambda = {}
    fog_failure_lambda = {}
    for i in range(1,5):
        edge_failure_lambda[i] = Failout(edge_reliability[i])
        fog_failure_lambda[i] = Failout(fog_reliability[i])
    return edge_failure_lambda, fog_failure_lambda

def define_MLP_ResiliNet_architecture_edge(edge_input, hidden_units, output_layer_name):
    return define_MLP_deepFogGuard_architecture_edge(edge_input, hidden_units, output_layer_name)

def define_MLP_ResiliNet_architecture_fog4(edge2_output, edge3_output, edge4_output, hidden_units, multiply_hyperconnection_weight_layer):
    return define_MLP_deepFogGuard_architecture_fog4(edge2_output, edge3_output, edge4_output, hidden_units, multiply_hyperconnection_weight_layer)

def define_MLP_ResiliNet_architecture_fog3(edge1_output, hidden_units, multiply_hyperconnection_weight_layer):
    return define_MLP_deepFogGuard_architecture_fog3(edge1_output, hidden_units, multiply_hyperconnection_weight_layer)

def define_MLP_ResiliNet_architecture_fog2(edge1_output, edge2_output, edge3_output, edge4_output, fog3_output, fog4_output, hidden_units, fog3_failure_lambda, fog4_failure_lambda, multiply_hyperconnection_weight_layer = None):
    if multiply_hyperconnection_weight_layer == None or multiply_hyperconnection_weight_layer["e1f2"] == None or multiply_hyperconnection_weight_layer["e2f2"] == None or multiply_hyperconnection_weight_layer["e3f2"] == None or multiply_hyperconnection_weight_layer["e4f2"] == None or multiply_hyperconnection_weight_layer["f3f2"] == None or multiply_hyperconnection_weight_layer["f4f2"] == None:
        fog2_input_left = Lambda(InputMux(fog3_failure_lambda.has_failed),name="node3_input_left")([edge1_output, fog3_output]) 
        skip_hyperconnections = add([edge2_output, edge3_output, edge4_output])
        fog2_input_right = Lambda(InputMux(fog4_failure_lambda.has_failed),name="node3_input_right")([skip_hyperconnections, fog4_output]) 
    else: 
        fog2_input_left = Lambda(InputMux(fog3_failure_lambda.has_failed),name="node3_input_left")([multiply_hyperconnection_weight_layer["e1f2"](edge1_output), multiply_hyperconnection_weight_layer["f3f2"](fog3_output)]) 
        skip_hyperconnections = add([multiply_hyperconnection_weight_layer["e2f2"](edge2_output), multiply_hyperconnection_weight_layer["e3f2"](edge3_output), multiply_hyperconnection_weight_layer["e4f2"](edge4_output)])
        fog2_input_right = Lambda(InputMux(fog4_failure_lambda.has_failed),name="node3_input_right")([skip_hyperconnections, multiply_hyperconnection_weight_layer["f4f2"](fog4_output)]) 
    
    fog2_input = Lambda(add_node_layers,name="node3_input")([fog2_input_left, fog2_input_right])
    fog2_output = define_MLP_architecture_fog_with_two_layers(fog2_input, hidden_units, "fog2_output_layer", "fog2_input_layer")
    return fog2_output

def define_MLP_ResiliNet_architecture_fog1(fog2_output, fog3_output, fog4_output, hidden_units, fog2_failure_lambda, multiply_hyperconnection_weight_layer = None):
    if multiply_hyperconnection_weight_layer == None or multiply_hyperconnection_weight_layer["f2f1"] == None or multiply_hyperconnection_weight_layer["f3f1"] == None or multiply_hyperconnection_weight_layer["f4f1"] == None:
        skip_hyperconnections = add([fog3_output, fog4_output])
        fog1_input = Lambda(InputMux(fog2_failure_lambda.has_failed),name="node2_input")([skip_hyperconnections, fog2_output]) 
    else:
        skip_hyperconnections = add([multiply_hyperconnection_weight_layer["f3f1"](fog3_output), multiply_hyperconnection_weight_layer["f4f1"](fog4_output)])
        fog1_input = Lambda(InputMux(fog2_failure_lambda.has_failed),name="node2_input")([skip_hyperconnections, multiply_hyperconnection_weight_layer["f2f1"](fog2_output)]) 
    fog1_output = define_MLP_architecture_fog_with_two_layers(fog1_input, hidden_units, "fog1_output_layer", "fog1_input_layer")
    return fog1_output


def define_MLP_ResiliNet_architecture_cloud(fog2_output, fog1_output, hidden_units, num_classes, fog1_failure_lambda, multiply_hyperconnection_weight_layer):
    if multiply_hyperconnection_weight_layer == None or multiply_hyperconnection_weight_layer["f1c"] == None or multiply_hyperconnection_weight_layer["f2c"] == None:
        cloud_input = Lambda(InputMux(fog1_failure_lambda.has_failed),name="node1_input")([fog2_output, fog1_output])
    else:
        cloud_input = Lambda(InputMux(fog1_failure_lambda.has_failed),name="node1_input")([multiply_hyperconnection_weight_layer["f2c"](fog2_output), multiply_hyperconnection_weight_layer["f1c"](fog1_output)])
    cloud_output = define_MLP_architecture_cloud(cloud_input, hidden_units, num_classes)
    return cloud_output