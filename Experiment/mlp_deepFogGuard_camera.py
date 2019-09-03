from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation, add
from Experiment.LambdaLayers import add_node_layers
from Experiment.mlp_Vanilla_camera import define_MLP_architecture_cloud, define_MLP_architecture_edge, define_MLP_architecture_fog_with_one_layer, define_MLP_architecture_fog_with_two_layers
from keras.models import Model
import random
def define_deepFogGuard_MLP(input_shape,
                            num_classes,
                            hidden_units,
                            survivability_setting = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], # survivability of a node between 0 and 1, [f1,f2,f3,f4,e1,e2,e3,e4]
                            skip_hyperconnection_config = [1,1,1,1,1,1,1], # binary representating if a skip hyperconnection is alive. source of skip hyperconnections: [e1,e2,e3,e4,f3,f4,f2]
                            hyperconnection_weights_scheme = 1):
    """Define a deepFogGuard model.
    ### Naming Convention
        ex: f2f1 = connection between fog node 2 and fog node 1
    ### Arguments
        num_vars (int): specifies number of variables from the data, used to determine input size.
        num_classes (int): specifies number of classes to be outputted by the model
        hidden_units (int): specifies number of hidden units per layer in network
        survivability_setting (list): specifies the survival rate of each node in the network
        skip_hyperconnection_config (list): specifies the alive skip hyperconnections in the network, default value is [1,1,1,1,1,1,1]
        hyperconnection_weights_scheme (int): determines if the hyperconnections should be based on surivive_rates, 1: weighted 1, 2: weighted by weighted survival of multiple nodes, 3: weighted by survival of single node only, 4: weights are randomly weighted from 0-1, 5: weights are randomly weighted from 0-10
    ### Returns
        Keras Model object
    """

    hyperconnection_weight = {} # define the hyperconnection_weight as dictionary
    connection_ends = ["e1f2","e2f2","e3f2","e4f2","f3f1","f4f1","f2c","e1f3","e2f4","e3f4","e4f4","f3f2","f4f2","f2f1","f1c"]

    hyperconnection_weight = set_hyperconnection_weights(hyperconnection_weight, hyperconnection_weights_scheme, survivability_setting, skip_hyperconnection_config, connection_ends)
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


    # edge nodes
    edge1_output = define_MLP_deepFogGuard_architecture_edge(input_edge1, hidden_units, "edge1_output_layer")
    edge2_output = define_MLP_deepFogGuard_architecture_edge(input_edge2, hidden_units, "edge2_output_layer")
    edge3_output = define_MLP_deepFogGuard_architecture_edge(input_edge3, hidden_units, "edge3_output_layer")
    edge4_output = define_MLP_deepFogGuard_architecture_edge(input_edge4, hidden_units, "edge4_output_layer")

    # fog node 4
    fog4_output = define_MLP_deepFogGuard_architecture_fog4(edge2_output, edge3_output, edge4_output, hidden_units, multiply_hyperconnection_weight_layer)

    # fog node 3
    fog3_output = define_MLP_deepFogGuard_architecture_fog3(edge1_output, hidden_units, multiply_hyperconnection_weight_layer)
    
    # fog node 2
    fog2_output = define_MLP_deepFogGuard_architecture_fog2(edge1_output, edge2_output, edge3_output, edge4_output, fog3_output, fog4_output, hidden_units, multiply_hyperconnection_weight_layer)

    # fog node 1
    fog1_output = define_MLP_deepFogGuard_architecture_fog1(fog2_output, fog3_output, fog4_output, hidden_units, multiply_hyperconnection_weight_layer)

    # cloud node
    cloud_output = define_MLP_deepFogGuard_architecture_cloud(fog2_output, fog1_output, hidden_units, num_classes, multiply_hyperconnection_weight_layer)

    model = Model(inputs=[img_input_1,img_input_2,img_input_3,img_input_4,img_input_5,img_input_6], outputs=cloud_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def set_hyperconnection_weights(hyperconnection_weight, hyperconnection_weights_scheme, survivability_setting, skip_hyperconnection_config, connection_ends):
    # weighted by 1
    if hyperconnection_weights_scheme == 1: 
        for connection_end in connection_ends:
            hyperconnection_weight[connection_end] = 1
    # normalized survivability
    elif hyperconnection_weights_scheme == 2:
        fog2_input_surviability = survivability_setting[2] + survivability_setting[3] + survivability_setting[4] \
                                + survivability_setting[5] + survivability_setting[6] + survivability_setting[7]
        hyperconnection_weight["e1f2"] = survivability_setting[4] / fog2_input_surviability
        hyperconnection_weight["e2f2"] = survivability_setting[5] / fog2_input_surviability
        hyperconnection_weight["e3f2"] = survivability_setting[6] / fog2_input_surviability
        hyperconnection_weight["e4f2"] = survivability_setting[7] / fog2_input_surviability
        hyperconnection_weight["f3f2"] = survivability_setting[2] / fog2_input_surviability
        hyperconnection_weight["f4f2"] = survivability_setting[3] / fog2_input_surviability
        fog1_input_surviability = survivability_setting[1] + survivability_setting[2] + survivability_setting[3] 
        hyperconnection_weight["f2f1"] = survivability_setting[1] / fog1_input_surviability
        hyperconnection_weight["f3f1"] = survivability_setting[2] / fog1_input_surviability
        hyperconnection_weight["f4f1"] = survivability_setting[3] / fog1_input_surviability
        hyperconnection_weight["e1f3"] = 1
        fog4_input_surviability = survivability_setting[5] + survivability_setting[6] + survivability_setting[7]
        hyperconnection_weight["e2f4"] = survivability_setting[5] / fog4_input_surviability
        hyperconnection_weight["e3f4"] = survivability_setting[6] / fog4_input_surviability
        hyperconnection_weight["e4f4"] = survivability_setting[7] / fog4_input_surviability
        cloud_input_surviability = survivability_setting[0] + survivability_setting[1] 
        hyperconnection_weight["f1c"] = survivability_setting[0] / cloud_input_surviability
        hyperconnection_weight["f2c"] = survivability_setting[1] / cloud_input_surviability
    # survivability
    elif hyperconnection_weights_scheme == 3:
        hyperconnection_weight["e1f2"] = survivability_setting[4]
        hyperconnection_weight["e2f2"] = survivability_setting[5]
        hyperconnection_weight["e3f2"] = survivability_setting[6]
        hyperconnection_weight["e4f2"] = survivability_setting[7]
        hyperconnection_weight["f3f1"] = survivability_setting[2]
        hyperconnection_weight["f4f1"] = survivability_setting[3]
        hyperconnection_weight["f2c"] = survivability_setting[1]
        hyperconnection_weight["e1f3"] = survivability_setting[4]
        hyperconnection_weight["e2f4"] = survivability_setting[5]
        hyperconnection_weight["e3f4"] = survivability_setting[6]
        hyperconnection_weight["e4f4"] = survivability_setting[7]
        hyperconnection_weight["f3f2"] = survivability_setting[2]
        hyperconnection_weight["f4f2"] = survivability_setting[3]
        hyperconnection_weight["f2f1"] = survivability_setting[1]
        hyperconnection_weight["f1c"] = survivability_setting[0]
    # randomly weighted between 0 and 1
    elif hyperconnection_weights_scheme == 4:
        for connection_end in connection_ends:
            hyperconnection_weight[connection_end] = random.uniform(0,1)
    # randomly weighted between 0 and 10
    elif hyperconnection_weights_scheme == 5:
        for connection_end in connection_ends:
            hyperconnection_weight[connection_end] = random.uniform(0,10)
    # randomly weighted by .5
    elif hyperconnection_weights_scheme == 6:
        for connection_end in connection_ends:
            hyperconnection_weight[connection_end] = 0.5
    else:
        raise ValueError("Incorrect scheme value")
    hyperconnection_weight = remove_skip_hyperconnection_for_sensitvity_experiment(skip_hyperconnection_config, hyperconnection_weight)
    return (hyperconnection_weight)

def remove_skip_hyperconnection_for_sensitvity_experiment(skip_hyperconnection_config, hyperconnection_weight):
    # take away the skip hyperconnection if the value in hyperconnections array is 0
    # skip_hyperconnection_config = [e1,e2,e3,e4,f3,f4,f2]
    # from edge node 1 to fog node 2
    if skip_hyperconnection_config[0] == 0:
        hyperconnection_weight["e1f2"] = 0
    #from edge node 2 to fog node 2
    if skip_hyperconnection_config[1] == 0:
        hyperconnection_weight["e2f2"] = 0
    # from edge node 3 to fog node 2
    if skip_hyperconnection_config[2] == 0:
        hyperconnection_weight["e3f2"] = 0
    # from edge node 4 to fog node 2
    if skip_hyperconnection_config[4] == 0:
        hyperconnection_weight["e4f2"] = 0
    # from fog node 3 to fog node 1
    if skip_hyperconnection_config[5] == 0:
        hyperconnection_weight["f3f1"] = 0
    # from fog node 4 to fog node 1
    if skip_hyperconnection_config[6] == 0:
        hyperconnection_weight["f4f1"] = 0
    # from fog node 2 to cloud node
    if skip_hyperconnection_config[6] == 0:
        hyperconnection_weight["f2c"] = 0
    return hyperconnection_weight
 
def define_hyperconnection_weight_lambda_layers(hyperconnection_weight, connection_ends):
    # define lambdas for multiplying node weights by connection weight
    multiply_hyperconnection_weight_layer = {}
    for connection_end in connection_ends:
        multiply_hyperconnection_weight_layer[connection_end] = Lambda((lambda x: x * hyperconnection_weight[connection_end]), name = "hyperconnection_weight_"+connection_end)
    return multiply_hyperconnection_weight_layer

def define_MLP_deepFogGuard_architecture_edge(edge_input, hidden_units, output_layer_name):
    edge_output = define_MLP_architecture_edge(edge_input, hidden_units, output_layer_name)
    return edge_output

def define_MLP_deepFogGuard_architecture_fog4(edge2_output, edge3_output, edge4_output, hidden_units, multiply_hyperconnection_weight_layer = None):
    if multiply_hyperconnection_weight_layer == None or multiply_hyperconnection_weight_layer["e2f4"] == None or multiply_hyperconnection_weight_layer["e3f4"] == None or multiply_hyperconnection_weight_layer["e4f4"] == None:
        fog4_input = Lambda(add_node_layers,name="fog4_Input_lambda")([edge2_output,edge3_output, edge4_output])
    else:
        fog4_input = Lambda(add_node_layers,name="fog4_Input_lambda")([multiply_hyperconnection_weight_layer["e2f4"](edge2_output),multiply_hyperconnection_weight_layer["e3f4"](edge3_output), multiply_hyperconnection_weight_layer["e4f4"](edge4_output)])
    fog4_output = define_MLP_architecture_fog_with_two_layers(fog4_input, hidden_units, "fog4_output_layer", "fog4_input_layer")
    return fog4_output

def define_MLP_deepFogGuard_architecture_fog3(edge1_output, hidden_units, multiply_hyperconnection_weight_layer = None):
    if multiply_hyperconnection_weight_layer != None and multiply_hyperconnection_weight_layer["e1f3"] != None:
        edge1_output = multiply_hyperconnection_weight_layer["e1f3"](edge1_output)
    fog3_output = define_MLP_architecture_fog_with_one_layer(edge1_output, hidden_units, "fog3_output_layer")
    return fog3_output

def define_MLP_deepFogGuard_architecture_fog2(edge1_output, edge2_output, edge3_output, edge4_output, fog3_output, fog4_output, hidden_units, multiply_hyperconnection_weight_layer = None):
    if multiply_hyperconnection_weight_layer == None or multiply_hyperconnection_weight_layer["e1f2"] == None or multiply_hyperconnection_weight_layer["e2f2"] == None or multiply_hyperconnection_weight_layer["e3f2"] == None or multiply_hyperconnection_weight_layer["e4f2"] == None or multiply_hyperconnection_weight_layer["f3f2"] == None or multiply_hyperconnection_weight_layer["f4f2"] == None:
        fog2_input = Lambda(add_node_layers,name="fog2_Input_lambda")([edge1_output, edge2_output, edge3_output, edge4_output, fog3_output, fog4_output])
    else: 
        fog2_input = Lambda(add_node_layers,name="fog2_Input_lambda")([multiply_hyperconnection_weight_layer["e1f2"](edge1_output), multiply_hyperconnection_weight_layer["e2f2"](edge2_output), multiply_hyperconnection_weight_layer["e3f2"](edge3_output), multiply_hyperconnection_weight_layer["e4f2"](edge4_output), multiply_hyperconnection_weight_layer["f3f2"](fog3_output), multiply_hyperconnection_weight_layer["f4f2"](fog4_output)])
    fog2_output = define_MLP_architecture_fog_with_two_layers(fog2_input, hidden_units, "fog2_output_layer", "fog2_input_layer")
    return fog2_output

def define_MLP_deepFogGuard_architecture_fog1(fog2_output, fog3_output, fog4_output, hidden_units, multiply_hyperconnection_weight_layer = None):
    if multiply_hyperconnection_weight_layer == None or multiply_hyperconnection_weight_layer["f2f1"] == None or multiply_hyperconnection_weight_layer["f3f1"] == None or multiply_hyperconnection_weight_layer["f4f1"]:
        fog1_input = Lambda(add_node_layers,name="fog1_Input_lambda")([fog2_output, fog3_output, fog4_output])
    else:
        fog1_input = Lambda(add_node_layers,name="fog1_Input_lambda")([multiply_hyperconnection_weight_layer["f2f1"](fog2_output), multiply_hyperconnection_weight_layer["f3f1"](fog3_output), multiply_hyperconnection_weight_layer["f4f1"](fog4_output)])
    fog1_output = define_MLP_architecture_fog_with_two_layers(fog1_input, hidden_units, "fog1_output_layer", "fog1_input_layer")
    return fog1_output


def define_MLP_deepFogGuard_architecture_cloud(fog2_output, fog1_output, hidden_units, num_classes, multiply_hyperconnection_weight_layer = None):
    if multiply_hyperconnection_weight_layer == None or multiply_hyperconnection_weight_layer["f1c"] == None or multiply_hyperconnection_weight_layer["f2c"] == None:
        cloud_input = Lambda(add_node_layers,name="Cloud_Input")([fog1_output,fog2_output])
    else:
        cloud_input = Lambda(add_node_layers,name="Cloud_Input")([multiply_hyperconnection_weight_layer["f1c"](fog1_output), multiply_hyperconnection_weight_layer["f2c"](fog2_output)])
    cloud_output = define_MLP_architecture_cloud(cloud_input, hidden_units, num_classes)
    return cloud_output