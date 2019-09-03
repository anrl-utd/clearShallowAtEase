from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation, add
import keras.backend as K
import keras.layers as layers
from Experiment.LambdaLayers import add_node_layers
from Experiment.mlp_deepFogGuard_camera import define_MLP_deepFogGuard_architecture_cloud, define_MLP_deepFogGuard_architecture_edge, define_MLP_deepFogGuard_architecture_fog1, define_MLP_deepFogGuard_architecture_fog2, define_MLP_deepFogGuard_architecture_fog3, define_MLP_deepFogGuard_architecture_fog4

from keras.models import Model
from keras.backend import constant
import random 

def define_deepFogGuardPlus_MLP(input_shape,
                            num_classes,
                            hidden_units,
                            failout_survival_setting = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                            ):
    """Define a deepFogGuardPlus model.
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
    
    # nodewise droput definitions
    edge_failure_lambda, fog_failure_lambda = MLP_nodewise_dropout_definitions(failout_survival_setting)

    # edge nodes
    edge1_output = define_MLP_deepFogGuard_architecture_edge(input_edge1, hidden_units, "edge1_output_layer")
    edge1_output = edge_failure_lambda[1](edge1_output)
    edge2_output = define_MLP_deepFogGuard_architecture_edge(input_edge2, hidden_units, "edge2_output_layer")
    edge2_output = edge_failure_lambda[2](edge2_output)
    edge3_output = define_MLP_deepFogGuard_architecture_edge(input_edge3, hidden_units, "edge3_output_layer")
    edge3_output = edge_failure_lambda[3](edge3_output)
    edge4_output = define_MLP_deepFogGuard_architecture_edge(input_edge4, hidden_units, "edge4_output_layer")
    edge4_output = edge_failure_lambda[4](edge4_output)

    # fog node 4
    fog4_output = define_MLP_deepFogGuard_architecture_fog4(edge2_output, edge3_output, edge4_output, hidden_units)
    fog4_output = fog_failure_lambda[4](fog4_output)

    # fog node 3
    fog3_output = define_MLP_deepFogGuard_architecture_fog3(edge1_output, hidden_units)
    fog3_output = fog_failure_lambda[3](fog3_output)

    # fog node 2
    fog2_output = define_MLP_deepFogGuard_architecture_fog2(edge1_output, edge2_output, edge3_output, edge4_output, fog3_output, fog4_output, hidden_units)
    fog2_output = fog_failure_lambda[2](fog2_output)

    # fog node 1
    fog1_output = define_MLP_deepFogGuard_architecture_fog1(fog2_output, fog3_output, fog4_output, hidden_units)
    fog1_output = fog_failure_lambda[1](fog1_output)

    # cloud node
    cloud_output = define_MLP_deepFogGuard_architecture_cloud(fog2_output, fog1_output, hidden_units, num_classes)

    model = Model(inputs=[img_input_1,img_input_2,img_input_3,img_input_4,img_input_5,img_input_6], outputs=cloud_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def MLP_nodewise_dropout_definitions(failout_survival_setting):
    fog_survivability = [0] * 5
    print(failout_survival_setting)
    fog_survivability[1] = failout_survival_setting[0]
    fog_survivability[2] = failout_survival_setting[1]
    fog_survivability[3] = failout_survival_setting[2]
    fog_survivability[4] = failout_survival_setting[3]

    edge_survivability = [0] * 5
    edge_survivability[1] = failout_survival_setting[4]
    edge_survivability[2] = failout_survival_setting[5]
    edge_survivability[3] = failout_survival_setting[6]
    edge_survivability[4] = failout_survival_setting[7]
    
    # variables for node-wise dropout
    edge_rand = [0] * 5
    fog_rand = [0] * 5
    edge_survivability_keras = [0] * 5
    fog_survivability_keras = [0] * 5
    for i in range(1,5):
        edge_rand[i] = K.variable(0)
        fog_rand[i] = K.variable(0)
        edge_survivability_keras[i] = K.variable(edge_survivability[i])
        fog_survivability_keras[i] = K.variable(fog_survivability[i])
    K.set_learning_phase(1)
    if K.learning_phase():
        # seeds so the random_number is different for each node 
        for i in range(1,5):
            edge_rand[i] = K.random_uniform(shape=edge_rand[i].shape)
            fog_rand[i] = K.random_uniform(shape=fog_rand[i].shape)
    # define lambda for failure, only fail during training
    edge_failure_lambda = {}
    fog_failure_lambda = {}
    for i in range(1,5):
        edge_failure_lambda[i] = layers.Lambda(lambda x : K.switch(K.greater(edge_rand[i],edge_survivability_keras[i]), x * 0, x),name = 'e'+str(i)+'_failure_lambda')
        fog_failure_lambda[i] = layers.Lambda(lambda x : K.switch(K.greater(fog_rand[i],fog_survivability_keras[i]), x * 0, x),name = 'f'+str(i)+'_failure_lambda')
    return edge_failure_lambda, fog_failure_lambda