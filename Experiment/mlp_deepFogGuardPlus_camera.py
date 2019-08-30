from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
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
                            standard_dropout = False):
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

    # IoT node
    img_input_1 = Input(shape = input_shape)    
    img_input_2 = Input(shape = input_shape)
    img_input_3 = Input(shape = input_shape)
    img_input_4 = Input(shape = input_shape)    
    img_input_5 = Input(shape = input_shape) 
    img_input_6 = Input(shape = input_shape) 
    
    # nodewise droput definitions
    edge_failure_lambda, fog_failure_lambda, e_dropout_multiply, f_dropout_multiply = MLP_nodewise_dropout_definitions(failout_survival_setting, standard_dropout)

    # edge nodes
    edge1_output = define_MLP_deepFogGuard_architecture_edge(img_input_1, hidden_units, "edge1_output_layer", e_dropout_multiply[1])
    edge1_output = edge_failure_lambda[1](edge1_output)
    edge2_output = define_MLP_deepFogGuard_architecture_edge(img_input_2, hidden_units, "edge2_output_layer", e_dropout_multiply[2])
    edge2_output = edge_failure_lambda[2](edge2_output)
    edge3_output = define_MLP_deepFogGuard_architecture_edge(img_input_3, hidden_units, "edge3_output_layer", e_dropout_multiply[3])
    edge3_output = edge_failure_lambda[3](edge3_output)
    edge4_output = define_MLP_deepFogGuard_architecture_edge(img_input_4, hidden_units, "edge4_output_layer", e_dropout_multiply[4])
    edge4_output = edge_failure_lambda[4](edge4_output)

    # fog node 4
    fog4_output = define_MLP_deepFogGuard_architecture_fog4(edge2_output, edge3_output, edge4_output, hidden_units, multiply_dropout_layer_f4 = f_dropout_multiply[4])
    fog4_output = fog_failure_lambda[4](fog4_output)

    # fog node 3
    fog3_output = define_MLP_deepFogGuard_architecture_fog3(edge1_output, hidden_units, multiply_dropout_layer_f3 = f_dropout_multiply[3])
    fog3_output = fog_failure_lambda[3](fog3_output)

    # fog node 2
    fog2_output = define_MLP_deepFogGuard_architecture_fog2(edge1_output, edge2_output, edge3_output, edge4_output, fog3_output, fog4_output, hidden_units, multiply_dropout_layer_f2 = f_dropout_multiply[2])
    fog2_output = fog_failure_lambda[2](fog2_output)

    # fog node 1
    fog1_output = define_MLP_deepFogGuard_architecture_fog1(fog2_output, fog3_output, fog4_output, hidden_units, multiply_dropout_layer_f1 = f_dropout_multiply[1])
    fog1_output = fog_failure_lambda[1](fog1_output)

    # cloud node
    cloud_output = define_MLP_deepFogGuard_architecture_cloud(fog2_output, fog1_output, hidden_units, num_classes)

    model = Model(inputs=img_input, outputs=cloud_output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def MLP_nodewise_dropout_definitions(failout_survival_setting, standard_dropout = False):
    fog_survivability = [0] * 5
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
    # node-wise dropout occurs only during training
    for i in range(1,5):
        edge_rand[i] = K.in_train_phase(K.random_uniform(shape=K.variable(0).shape), K.variable(0))
        fog_rand[i] = K.in_train_phase(K.random_uniform(shape=K.variable(0).shape), K.variable(0))
    # define lambda for failure, only fail during training
    edge_failure_lambda = {}
    fog_failure_lambda = {}
    for i in range(1,5):
        edge_failure_lambda[i] = layers.Lambda(lambda x : K.switch(K.greater(edge_rand[i],edge_survivability_keras[i]), x * 0, x),name = 'e'+str(i)+'_failure_lambda')
        fog_failure_lambda[i] = layers.Lambda(lambda x : K.switch(K.greater(fog_rand[i],fog_survivability_keras[i]), x * 0, x),name = 'f'+str(i)+'_failure_lambda')
    if standard_dropout:
        # define lambda for standard dropout (adjust output weights based on node survivability, w' = w * s)
        e_dropout_multiply = {}
        f_dropout_multiply = {}
        for i in range(1,5):
            e_dropout_multiply[i] = layers.Lambda(lambda x : K.in_train_phase(x, x * edge_survivability[i]),name = 'e'+str(i)+'_standard_dropout_lambda') 
            f_dropout_multiply[i] = layers.Lambda(lambda x : K.in_train_phase(x, x * fog_survivability[i]),name = 'f'+str(i)+'_standard_dropout_lambda')
        return edge_failure_lambda, fog_failure_lambda, e_dropout_multiply, f_dropout_multiply
    else:
        return edge_failure_lambda, fog_failure_lambda, None, None