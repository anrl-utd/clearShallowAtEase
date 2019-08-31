from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
import keras.backend as K
import keras.layers as layers
from Experiment.LambdaLayers import add_node_layers
from Experiment.mlp_deepFogGuard_health import define_MLP_deepFogGuard_architecture_cloud, define_MLP_deepFogGuard_architecture_edge, define_MLP_deepFogGuard_architecture_fog1, define_MLP_deepFogGuard_architecture_fog2, define_MLP_deepFogGuard_architecture_IoT

from keras.models import Model
from keras.backend import constant
import random 

def define_deepFogGuardPlus_MLP(num_vars,
                            num_classes,
                            hidden_units,
                            failout_survival_setting = [1.0,1.0,1.0]
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

    # IoT node
    img_input = Input(shape = (num_vars,))
    iot_output = define_MLP_deepFogGuard_architecture_IoT(img_input, hidden_units)

    # nodewise droput definitions
    edge_failure_lambda, fog2_failure_lambda, fog1_failure_lambda  = MLP_nodewise_dropout_definitions(failout_survival_setting)
  
    # edge node
    edge_output = define_MLP_deepFogGuard_architecture_edge(iot_output, hidden_units)
    edge_output = edge_failure_lambda(edge_output)

    # fog node 2
    fog2_output = define_MLP_deepFogGuard_architecture_fog2(iot_output, edge_output, hidden_units)
    fog2_output = fog2_failure_lambda(fog2_output)

    # fog node 1
    fog1_output = define_MLP_deepFogGuard_architecture_fog1(edge_output, fog2_output, hidden_units)
    fog1_output = fog1_failure_lambda(fog1_output)

    # cloud node
    cloud_output = define_MLP_deepFogGuard_architecture_cloud(fog2_output, fog1_output, hidden_units, num_classes)


    model = Model(inputs=img_input, outputs=cloud_output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def MLP_nodewise_dropout_definitions(failout_survival_setting):
    edge_survivability = failout_survival_setting[0]
    fog2_survivability = failout_survival_setting[1]
    fog1_survivability = failout_survival_setting[2]

    # variables for node-wise dropout
    edge_rand = K.variable(0)
    fog2_rand = K.variable(0)
    fog1_rand = K.variable(0)
    # variables for node-wise dropout
    edge_survivability_keras = K.variable(edge_survivability)
    fog2_survivability_keras = K.variable(fog2_survivability)
    fog1_survivability_keras = K.variable(fog1_survivability)
    # node-wise dropout occurs only during training
    K.set_learning_phase(1)
    if K.learning_phase():
        # seeds so the random_number is different for each node 
        edge_rand = K.random_uniform(shape=edge_rand.shape)
        fog2_rand = K.random_uniform(shape=fog2_rand.shape)
        fog1_rand = K.random_uniform(shape=fog2_rand.shape)
    # define lambda for failure, only fail during training
    edge_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(edge_rand,edge_survivability_keras), x * 0, x),name = 'e_failure_lambda')
    fog2_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(fog2_rand,fog2_survivability_keras), x * 0, x),name = 'f2_failure_lambda')
    fog1_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(fog1_rand,fog1_survivability_keras), x * 0, x),name = 'f1_failure_lambda')
    return edge_failure_lambda, fog2_failure_lambda, fog1_failure_lambda