from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
import keras.backend as K
import keras.layers as layers
from Experiment.LambdaLayers import add_node_layers
from Experiment.mlp_deepFogGuard_health import define_MLP_deepFogGuard_architecture_cloud, define_MLP_deepFogGuard_architecture_edge, define_MLP_deepFogGuard_architecture_fog1, define_MLP_deepFogGuard_architecture_fog2, define_MLP_deepFogGuard_architecture_IoT
from Experiment.mlp_deepFogGuard_health import define_hyperconnection_weight_lambda_layers, set_hyperconnection_weights
from Experiment.Failout import Failout
from Experiment.mlp_deepFogGuard_health import default_skip_hyperconnection_config
from keras.models import Model
from keras.backend import constant
import random 


def define_ResiliNet_MLP(num_vars,
                            num_classes,
                            hidden_units,
                            failout_survival_setting = [0.95,0.95,0.95],
                            reliability_setting = [1.0,1.0,1.0], 
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

    hyperconnection_weight_IoTe, hyperconnection_weight_IoTf2,hyperconnection_weight_ef2,hyperconnection_weight_ef1,hyperconnection_weight_f2f1, hyperconnection_weight_f2c, hyperconnection_weight_f1c = set_hyperconnection_weights(
        hyperconnection_weights_scheme, 
        reliability_setting, 
        skip_hyperconnection_config)
    multiply_hyperconnection_weight_layer_IoTe, multiply_hyperconnection_weight_layer_IoTf2, multiply_hyperconnection_weight_layer_ef2, multiply_hyperconnection_weight_layer_ef1, multiply_hyperconnection_weight_layer_f2f1, multiply_hyperconnection_weight_layer_f2c, multiply_hyperconnection_weight_layer_f1c = define_hyperconnection_weight_lambda_layers(
        hyperconnection_weight_IoTe,
        hyperconnection_weight_IoTf2, 
        hyperconnection_weight_ef2, 
        hyperconnection_weight_ef1, 
        hyperconnection_weight_f2f1, 
        hyperconnection_weight_f2c, 
        hyperconnection_weight_f1c)

    # IoT node
    iot_output = Input(shape = (num_vars,))
    iot_skip_output = define_MLP_deepFogGuard_architecture_IoT(iot_output, hidden_units)

    # failout definitions
    edge_failure_lambda, fog2_failure_lambda, fog1_failure_lambda  = MLP_failout_definitions(failout_survival_setting)

    # edge node
    edge_output = define_MLP_deepFogGuard_architecture_edge(iot_output, hidden_units, multiply_hyperconnection_weight_layer_IoTe)
    edge_output = edge_failure_lambda(edge_output)

    # fog node 2
    fog2_output = define_MLP_deepFogGuard_architecture_fog2(iot_skip_output, edge_output, hidden_units, multiply_hyperconnection_weight_layer_IoTf2, multiply_hyperconnection_weight_layer_ef2)
    fog2_output = fog2_failure_lambda(fog2_output)

    # fog node 1
    fog1_output = define_MLP_deepFogGuard_architecture_fog1(edge_output, fog2_output, hidden_units, multiply_hyperconnection_weight_layer_ef1, multiply_hyperconnection_weight_layer_f2f1)
    fog1_output = fog1_failure_lambda(fog1_output)

    # cloud node
    cloud_output = define_MLP_deepFogGuard_architecture_cloud(fog2_output, fog1_output, hidden_units, num_classes, multiply_hyperconnection_weight_layer_f1c, multiply_hyperconnection_weight_layer_f2c)


    model = Model(inputs=iot_output, outputs=cloud_output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def MLP_failout_definitions(failout_survival_setting):
    edge_reliability = failout_survival_setting[0]
    fog2_reliability = failout_survival_setting[1]
    fog1_reliability = failout_survival_setting[2]

    edge_failure_lambda = Failout(edge_reliability)
    fog2_failure_lambda = Failout(fog2_reliability)
    fog1_failure_lambda = Failout(fog1_reliability)
    return edge_failure_lambda, fog2_failure_lambda, fog1_failure_lambda
