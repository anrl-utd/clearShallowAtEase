from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings

import keras.backend as K
import keras.layers as layers
from keras_applications.imagenet_utils import _obtain_input_shape, get_submodules_from_kwargs
import random 
from Experiment.cnn_Vanilla import define_cnn_architecture_IoT, define_cnn_architecture_cloud, define_cnn_architecture_edge, define_cnn_architecture_fog
from Experiment.common_exp_methods import compile_keras_parallel_model

def define_deepFogGuard_CNN(input_shape=None,
                            alpha=1.0,
                            depth_multiplier=1,
                            include_top=True,
                            input_tensor=None,
                            pooling=None,
                            classes=1000,
                            strides = (2,2),
                            skip_hyperconnection_config = [1,1], # binary representating if a skip hyperconnection is alive [e1,IoT]
                            reliability_setting=[1.0,1.0], # reliability of a node between 0 and 1 [f1,e1]
                            hyperconnection_weights_scheme = 1,
                            num_gpus = 1,
                            **kwargs):
    """Instantiates the MobileNet architecture.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
            width multiplier in the MobileNet paper.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution. This
            is called the resolution multiplier in the MobileNet paper.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    hyperconnection_weight_IoTe, hyperconnection_weight_IoTf,hyperconnection_weight_ef,hyperconnection_weight_ec,hyperconnection_weight_fc = set_hyperconnection_weights(
        hyperconnection_weights_scheme, 
        reliability_setting, 
        skip_hyperconnection_config)
    multiply_hyperconnection_weight_layer_IoTe, multiply_hyperconnection_weight_layer_IoTf, multiply_hyperconnection_weight_layer_ef, multiply_hyperconnection_weight_layer_ec, multiply_hyperconnection_weight_layer_fc = define_hyperconnection_weight_lambda_layers(
        hyperconnection_weight_IoTe,
        hyperconnection_weight_IoTf,
        hyperconnection_weight_ef,
        hyperconnection_weight_ec,
        hyperconnection_weight_fc)

    # Determine proper input shape and default size.
    img_input = layers.Input(shape=input_shape)  

    # iot node
    iot_output,skip_iotfog = define_cnn_deepFogGuard_architecture_IoT(input_shape,alpha,img_input, strides = strides)

    # edge node
    edge_output, skip_edgecloud = define_cnn_deepFogGuard_architecture_edge(iot_output,alpha, depth_multiplier, multiply_hyperconnection_weight_layer_IoTe, strides = strides)

    # fog node
    fog_output = define_cnn_deepFogGuard_architecture_fog(skip_iotfog, edge_output, alpha, depth_multiplier, multiply_hyperconnection_weight_layer_IoTf, multiply_hyperconnection_weight_layer_ef, strides = strides)

    # cloud node
    cloud_output = define_cnn_deepFogGuard_architecture_cloud(fog_output, skip_edgecloud, alpha, depth_multiplier, classes, include_top, pooling,multiply_hyperconnection_weight_layer_fc, multiply_hyperconnection_weight_layer_ec)

    model, parallel_model = compile_keras_parallel_model(img_input, cloud_output, num_gpus)
    return model, parallel_model

def define_cnn_deepFogGuard_architecture_IoT(input_shape, alpha, img_input, strides = (2,2)):
    # changed the strides from 2 to 1 since cifar-10 images are smaller
    iot_output = define_cnn_architecture_IoT(img_input,alpha,strides = strides)

    # used stride 1 to match (32,32,3) to (32,32,64)
    # 1x1 conv2d is used to change the filter size (from 3 to 64)
    # cifar-10
    if strides == (1,1):
        skip_iotfog = layers.Conv2D(64,(1,1),strides = 1, use_bias = False, name = "skip_hyperconnection_iotfog")(iot_output)
    elif strides == (2,2):
        skip_iotfog = layers.Conv2D(64,(1,1),strides = 2, use_bias = False, name = "skip_hyperconnection_iotfog")(iot_output)
    else:
        raise ValueError("Invalid stride configuration")
    return iot_output, skip_iotfog

def define_cnn_deepFogGuard_architecture_edge(iot_output, alpha, depth_multiplier, multiply_hyperconnection_weight_layer_IoTe = None, strides = (2,2), edge_failure_lambda = None):
    if multiply_hyperconnection_weight_layer_IoTe != None:
        iot_output = multiply_hyperconnection_weight_layer_IoTe(iot_output)
    edge_output = define_cnn_architecture_edge(iot_output,alpha,depth_multiplier, strides= strides)
    if edge_failure_lambda != None:
         edge_output = edge_failure_lambda(edge_output)
    # used stride 4 to match (31,31,64) to (7,7,256)
    # 1x1 conv2d is used to change the filter size (from 64 to 256)
    skip_edgecloud = layers.Conv2D(256,(1,1),strides = 4, use_bias = False, name = "skip_hyperconnection_edgecloud")(edge_output)
    return edge_output, skip_edgecloud
   

def define_cnn_deepFogGuard_architecture_fog(skip_iotfog, edge_output, alpha, depth_multiplier, multiply_hyperconnection_weight_layer_IoTf = None, multiply_hyperconnection_weight_layer_ef = None, strides = (2,2)):
    if multiply_hyperconnection_weight_layer_IoTf == None or multiply_hyperconnection_weight_layer_ef == None:
        fog_input = layers.add([skip_iotfog, edge_output], name = "connection_fog")
    else:
        fog_input = layers.add([multiply_hyperconnection_weight_layer_IoTf(skip_iotfog), multiply_hyperconnection_weight_layer_ef(edge_output)], name = "connection_fog")
    fog = define_cnn_architecture_fog(fog_input,alpha,depth_multiplier)
    # cnn for imagenet does not need padding
    if strides == (2,2):
        fog_output = fog
    elif strides == (1,1):
        # pad from (7,7,256) to (8,8,256)
        fog_output = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)), name = "fogcloud_connection_padding")(fog)
    else:
        raise ValueError("Incorrect stride value")
    
    return fog_output

def define_cnn_deepFogGuard_architecture_cloud(fog_output, skip_edgecloud, alpha, depth_multiplier, classes, include_top, pooling, multiply_hyperconnection_weight_layer_fc = None, multiply_hyperconnection_weight_layer_ec = None):
    if multiply_hyperconnection_weight_layer_fc == None or multiply_hyperconnection_weight_layer_ec == None:
        cloud_input = layers.add([fog_output, skip_edgecloud], name = "Cloud_Input")
    else:
        cloud_input = layers.add([multiply_hyperconnection_weight_layer_fc(fog_output), multiply_hyperconnection_weight_layer_ec(skip_edgecloud)], name = "Cloud_Input")
    cloud_output = define_cnn_architecture_cloud(cloud_input,alpha,depth_multiplier,classes,include_top,pooling)
    return cloud_output

def set_hyperconnection_weights(hyperconnection_weights_scheme,reliability_setting, skip_hyperconnection_config):
    # weighted by 1
    if hyperconnection_weights_scheme == 1: 
        hyperconnection_weight_IoTe = 1
        hyperconnection_weight_IoTf = 1
        hyperconnection_weight_ef = 1
        hyperconnection_weight_ec = 1
        hyperconnection_weight_fc = 1
    # normalized reliability
    elif hyperconnection_weights_scheme == 2:
        hyperconnection_weight_IoTe = 1
        hyperconnection_weight_IoTf = 1 / (1 + reliability_setting[1])
        hyperconnection_weight_ef = reliability_setting[1] / (1 + reliability_setting[1])
        hyperconnection_weight_ec = reliability_setting[1] / (reliability_setting[0] + reliability_setting[1])
        hyperconnection_weight_fc = reliability_setting[0] / (reliability_setting[0] + reliability_setting[1])
    # reliability
    elif hyperconnection_weights_scheme == 3:
        hyperconnection_weight_IoTe = 1
        hyperconnection_weight_IoTf = 1
        hyperconnection_weight_ef = reliability_setting[1]
        hyperconnection_weight_ec = reliability_setting[1]
        hyperconnection_weight_fc = reliability_setting[0] 
    # randomly weighted between 0 and 1
    elif hyperconnection_weights_scheme == 4:
        hyperconnection_weight_IoTe = random.uniform(0,1)
        hyperconnection_weight_IoTf = random.uniform(0,1)
        hyperconnection_weight_ef = random.uniform(0,1)
        hyperconnection_weight_ec = random.uniform(0,1)
        hyperconnection_weight_fc = random.uniform(0,1)
    # randomly weighted between 0 and 10
    elif hyperconnection_weights_scheme == 5:
        hyperconnection_weight_IoTe = random.uniform(0,10)
        hyperconnection_weight_IoTf = random.uniform(0,10)
        hyperconnection_weight_ef = random.uniform(0,10)
        hyperconnection_weight_ec = random.uniform(0,10)
        hyperconnection_weight_fc = random.uniform(0,10)
    else:
        raise ValueError("Incorrect scheme value")
    hyperconnection_weight_IoTf, hyperconnection_weight_ec = remove_skip_hyperconnection_for_sensitvity_experiment(
        skip_hyperconnection_config, 
        hyperconnection_weight_IoTf,
        hyperconnection_weight_ec)
    return (hyperconnection_weight_IoTe, hyperconnection_weight_IoTf,hyperconnection_weight_ef,hyperconnection_weight_ec,hyperconnection_weight_fc)
  
def remove_skip_hyperconnection_for_sensitvity_experiment(skip_hyperconnection_config, connection_weight_IoTf, connection_weight_ec):
    # take away the skip hyperconnection if the value in hyperconnections array is 0
    # from edge to cloud
    if skip_hyperconnection_config[0] == 0:
        connection_weight_IoTf = 0
    # from iot to fog
    if skip_hyperconnection_config[1] == 0:
        connection_weight_ec = 0
    return connection_weight_IoTf, connection_weight_ec
    
def define_hyperconnection_weight_lambda_layers(hyperconnection_weight_IoTe, hyperconnection_weight_IoTf,hyperconnection_weight_ef,hyperconnection_weight_ec,hyperconnection_weight_fc):
    # define lambdas for multiplying node weights by connection weight
    multiply_hyperconnection_weight_layer_IoTe = layers.Lambda((lambda x: x * hyperconnection_weight_IoTe), name = "connection_weight_IoTe")
    multiply_hyperconnection_weight_layer_IoTf = layers.Lambda((lambda x: x * hyperconnection_weight_IoTf), name = "connection_weight_IoTf")
    multiply_hyperconnection_weight_layer_ef = layers.Lambda((lambda x: x * hyperconnection_weight_ef), name = "connection_weight_ef")
    multiply_hyperconnection_weight_layer_ec = layers.Lambda((lambda x: x * hyperconnection_weight_ec), name = "connection_weight_ec")
    multiply_hyperconnection_weight_layer_fc = layers.Lambda((lambda x: x * hyperconnection_weight_fc), name = "connection_weight_fc")
    return multiply_hyperconnection_weight_layer_IoTe, multiply_hyperconnection_weight_layer_IoTf, multiply_hyperconnection_weight_layer_ef, multiply_hyperconnection_weight_layer_ec, multiply_hyperconnection_weight_layer_fc