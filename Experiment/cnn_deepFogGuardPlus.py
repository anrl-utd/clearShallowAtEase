
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings

import keras.backend as K
import keras.layers as layers
from keras.backend import zeros
from keras_applications.imagenet_utils import _obtain_input_shape, get_submodules_from_kwargs
from keras_applications import imagenet_utils
import keras 
from Experiment.cnn_deepFogGuard import define_cnn_deepFogGuard_architecture_IoT, define_cnn_deepFogGuard_architecture_cloud, define_cnn_deepFogGuard_architecture_edge, define_cnn_deepFogGuard_architecture_fog

# ResiliNet
def define_deepFogGuardPlus_CNN(input_shape=None,
                                alpha=1.0,
                                depth_multiplier=1,
                                include_top=True,
                                pooling=None,
                                classes=1000, 
                                strides = (2,2),
                                failout_survival_setting = [1.0,1.0],
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
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        survive_rates: survival rates of network nodes, default value is [1,1]
    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
   
    # Determine proper input shape and default size.
    img_input = layers.Input(shape=input_shape)  

    # nodewise dropout definitions
    edge_failure_lambda, fog_failure_lambda = cnn_nodewise_dropout_definitions(failout_survival_setting)

     # iot node
    iot_output,skip_iotfog = define_cnn_deepFogGuard_architecture_IoT(input_shape,alpha,img_input, strides = strides)
    
    # edge node
    edge_output, skip_edgecloud = define_cnn_deepFogGuard_architecture_edge(iot_output,alpha, depth_multiplier, strides = strides)
    edge_output = edge_failure_lambda(edge_output)
    skip_edgecloud = edge_failure_lambda(skip_edgecloud)
    
    # fog node
    fog_output = define_cnn_deepFogGuard_architecture_fog(skip_iotfog, edge_output, alpha, depth_multiplier)
    fog_output = fog_failure_lambda(fog_output)

    # cloud node
    cloud_output = define_cnn_deepFogGuard_architecture_cloud(fog_output, skip_edgecloud, alpha, depth_multiplier, classes, include_top, pooling)
    
    # Create model.
    model = keras.Model(img_input, cloud_output, name='ANRL_mobilenet')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_nodewise_dropout_definitions(failout_survival_setting):
    edge_survivability = failout_survival_setting[0]
    fog_survivability = failout_survival_setting[1]
    
    # variables for node-wise dropout
    edge_rand = K.variable(0)
    fog_rand = K.variable(0)
    edge_survivability_keras = K.variable(edge_survivability)
    fog_survivability_keras = K.variable(fog_survivability)
   # node-wise dropout occurs only during training
    K.set_learning_phase(1)
    if K.learning_phase():
        # seeds so the random_number is different for each node 
        edge_rand = K.random_uniform(shape=edge_rand.shape)
        fog_rand = K.random_uniform(shape=fog_rand.shape)
    # define lambda for failure, only fail during training
    edge_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(edge_rand,edge_survivability_keras), x * 0, x),name = 'edge_failure_lambda')
    fog_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(fog_rand,fog_survivability_keras), x * 0, x),name = 'fog_failure_lambda')
    return edge_failure_lambda, fog_failure_lambda