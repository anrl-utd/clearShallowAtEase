from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings

import keras.backend as K
import keras.layers as layers
from Experiment.MobileNet_blocks import _conv_block, _depthwise_conv_block
from Experiment.common_exp_methods import compile_keras_parallel_model

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')
def define_vanilla_model_CNN(input_shape=None,
                            alpha=1.0,
                            depth_multiplier=1,
                            include_top=True,
                            input_tensor=None,
                            pooling=None,
                            classes=1000,
                            strides = (2,2),
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
    # Determine proper input shape and default size.
    img_input = layers.Input(shape=input_shape)  
    # changed the strides from 2 to 1 since cifar-10 images are smaller
    # IoT Node
    iot = define_cnn_architecture_IoT(img_input,alpha, strides = strides)
    # edge 
    edge = define_cnn_architecture_edge(iot,alpha,depth_multiplier, strides = strides)
    
    # fog node
    fog = layers.Lambda(lambda x : x * 1,name = 'node2_input')(edge)
    fog = define_cnn_architecture_fog(fog,alpha,depth_multiplier)
    
    # cloud node
    cloud = layers.Lambda(lambda x : x * 1,name = 'node1_input')(fog)
    cloud = define_cnn_architecture_cloud(cloud,alpha,depth_multiplier,classes,include_top,pooling)

    model, parallel_model = compile_keras_parallel_model(img_input, cloud, num_gpus)
    return model, parallel_model

def define_cnn_architecture_IoT(img_input,alpha, strides = (2,2)):
    return  _conv_block(img_input, 32, alpha, strides=strides)

def define_cnn_architecture_edge(iot_output,alpha, depth_multiplier, strides =(2,2)):
    edge = _depthwise_conv_block(iot_output, 64, alpha, depth_multiplier, block_id=1)
    edge = _depthwise_conv_block(edge, 128, alpha, depth_multiplier,
                              strides=strides, block_id=2)
    edge_output = _depthwise_conv_block(edge, 128, alpha, depth_multiplier, block_id=3)
    return edge_output

def define_cnn_architecture_fog(edge_output,alpha, depth_multiplier):
    fog = _depthwise_conv_block(edge_output, 256, alpha, depth_multiplier,
                          strides=(2, 2), block_id=4)
    fog = _depthwise_conv_block(fog, 256, alpha, depth_multiplier, block_id=5)
    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier, block_id=7)
    fog_output = _depthwise_conv_block(fog, 512, alpha, depth_multiplier, block_id=8)
    return fog_output

def define_cnn_architecture_cloud(fog_output,alpha,depth_multiplier, classes,include_top,pooling):
    cloud = _depthwise_conv_block(fog_output, 512, alpha, depth_multiplier, block_id=9)
    cloud = _depthwise_conv_block(cloud, 512, alpha, depth_multiplier, block_id=10)
    cloud = _depthwise_conv_block(cloud, 512, alpha, depth_multiplier, block_id=11)

    cloud = _depthwise_conv_block(cloud, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    cloud = _depthwise_conv_block(cloud, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        if K.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        cloud = layers.GlobalAveragePooling2D()(cloud)
        cloud = layers.Reshape(shape, name='reshape_1')(cloud)
        cloud = layers.Dropout(1e-3, name='dropout')(cloud)
        cloud = layers.Conv2D(classes, (1, 1),
                          padding='same',
                          name='conv_preds')(cloud)
        cloud = layers.Reshape((classes,), name='reshape_2')(cloud)
        cloud_output = layers.Activation('softmax', name='output')(cloud)
    else:
        if pooling == 'avg':
            cloud_output = layers.GlobalAveragePooling2D()(cloud)
        elif pooling == 'max':
            cloud_output = layers.GlobalMaxPooling2D()(cloud)
    return cloud_output
