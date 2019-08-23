from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings

import keras.backend as K
import keras
import keras.layers as layers
from KerasSingleLaneExperiment.cnn_utils import _conv_block,_depthwise_conv_block
from KerasSingleLaneExperiment.cnn import define_cnn_architecture_IoT,define_cnn_architecture_cloud,define_cnn_architecture_edge,define_cnn_architecture_fog
BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')
def define_Vanilla_CNN(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              include_top=True,
              input_tensor=None,
              pooling=None,
              classes=1000,
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
    iot = define_cnn_architecture_IoT(img_input,alpha)
    # edge 
    edge = define_cnn_architecture_edge(iot,alpha,depth_multiplier)
    # fog node
    fog = define_cnn_architecture_fog(edge,alpha,depth_multiplier)
    # layer alias to name cloud input (alias is used for random guessing)
    # don't need between edge and IoT because 0 will propagate to this node
    fog = layers.Lambda(lambda x : x * 1,name = 'connection_cloud')(fog)
    # cloud node
    cloud = define_cnn_architecture_cloud(fog,alpha,depth_multiplier,classes,include_top,pooling)

    # Create model.
    model = keras.Model(img_input, cloud, name='ANRL_mobilenet')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
