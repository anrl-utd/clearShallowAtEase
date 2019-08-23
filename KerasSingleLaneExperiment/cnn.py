from KerasSingleLaneExperiment.cnn_utils import _conv_block,_depthwise_conv_block
import keras.backend as backend
import keras.layers as layers

def define_cnn_architecture_IoT(img_input,alpha, strides = (2,2)):
    return  _conv_block(img_input, 32, alpha, strides=strides)

def define_cnn_architecture_edge(iot_output,alpha, depth_multiplier, strides =(2,2)):
    edge_input = _depthwise_conv_block(iot_output, 64, alpha, depth_multiplier, block_id=1)
    edge = _depthwise_conv_block(edge_input, 128, alpha, depth_multiplier,
                              strides=strides, block_id=2)
    edge_output = _depthwise_conv_block(edge, 128, alpha, depth_multiplier, block_id=3)
    return edge_output

def define_cnn_architecture_fog(edge_output,alpha, depth_multiplier):
    fog_input = _depthwise_conv_block(edge_output, 256, alpha, depth_multiplier,
                          strides=(2, 2), block_id=4)
    fog = _depthwise_conv_block(fog_input, 256, alpha, depth_multiplier, block_id=5)
    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier, block_id=7)
    fog_output = _depthwise_conv_block(fog, 512, alpha, depth_multiplier, block_id=8)
    return fog_output

def define_cnn_architecture_cloud(fog_output,alpha,depth_multiplier, classes,include_top,pooling):
    cloud_input = _depthwise_conv_block(fog_output, 512, alpha, depth_multiplier, block_id=9)
    cloud = _depthwise_conv_block(cloud_input, 512, alpha, depth_multiplier, block_id=10)
    cloud = _depthwise_conv_block(cloud, 512, alpha, depth_multiplier, block_id=11)

    cloud = _depthwise_conv_block(cloud, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    cloud = _depthwise_conv_block(cloud, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        if backend.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        cloud = layers.GlobalAveragePooling2D()(cloud)
        cloud = layers.Reshape(shape, name='reshape_1')(cloud)
        cloud = layers.Conv2D(classes, (1, 1),
                          padding='same',
                          name='conv_preds')(cloud)
        cloud = layers.Reshape((classes,), name='reshape_2')(cloud)
        cloud_output = layers.Activation('softmax', name='act_softmax')(cloud)
    else:
        if pooling == 'avg':
            cloud_output = layers.GlobalAveragePooling2D()(cloud)
        elif pooling == 'max':
            cloud_output = layers.GlobalMaxPooling2D()(cloud)
    return cloud_output