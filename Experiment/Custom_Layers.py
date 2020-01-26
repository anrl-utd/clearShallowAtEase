from keras.layers import Layer, Add, Lambda
import keras.layers as layers
import keras.backend as K

class Failout(Layer):
    """Applies Failout to the output of a node.
    # Arguments
        reliability: float between 0 and 1. Probability of survival of a node (1 - prob_failure).
        seed: A Python integer to use as random seed.
    """
    def __init__(self, reliability, seed=None, **kwargs):
        super(Failout, self).__init__(**kwargs)
        self.seed = seed
        self.reliability = K.variable(reliability)
        self.has_failed = None

    def call(self, inputs, training=None):
        rand = K.random_uniform(K.variable(0).shape, seed = self.seed)
        # assumes that there is only one input in inputs
        fail = Lambda(lambda x: x * 0)
        self.has_failed = K.greater(rand, self.reliability)
        failed_inputs = K.switch(self.has_failed,fail(inputs),inputs)
        failout = K.in_train_phase(failed_inputs, inputs, training)
        return failout

    def get_config(self):
        config = {
                  'seed': self.seed,
                  'reliability': self.reliability,
                  'has_failed': self.has_failed
                }
        base_config = super(Failout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class InputMux(Add):
    """
    Input Multiplexer for a node that receives input from more than one downstream nodes, 
    # Arguments
        node_has_failed: Boolean Tensor showing if the downstream node has failed
    """
    def __init__(self, has_failed, **kwargs):
        super().__init__(has_failed, **kwargs)
        self.has_failed = has_failed
        # self.name = name

    def _merge_function(self, inputs):
        """
        # inputs
        the two incoming connections to a node. inputs[0] MUST be the input from skip hyperconnection
        and inputs[1] MUST be the input from the node below.
        """
        
        selected = K.switch(self.has_failed, inputs[0], inputs[1]) # selects one of the inputs. 
        # If the node below has failed, use the input from skip hyperconnection, otherwise, use the input from the node below
        
        added = layers.add(inputs) # calls the add function

        output = K.in_train_phase(added, selected)
        return output


class InputMuxMobileNet(InputMux):
    """
    Input Multiplexer MobileNet CNN
    """
    def __init__(self, has_failed, **kwargs):
        super(InputMuxMobileNet, self).__init__(**kwargs)

    def _merge_function(self, inputs):
        # 1x1 conv2d is used to change the filter size 
        # ? (alpha=0.5), ? (alpha=0.75)
        skip_iotfog = layers.Conv2D(96,(1,1),strides = 4, use_bias = False, name = "random_name")(inputs[0])

        edge_output = layers.Conv2D(64,(1,1),strides = 4, use_bias = False, name = "random_name2")(inputs[1])

        selected = K.switch(self.has_failed, skip_iotfog, edge_output) # selects one of the inputs. 
        # If the node below has failed, use the input from skip hyperconnection, otherwise, use the input from the node below
        
        added = layers.add(inputs) # calls the add function

        output = K.in_train_phase(added, selected)
        return output


