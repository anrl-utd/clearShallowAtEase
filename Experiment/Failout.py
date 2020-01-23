from keras.layers import Layer, Add
from keras.layers import Lambda
import keras.backend as K
class Failout(Layer):
    """Applies Failout to the input.
    # Arguments
        reliability: float between 0 and 1. Probability of node failure.
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
    Input Multiplexer for a node that receives input from more than one downstream nodes
    # Arguments
        node_has_failed: Boolean Tensor showing if a node has failer
    """
    def __init__(self, has_failed, **kwargs):
        super(InputMux, self).__init__(**kwargs)
        self.has_failed = has_failed
        # self.name = name

    def _merge_function(self, inputs):
        output = K.switch(self.has_failed, inputs[0],inputs[1])
        # mux = Lambda(lambda x : x * 1,name = self.name)(input)
        return output