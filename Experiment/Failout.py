from keras.layers import Layer
from keras.layers import Lambda
import keras.backend as K
class Failout(Layer):
    """Applies Failout to the input.
    Failout consists in randomly dropping out entire output of a layer by setting output to 0 at each update during training time,
    which helps increase resilencey of a distributed neural network 
    # Arguments
        reliability: float between 0 and 1. Probability of node failure.
        seed: A Python integer to use as random seed.
    """
    def __init__(self, reliability, seed=None, **kwargs):
        super(Failout, self).__init__(**kwargs)
        self.seed = seed
        self.reliability = K.variable(reliability)

    def call(self, inputs, training=None):
        rand = K.random_uniform(K.variable(0).shape, seed = self.seed)
        # assumes that there is only one input in inputs
        fail = Lambda(lambda x: x * 0)
        condition = K.switch(K.greater(rand, self.reliability),fail(inputs),inputs)
        failout = K.in_train_phase(condition, inputs, training)
        return failout

    def get_config(self):
        config = {
                  'seed': self.seed,
                  'reliability': self.reliability
                }
        base_config = super(Failout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
