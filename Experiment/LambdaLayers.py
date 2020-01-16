from keras.layers import add
import keras.backend as K
from keras.layers import Lambda

def add_node_layers(input_tensors):
    """lambda function to add physical nodes in the network 
    ### Arguments
        input_tensors (list): list of tensors
    ### Returns
        returns the sum of the output layers
    """  
    output = []
    if(len(input_tensors) == 1):
            return input_tensors
    else:
        for tensors in input_tensors:
            output.append(tensors)
        return add(output)

def switch(inputs):
    lambdaswitch = Lambda(lambda x: K.switch(x[0],x[1],x[2]))
    return lambdaswitch([inputs[0], inputs[1], inputs[2]])