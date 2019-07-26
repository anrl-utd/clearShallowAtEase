from keras.layers import add

def add_node_layers(input_tensors):
    """lambda function to add physical nodes in the network 
    ### Arguments
        input_tensors (list): list of tensors
    ### Returns
        returns the sum of the output layers
    """  
    output = []
    for tensors in input_tensors:
       output.append(tensors)
    return add(output)
