
from collections import Counter
import random 
from keras.models import Model
import keras.backend as K
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def predict(model,no_information_flow,train_labels,test_data,test_labels, experiment_name):
    """Performs prediction for test data, based on th learned parameters. (Performs random guess if there is no information flow in the DNN)
    ### Arguments
        model (Model): Keras model
        train_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distribution
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
        test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
    ### Returns
        return a tuple of accuracy (as a float) and whether there is no information flow (as an integer)
    """
    
    if no_information_flow is True:
        if experiment_name == "CIFAR" :
            # make into 1d vector
            train_labels = [item for sublist in train_labels for item in sublist]
        elif experiment_name == "Camera" :
            # reformat by switching the 1st and 2nd dimension
            test_data = np.transpose(test_data,axes=[1,0,2,3,4])
        # print("There is no data flow in the network")
        preds = random_guess(train_labels,test_data)
        no_information_flow_count = 1
    else:
        preds = model.predict(test_data)
        preds = np.argmax(preds,axis=1)
        no_information_flow_count = 0

    # camera experiments should report precision 
    if experiment_name == 'Camera':
        acc = f1_score(test_labels, preds, average='micro')
    else:
        acc = accuracy_score(test_labels,preds)
    return acc,no_information_flow_count

def random_guess(train_labels,test_data):
    """function returns a array of predictions from random guessing based on training class distribution 
    ### Arguments
        train_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distribution
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
    ### Returns
        return a 1-D array of predictions, shape is the number of test examples
    """
    # count the frequency of each class
    if "list" in str(type(train_labels)): # list
        class_frequency = Counter(train_labels)
    else: # numpy array 
        class_frequency = Counter(train_labels.flatten())
    # sort the class frequency by keys, so it can be used in toss coin function. (because the intervals in toss_coin are inherently sorted)
    class_frequency_sorted_by_keys = list(dict(sorted(class_frequency.items())).values())
    total_frequency = len(train_labels)
    # find relative frequency = class_frequency / total_frequency
    relative_class_frequency = [freq / total_frequency for freq in class_frequency_sorted_by_keys]
    # append a 0 to the beginning of a new list
    cumulative_frequency = [0] + relative_class_frequency
    # calculate cumulative relative frequency 
    for index in range(1,len(cumulative_frequency)):
        cumulative_frequency[index] += cumulative_frequency[index-1]
    # make a guess for each test example
    # check if the test data are images
    guess_preds = [toss_coin(cumulative_frequency) for example in test_data]
    return guess_preds

def toss_coin(cumulative_frequency):    
    """tosses a coin and determines a class based on the cumulative frequency
    ### Arguments
        cumulative_frequency (list): list of the cumulative frequencies for the class distribution, first value is always 0
    ### Returns
        return an int output
    """
    rand_num = random.random()
    for index in range(1,len(cumulative_frequency)):
        if rand_num <= cumulative_frequency[index] and rand_num >= cumulative_frequency[index-1]:
            return index - 1
    return 0

def identify_no_information_flow(model,test_data,exp):
    node_input = {} # define the input to all nodes as dictionary
    new_model = {}
    new_model_val = {}
    horizontal_DNN = True
    if exp == "CIFAR/Imagenet":
        num_nodes = 2
    if exp == "Health":
        num_nodes = 3
    if exp == "Camera":
        num_nodes = 5
        horizontal_DNN = False
    for i in range(1, num_nodes+1):
        name = "node"+str(i)+"_input"
        node_input[i] = model.get_layer(name = name).output
        # get the output from the layer
        new_model[i] = Model(inputs = model.input,outputs=node_input[i])
        new_model_val[i] = new_model[i].predict(test_data)
        input_of_node_is_zero = np.array_equal(new_model_val[i],new_model_val[i] * 0)
        if horizontal_DNN:
            if input_of_node_is_zero is True:
                return True
        else:
            if name == "node1_input" or name == "node2_input" or name == "node3_input":
                if input_of_node_is_zero is True: # if input of any of the nodes 1,2,3 is zero, we have no_information_flow
                    return True
            else: # if name is node4_input or node5_input
                if name == "node4_input":
                    if input_of_node_is_zero is True:
                        continue
                    else: # since we are processing the nodes in order, when we get here, it means we for nodes 1,2,3,4 (input_of_node_is_zero is False)
                        return False
                if name == "node5_input": # since we are processing the nodes in order, when we get here, it means we for nodes 1,2,3 (input_of_node_is_zero is False) and for node 4 (input_of_node_is_zero is True)
                    if input_of_node_is_zero is True:
                        return True

    return False # Otherwise, return False