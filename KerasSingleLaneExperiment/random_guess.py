
from collections import Counter
import random 
from keras.models import Model
import keras.backend as K
from sklearn.metrics import accuracy_score
import numpy as np

def model_guess(model,train_labels,test_data,test_labels,file_name = None):
    """Returns a guess of the data based on training class distribution if there is no data connection in the network
    ### Arguments
        model (Model): Keras model
        train_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distribution
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
        test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
        file_name (string): specifies the file name for output, not used anymore
    ### Returns
        return a tuple of accuracy as a float and whether there was total network failure as an integer
    """
    preds = model.predict(test_data)
    preds = np.argmax(preds,axis=1)
    # check if the connection is 0 which means that there is no data flowing in the network
    f3 = model.get_layer(name = "Cloud_Input").output
    # get the output from the layer
    output_model_f3 = Model(inputs = model.input,outputs=f3)
    f3_output = output_model_f3.predict(test_data)
    no_connection_flow_f3 = np.array_equal(f3_output,f3_output * 0)
    # there is no connection flow, make random guess 
    # variable that keeps track if the network has failed
    failure = 0
    if no_connection_flow_f3:
        print("There is no data flow in the network")
        preds = random_guess(train_labels,test_data)
        failure = 1
    acc = accuracy_score(test_labels,preds)
    return acc,failure

def cnnmodel_guess(model,train_labels,test_data,test_labels,file_name = None):
    """Returns a guess of the data based on training class distribution if there is no data connection in the CNN network
    ### Arguments
        model (Model): Keras model
        train_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distribution
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
        test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
        file_name (string): specifies the file name for output, not used anymore
    ### Returns
        return a tuple of accuracy as a float and whether there was total network failure as an integer
    """
    preds = model.predict(test_data)
    preds = np.argmax(preds,axis=1)
    # check if the connection is 0 which means that there is no data flowing in the network
    f1 = model.get_layer(name = "connection_cloud").output
    # get the output from the layer
    output_model_f1 = Model(inputs = model.input,outputs=f1)
    f1_output = output_model_f1.predict(test_data)
    no_connection_flow_f1 = np.array_equal(f1_output,f1_output * 0)
    # there is no connection flow, make random guess 
    # variable that keeps track if the network has failed
    failure = 0
    train_labels = [item for sublist in train_labels for item in sublist]
    if no_connection_flow_f1:
        print("There is no data flow in the network")
        preds = random_guess(train_labels,test_data)
        failure = 1
    acc = accuracy_score(test_labels,preds)
    return acc,failure


def random_guess(train_labels,test_data):
    """function returns a array of predictions from random guessing based on training class distribution 
    ### Arguments
        train_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distribution
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
    ### Returns
        return a 1-D array of predictions, shape is the number of test examples
    """
    # count the frequency of each class
    if "list" in str(type(train_labels)):
        class_frequency = Counter(train_labels)
    # numpy array 
    else:
        class_frequency = Counter(train_labels.flatten())
    # sort by keys and get the values
    sorted_class_frequency = list(dict(sorted(class_frequency.items())).values())
    total_frequency = len(train_labels)
    # find relative frequency 
    sorted_class_frequency = [freq / total_frequency for freq in sorted_class_frequency]
    # append a 0 to the beginning of a new list
    cumulative_frequency = [0] + sorted_class_frequency
    # calculate cumulative relative frequency 
    for index in range(1,len(cumulative_frequency)):
        cumulative_frequency[index] += cumulative_frequency[index-1]
    # make a guess for each test example
    guess_preds = [guess(cumulative_frequency) for example in test_data]
    return guess_preds

def guess(cumulative_frequency):    
    """makes a random number and determines a class based on the cumulative frequency
    ### Arguments
        cumulative_frequency (list): list of the cumulative frequencies for the class distribution, first value is always 0
    ### Returns
        return an int output
    """
    rand_num = random.random()
    for index in range(1,len(cumulative_frequency)):
        if rand_num <= cumulative_frequency[index] and rand_num >= cumulative_frequency[index-1]:
            return index
    return 0



        
