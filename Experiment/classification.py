
from collections import Counter
import random 
from keras.models import Model
import keras.backend as K
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def predict(model,train_labels,test_data,test_labels, is_cnn):
    """Performs prediction for test data, based on th learned parameters. (Performs random guess if there is no information flow in the DNN)
    ### Arguments
        model (Model): Keras model
        train_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distribution
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
        test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
        is_cnn (boolean): indicates if the predict is going to be used for CNN or MLP
    ### Returns
        return a tuple of accuracy (as a float) and whether there is no information flow (as an integer)
    """
    # check if the cloud input is 0 which means that there is no data flowing in the network
    cloud_input = model.get_layer(name = "Cloud_Input").output
    # get the output from the layer
    new_model = Model(inputs = model.inputs,outputs=cloud_input)
    cloud_output = new_model.predict(test_data)
    no_information_flow = np.array_equal(cloud_output,cloud_output * 0)
    # there is no connection flow, make random guess 

    if no_information_flow:
        if is_cnn:
            # make into 1d vector
            train_labels = [item for sublist in train_labels for item in sublist]
        else: # it is MLP
            # check if there are 6 images in the first dimension (used for Camera)
            if len(test_data) == 6:
                # reformat by switching the 1st and 2nd dimension
                test_data = np.transpose(test_data,axes=[1,0,2,3,4])
        print("There is no data flow in the network")
        preds = random_guess(train_labels,test_data)
        no_information_flow_count = 1
    else:
        preds = model.predict(test_data)
        preds = np.argmax(preds,axis=1)
        no_information_flow_count = 0

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
    print("class",class_frequency)
    # sort by keys and get the values
    # sorted_class_frequency = list(dict(sorted(class_frequency.items())).values())
    total_frequency = len(train_labels)
    # find relative frequency = class_frequency / total_frequency
    relative_class_frequency = [freq / total_frequency for freq in class_frequency]
    # append a 0 to the beginning of a new list
    cumulative_frequency = [0] + relative_class_frequency
    print("cumulative_frequency",cumulative_frequency)
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



        

