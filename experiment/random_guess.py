
from collections import Counter
import random 
from keras.models import Model
import keras.backend as K
from sklearn.metrics import accuracy_score
import numpy as np

# use a model with trained weights to guess if there are no connections 
def model_guess(model,train_labels,test_data,test_labels,file_name = None):
    preds = model.predict(test_data)
    preds = np.argmax(preds,axis=1)
    # check if the connection is 0 which means that there is no data flowing in the network
    f1 = model.get_layer(name = "F1_F2").output
    f2 = model.get_layer(name = "F1F2_F3").output
    f3 = model.get_layer(name = "F2F3_FC").output
    # get the output from the layer
    output_model_f1 = Model(inputs = model.input,outputs=f1)
    output_model_f2 = Model(inputs = model.input,outputs=f2)
    output_model_f3 = Model(inputs = model.input,outputs=f3)
    f1_output = output_model_f1.predict(test_data)
    f2_output = output_model_f2.predict(test_data)
    f3_output = output_model_f3.predict(test_data)
    no_connection_flow_f1 = np.array_equal(f1_output,f1_output * 0)
    no_connection_flow_f2 = np.array_equal(f2_output,f2_output * 0)
    no_connection_flow_f3 = np.array_equal(f3_output,f3_output * 0)
    # there is no connection flow, make random guess 
    # variable that keeps track if the network has failed
    failure = 0
    if no_connection_flow_f1 or no_connection_flow_f2 or no_connection_flow_f3:
        print("There is no data flow in the network")
        preds = random_guess(train_labels,test_data)
        # if file_name != None:
        #     with open(file_name,'a+') as file:
        #         file.write('There is no data flow in the network' + '\n')
        failure = 1
    acc = accuracy_score(test_labels,preds)
    return acc,failure

# function returns a array of predictions based on random guessing
# random guessing is determined by the class distribution from the training data. 
# input: list of training labels
# input: matrix of test_data, rows are examples and columns are variables 
def random_guess(train_labels,test_data):
    # count the frequency of each class
    class_frequency = Counter(train_labels)
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
# makes a random number and determines a class based on the cumulative frequency
def guess(cumulative_frequency):
    # set the seed for more deterministc outputs 
    random.seed(11)
    rand_num = random.random()
    for index in range(1,len(cumulative_frequency)):
        if rand_num <= cumulative_frequency[index] and rand_num >= cumulative_frequency[index-1]:
            return index
    return 0



        
