import numpy as np
import pandas as pd
import sys
from collections import Counter
# from imblearn.over_sampling import SMOTE


def load_data(path):
    """reads in text file from a path and returns data and the labels as numpy arrays
    ### Arguments
        path (string): path where file is located to read 
    ### Returns
        return [data (numpy), labels (numpy)]
    """  
    file = open(path,'r')
    data = []
    for line in file:
        data.append(line.split())
    file.close()
    labels = []
    # take away the headers
    data = data[1:len(data)] 
    # convert from string to float
    data = [[float(dataPoint) for dataPoint in row] for row in data]
     # extract the classes and remove the classes column
    for index in range(len(data)):
        labels.append(data[index][-1])
        del data[index][-1]
    #print(np.asarray(data))
    return np.asarray(data),np.asarray(labels)

def combine_data(path):
    """combines all the patients into one complete dataset 
    ### Arguments
        path (string): file path of all the individual logs
    ### Returns
        save complete data log locally
    """  
    training_data,training_labels = load_data(path + '1' + '.log')
    # load each patient data
    for subject in range(2,11):
        file_path = path + str(subject) + '.log'
        data,labels = load_data(file_path)
        training_data = np.concatenate((training_data,data),axis=0)
        training_labels =  np.concatenate((training_labels,labels),axis=0)
        print("after concatenating",training_data.shape)
    training_data = np.column_stack((training_data,training_labels))
    np.savetxt("mHealth_complete.log", training_data, fmt='%d')

def deleteZeros(path):
     """deletes all examples with 0 as class label
     ### Arguments
        path (string): file path of a log
     ### Returns
        save changed log locally
     """  
     f = pd.read_table(path, header=None, delim_whitespace=True)
     f= f[f[23] != 0]
     np.savetxt(path, f.values, fmt='%d')
     print(f[23].value_counts().sort_index(ascending=[True]).tolist())
     return f

