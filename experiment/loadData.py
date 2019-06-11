import numpy as np
import pandas as pd
import sys
from collections import Counter
# from imblearn.over_sampling import SMOTE

# reads in text file from a path and returns data and the labels as numpy arrays
def load_data(path):
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

# divides dataset into training, validation, and test set
# training: subject 1-8
# validation: subject 9
# test: subject 10
def divide_data():
    path = 'MHEALTHDATASET/mHealth_subject'
    training_data,training_labels = load_data(path + '1' + '.log')
    print("initial",training_data.shape)
    # make training data
    for subject in range(2,9):
        file_path = path + str(subject) + '.log'
        data,labels = load_data(file_path)
        training_data = np.concatenate((training_data,data),axis=0)
        training_labels =  np.concatenate((training_labels,labels),axis=0)
        print("after concatenating",training_data.shape)

    # make validation data
    validation_data,validation_labels = load_data(path + '9' + '.log')
    # make test data
    test_data, test_labels = load_data(path + '10' + '.log')

    print(training_data.shape)
    print(training_labels.shape)
    # add data with labels
    training_data = np.column_stack((training_data,training_labels))
    validation_data = np.column_stack((validation_data,validation_labels))
    test_data = np.column_stack((test_data,test_labels))

    np.savetxt("mHealth_train.log", training_data, fmt='%d')
    # np.savetxt("mHealth_validation.log", validation_data, fmt='%d')
    # np.savetxt("mHealth_test.log", test_data, fmt='%d')

# combines all the patients into one complete dataset   
def combine_data():
    path = 'MHEALTHDATASET/mHealth_subject'
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
# deletes all examples with 0 as label
def deleteZeros(path):
     f = pd.read_table(path, header=None, delim_whitespace=True)
     f= f[f[23] != 0]
     np.savetxt(path, f.values, fmt='%d')
     #dataframe.sort_index(by='count', ascending=[True])
     print(f[23].value_counts().sort_index(ascending=[True]).tolist())
     return f

def replaceZeros(path):
    df = pd.read_table(path, header=None, delim_whitespace=True)
    df.replace(0,0.1,inplace=True)
    #print(df.values)
    np.savetxt(path, df.values,fmt='%g')

# loops through all data files and deletes examples with 0 as label
def deleteZerosForAllFiles():
     for i in range(1,11):
        print("Subject " + str(i))
        deleteZeros('MHEALTHDATASET/mHealth_subject' + str(i) + '.log')
# removes examples that are duplicated
def removeDuplicates():
    # remove duplicates in training file
    training_data, training_labels = load_data("mHealth_train.log")
    training_data = np.column_stack((training_data,training_labels))
    # remove non-unique rows
    train_uniques = np.vstack({tuple(row) for row in training_data})
    # remove duplicates in validation file
    validation_data, validation_labels = load_data("mHealth_validation.log")
    validation_data = np.column_stack((validation_data,validation_labels))
    validation_uniques = np.vstack({tuple(row) for row in validation_data})
    # remove duplicates in test file
    test_data, test_labels = load_data("mHealth_test.log")
    test_data = np.column_stack((test_data,test_labels))
    test_uniques = np.vstack({tuple(row) for row in validation_data})
    np.savetxt("mHealth_uniques_train.log", train_uniques, fmt='%d')
    np.savetxt("mHealth_uniques_validation.log", validation_uniques, fmt='%d')
    np.savetxt("mHealth_uniques_test.log", test_uniques, fmt='%d')

# # oversamples minority cases to achieve a balanced dataset
# def perform_SMOTE():
#     # read files
#     training_data, training_labels = load_data("mHealth_uniques_train.log")
#     validation_data, validation_labels = load_data("mHealth_uniques_validation.log")
#     test_data, test_labels = load_data("mHealth_uniques_test.log")
#     # perform SMOTE on data
#     training_data, training_labels = SMOTE().fit_resample(training_data,training_labels)
#     validation_data, validation_labels = SMOTE().fit_resample(validation_data,validation_labels)
#     test_data, test_labels = SMOTE().fit_resample(test_data, test_labels)

#     # combine data and labels
#     training_data = np.column_stack((training_data,training_labels))
#     validation_data = np.column_stack((validation_data,validation_labels))
#     test_data = np.column_stack((test_data,test_labels))

#     # save data
#     # np.savetxt("mHealth_SMOTE_uniques_train.log", training_data, fmt='%d')
#     # np.savetxt("mHealth_SMOTE_uniques_validation.log", validation_data, fmt='%d')
#     # np.savetxt("mHealth_SMOTE_uniques_test.log", test_data, fmt='%d')
if __name__ == "__main__": 
    #deleteZerosForAllFiles()
    #removeDuplicates()
    #perform_SMOTE()
    #divide_data()
    #replaceZeros('mHealth_test.log')
    combine_data()
    