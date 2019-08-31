from Experiment.FailureIteration import numSurvivedComponents, calcWeight, normalizeWeights, calcAverageAccuracy, convertBinaryToList
from Experiment.utility import fail_node
def iterateFailuresExperimentFromImageDataGenerator(surv,numComponents,model,accuracyList,weightList,output_list, test_generator):
    """runs through all failure configurations for one model
    ### Arguments
        surv (list): contains the survival rate of all nodes, ordered from edge to fog node
        numComponents (int): number of nodes that can fail
        model (Model): Keras model
        accuracyList (list): list of all the survival configuration accuracies 
        weightList (list): list of all the survival configuration probabilites 
        output_list (list): list that contains string output of the experiment
        train_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distributio
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
        test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
    ### Returns
        return how many survival configurations had total network failure
    """  
    failure_count = 0
    maxNumComponentFailure = 2 ** numComponents
    for i in range(maxNumComponentFailure):
        numSurvived = numSurvivedComponents(i)
        if ( numSurvived >= numComponents - maxNumComponentFailure ):
            listOfZerosOnes = convertBinaryToList(i, numComponents)
            failures = [int(failure) for failure in listOfZerosOnes]
            # saves a copy of the original model so it does not change during failures 
            old_weights = model.get_weights()
            fail_node(model,failures)
            print(failures)
            output_list.append(str(failures))
            accuracy,failure = calcModelAccuracy(model,output_list,test_generator)
            # add number of failures for a model
            failure_count += failure
            # change the changed weights to the original weights
            model.set_weights(old_weights)
            # calculate weight of the result based on survival rates 
            weight = calcWeight(surv, listOfZerosOnes)
            accuracyList.append(accuracy)
            weightList.append(weight)
            print("numSurvived:",numSurvived," weight:", weight, " acc:",accuracy)
            output_list.append("numSurvived: " + str(numSurvived) + " weight: " + str(weight) + " acc: " + str(accuracy) + '\n')
    return failure_count

def calcModelAccuracy(model,output_list,test_generator):
    pass
def calculateExpectedAccuracyFromImageGenerator(model,surv,output_list,test_generator):
    """run full survival configuration failure
    ### Arguments
        model (Model): Keras model
        surv (list): contains the survival rate of all nodes, ordered from edge to fog node
        output_list (list): list that contains string output of the experiment
        training_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distributio
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
        test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
    ### Returns
        return weighted accuracy 
    """  
    numComponents = len(surv)
    accuracyList = []
    weightList = []
    failure_count = iterateFailuresExperimentFromImageDataGenerator(surv,numComponents, model,accuracyList,weightList,output_list,test_generator)
    weightList = normalizeWeights(weightList)
    avg_acc = calcAverageAccuracy(accuracyList, weightList)
    output_list.append('Number of Failures: ' + str(failure_count) + '\n')
    output_list.append('Average Accuracy: ' + str(avg_acc) + '\n')
    print('Number of Failures: ',str(failure_count))
    print("Average Accuracy:", avg_acc)
    return avg_acc
