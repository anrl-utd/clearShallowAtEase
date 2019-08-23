
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


from KerasSingleLaneExperiment.main import fail_node,test
from KerasSingleLaneExperiment.random_guess import model_guess, cnnmodel_guess


def iterateFailuresExperiment(surv,numComponents,model,accuracyList,weightList,output_list,training_labels,test_data,test_labels):
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
            is_cnn = fail_node(model,failures)
            print(failures)
            output_list.append(str(failures))
            accuracy,failure = calcModelAccuracy(model,output_list,training_labels,test_data,test_labels,is_cnn)
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
                
def calcAverageAccuracy(accuracyList, weightList):
    """calculates weighted accuracy based on failure probabilities 
    ### Arguments
        accuracyList (list): list of all the survival configuration accuracies 
        weightList (list): list of all the survival configuration probabilities
    ### Returns
        return weighted average accuracy 
    """  
    averageAccuracy = 0
    for i in range(len(accuracyList)):
        averageAccuracy += accuracyList[i] * weightList[i]
    return averageAccuracy
        
def calcWeight(survivability, listOfZerosOnes):
    """calculates the weight of each combination of component failures
    ### Arguments
        survivability (list): list of probabilities
        listOfZerosOnes (list): list of the node survival outcomes
    ### Returns
        return probability of a particular survival outcome
    """  
    weight = 1
    for i in range(len(listOfZerosOnes)):
        if (listOfZerosOnes[i] == '1'): # if it survives
            weight = weight * survivability[i]
        else: # if it fails
            weight = weight * (1 - survivability[i])
    return weight
    

# 
def numSurvivedComponents(number):
    """calculates the number of survived components by counting ones in a bit string
    ### Arguments
        number (int): number to be converted to binary
    ### Returns
        return number of survived nodes
    """  
    # convert given number into binary
    # output will be like bin(11)=0b1101
    binary = bin(number)
    # now separate out all 1's from binary string
    # we need to skip starting two characters
    # of binary string i.e; 0b
    setBits = [ones for ones in binary[2:] if ones=='1']
    return len(setBits)

def convertBinaryToList(number, numBits):
    """converts a number (e.g. 128) to its binary representation in a list. It converts number 128 to ['1', '0', '0', '0', '0', '0', '0', '0']    
    ### Arguments
        number (int): number to be converted to binary
        numBits (int): number of maximum bits 
    ### Returns
        return binary number to a list representing the binary number
    """  
    # convert given number into binary
    # output will be like bin(11)=0b1101
    binary = bin(number)
    lst = [bits for bits in binary[2:]]
    # pad '0's to the begging of the list, if its length is not 'numBits'
    for padding in range(max(0,numBits - len(lst))):
        lst.insert(0,'0')
    return lst
 
def calcModelAccuracy(model,output_list,training_labels,test_data,test_labels, is_cnn):
    """Calculates model accuracy based on node failure  
    ### Arguments
        model (Model): Keras model
        output_list (list): list that contains string output of the experiment
        training_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distributio
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
        test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
        is_cnn (boolean): used to determine which guess function to use
    ### Returns
        return model accuracy and whether a failure occured 
    """  
    # accuracy based on whether the model is fully connected or not 
    if is_cnn:
         acc,failure = cnnmodel_guess(model,training_labels,test_data,test_labels)
    else:
        acc,failure = model_guess(model,training_labels,test_data,test_labels)
    return acc,failure


def normalizeWeights(weights):
    """Calculates model accuracy based on node failure  
    ### Arguments
       weights(list): list of all the probability weights
    ### Returns
        return normalized lost of probability weights
    """  
    sumWeights = sum(weights)
    weightNormalized = [(x/sumWeights) for x in weights]
    return weightNormalized
 
def calculateExpectedAccuracy(model,surv,output_list,training_labels,test_data,test_labels):
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
    failure_count = iterateFailuresExperiment(surv,numComponents, model,accuracyList,weightList,output_list,training_labels,test_data,test_labels)
    weightList = normalizeWeights(weightList)
    avg_acc = calcAverageAccuracy(accuracyList, weightList)
    output_list.append('Number of Failures: ' + str(failure_count) + '\n')
    output_list.append('Average Accuracy: ' + str(avg_acc) + '\n')
    print('Number of Failures: ',str(failure_count))
    print("Average Accuracy:", avg_acc)
    return avg_acc
