
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


from Experiment.common_exp_methods import fail_node
from Experiment.random_guess import model_guess, cnnmodel_guess



def iterateAllFailureCombinationsCalcAccuracy(survivability_settings,numNodes,model,accuracyList,weightList,output_list,training_labels,test_data,test_labels):
    """runs through all node failure combinations and calculates the accuracy (and weight) of that particular node failure combination
    ### Arguments
        survivability_settings (list): List of the survival rate of all nodes, ordered from edge to fog node
        numNodes (int): number of physical nodes
        model (Model): Keras model
        accuracyList (list): list of all the accuracies (one per node failure combination)
        weightList (list): list of all the weights (one per node failure combination). Weight is the probability of that node failure combination
        output_list (list): list that contains string output of the experiment
        train_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distributio
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
        test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
    ### Returns
        return how many survival configurations had total network failure
    """  
    no_information_flow_count = 0
    maxNumNodeFailure = 2 ** numNodes
    for i in range(maxNumNodeFailure):
        numSurvivedNodes = numSurvivedComponents(i)
        node_failure_combination = convertBinaryToList(i, numNodes)
        
        # saves a copy of the original model so it does not change during failures 
        old_weights = model.get_weights()
        is_cnn = fail_node(model,node_failure_combination)
        print(node_failure_combination)
        output_list.append(str(node_failure_combination))
        accuracy,no_information_flow = calcModelAccuracy(model,output_list,training_labels,test_data,test_labels,is_cnn)
        # add number of no_information_flow for a model
        no_information_flow_count += no_information_flow
        # change the changed weights to the original weights
        model.set_weights(old_weights)
        # calculate weight of the result based on survival rates 
        for survivability_setting in survivability_settings:
            weight = calcWeight(survivability_setting, node_failure_combination)
        accuracyList.append(accuracy)
        weightList.append(weight)
        print("numSurvivedNodes:",numSurvivedNodes," weight:", weight, " acc:",accuracy)
        output_list.append("numSurvivedNodes: " + str(numSurvivedNodes) + " weight: " + str(weight) + " acc: " + str(accuracy) + '\n')
    return no_information_flow_count

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
        
def calcWeight(survivability_setting, node_failure_combination):
    """calculates the weight of each combination of component failures
    ### Arguments
        survivability_setting (list): list of probabilities
        node_failure_combination (list): list of the node survival outcomes
    ### Returns
        return probability of a particular survival outcome
    """  
    weight = 1
    for i in range(len(node_failure_combination)):
        if (node_failure_combination[i] == '1'): # if it survives
            weight = weight * survivability_setting[i]
        else: # if it fails
            weight = weight * (1 - survivability_setting[i])
    return weight
    

def numSurvivedComponents(number):
    """calculates the number of survived components (physical nodes) by counting ones in a bit string
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
    lst_integer = [int(num) for num in lst] # converts ["0","1","0"] to [0,1,0]
    return lst_integer
 
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
        return model accuracy and whether a no_information_flow occured 
    """  
    # accuracy based on whether the model is fully connected or not 
    if is_cnn:
         acc,no_information_flow = cnnmodel_guess(model,training_labels,test_data,test_labels)
    else:
        acc,no_information_flow = model_guess(model,training_labels,test_data,test_labels)
    return acc,no_information_flow


def normalize(weights):
    """Normalizes the elements of a list, so that they sum to 1
    ### Arguments
       weights(list): list of all the probability weights
    ### Returns
        return normalized lost of probability weights
    """  
    sumWeights = sum(weights)
    normalized = [(x/sumWeights) for x in weights]
    return normalized
 
def calculateExpectedAccuracy(model,survivability_setting,output_list,training_labels,test_data,test_labels):
    """Calculates the expected accuracy of the model under certain survivability setting
    ### Arguments
        model (Model): Keras model
        survivability_setting (list): List of the survival rate of all nodes
        output_list (list): list that contains string output of the experiment
        training_labels (numpy array): 1D array that corresponds to each row in the training data with a class label, used for calculating train class distributio
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
        test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
    ### Returns
        return weighted accuracy 
    """  
    numNodes = len(survivability_setting)
    accuracyList = []
    weightList = []
    no_information_flow_count = iterateAllFailureCombinationsCalcAccuracy(survivability_setting,numNodes, model,accuracyList,weightList,output_list,training_labels,test_data,test_labels)
    weightList = normalize(weightList)
    avg_acc = calcAverageAccuracy(accuracyList, weightList)
    output_list.append('Times we had no information flow: ' + str(no_information_flow_count) + '\n')
    output_list.append('Average Accuracy: ' + str(avg_acc) + '\n')
    print('Times we had no information flow: ',str(no_information_flow_count))
    print("Average Accuracy:", avg_acc)
    return avg_acc

