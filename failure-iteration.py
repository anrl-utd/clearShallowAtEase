from restore_activeGuard import test

def iterateFailures( numFailureCombinations, maxNumComponentFailure, debug):   
   for i in range(numFailureCombinations):
        numSurvived = numSurvivedComponents(i)
        if ( numSurvived >= numComponents - maxNumComponentFailure ):
            listOfZerosOnes = convertBinaryToList(i, numComponents)
            accuracy = calcAccuracy(listOfZerosOnes)[0]
            weight = calcWeight(surv, listOfZerosOnes)
            acuracyList.append(accuracy)
            weightList.append(weight)
            if debug:
                print(numSurvived, weight, accuracy)
        

def calcAverageAccuracy(acuracyList, weightList):
    averageAccuracy = 0
    for i in range(len(acuracyList)):
        averageAccuracy += acuracyList[i] * weightList[i]
    return averageAccuracy
        
# calculates the weight of each combination of component failures
def calcWeight(survivability, listOfZerosOnes):
    weight = 1
    for i in range(len(listOfZerosOnes)):
        if (listOfZerosOnes[i] == '1'): # if it survives
            weight = weight * survivability[i]
        else: # if it fails
            weight = weight * (1 - survivability[i])
    return weight
    
def numSurvivedComponents(number):
    return countOnes(number)

# counts the mumber of ones in a bit string
def countOnes(number):
     # convert given number into binary
     # output will be like bin(11)=0b1101
     binary = bin(number)
     # now separate out all 1's from binary string
     # we need to skip starting two characters
     # of binary string i.e; 0b
     setBits = [ones for ones in binary[2:] if ones=='1']
     return len(setBits)

# converts a number (e.g. 128) to its binary representation in a list. It converts number 128 to ['1', '0', '0', '0', '0', '0', '0', '0']    
def convertBinaryToList(number, numBits):
    # convert given number into binary
    # output will be like bin(11)=0b1101
    binary = bin(number)
    lst = [bits for bits in binary[2:]]
    # pad '0's to the begging of the list, if its length is not 'numBits'
    for padding in range(max(0,numBits - len(lst))):
        lst.insert(0,'0')
    return lst
    
def calcAccuracy(listOfZerosOnes):
    return test([float(listOfZerosOnes[i]) for i in range(len(listOfZerosOnes))])

def normalizeWeights(weights):
    sumWeights = sum(weights)
    weightNormalized = [(x/sumWeights) for x in weights]
    return weightNormalized
 
# Driver program
if __name__ == "__main__":  
    surv = [0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.7, 0.66]
    #surv = [1, 0.99, 0.95, 0.95, 0.9, 0.9, 0.9, 0.9]
    numComponents = len(surv) # will be 8
    maxNumComponentFailure = 8
    debug = False

    acuracyList = []
    weightList = []
    iterateFailures(2 ** numComponents, maxNumComponentFailure, debug)
    weightList = normalizeWeights(weightList)

    print("Average Accuracy:", calcAverageAccuracy(acuracyList, weightList))
