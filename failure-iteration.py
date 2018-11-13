from restore_baseline import test
import time

def iterateFailures( numFailureCombinations, maxNumComponentFailure, surv, mn=1):   
   for i in range(numFailureCombinations):
        numSurvived = numSurvivedComponents(i)
        if ( numSurvived >= numComponents - maxNumComponentFailure ):
            listOfZerosOnes = convertBinaryToList(i, numComponents)
            stats = calcStats(listOfZerosOnes, mn)
            uStats = stats[0]
            bStats = stats[1]
            weight = calcWeight(surv, listOfZerosOnes)

            uAccuracyList.append(uStats[0])
            uRecallList.append(uStats[1])
            uPrecisionList.append(uStats[2])
            
            bAccuracyList.append(bStats[0])
            bRecallList.append(bStats[1])
            bPrecisionList.append(bStats[2])
            
            weightList.append(weight)        

def calcAverageStats(statList, weightList):
    averageStat = 0
    for i in range(len(statList)):
        averageStat += statList[i] * weightList[i]
    return averageStat
        
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
    
def calcStats(listOfZerosOnes, mn):
    return test([float(listOfZerosOnes[i]) for i in range(len(listOfZerosOnes))], model_number=mn)

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
    num_models = 30
    log_file = 'final_logs/baseline.txt'

    for x in range(20, num_models + 1):
        uAccuracyList = []
        bAccuracyList = []
        uRecallList = []
        bRecallList = []
        uPrecisionList = []
        bPrecisionList = []

        if x > 10 and x <= 20:
            surv = [0.99, 0.98, 0.94, 0.93, 0.9, 0.9, 0.87, 0.87]

        if x > 20:
            surv = [0.8, 0.8, 0.75, 0.7, 0.65, 0.65, 0.6, 0.6]
        
        weightList = []
        iterateFailures(2 ** numComponents, maxNumComponentFailure, surv, mn=x)
        weightList = normalizeWeights(weightList)

        # write accuracy arrays to log
        with open(log_file, 'a') as myFile:
            myFile.write(str(x))
            myFile.write('\n')
            for i in range(len(uAccuracyList)):
                myFile.write(str(uAccuracyList[i]) + ' ' + str(uRecallList[i]) + ' ' + str(uPrecisionList[i]) + ' ' + str(bAccuracyList[i]) + ' ' + str(bRecallList[i]) + ' ' + str(bPrecisionList[i]))
                myFile.write('\n')
            myFile.write('\n')

            myFile.write(str(calcAverageStats(uAccuracyList, weightList)) + '\n')
            myFile.write(str(calcAverageStats(uRecallList, weightList)) + '\n')
            myFile.write(str(calcAverageStats(uPrecisionList, weightList)) + '\n')
            myFile.write(str(calcAverageStats(bAccuracyList, weightList)) + '\n')
            myFile.write(str(calcAverageStats(bRecallList, weightList)) + '\n')
            myFile.write(str(calcAverageStats(bPrecisionList, weightList)) + '\n')
            myFile.write('\n')
            myFile.write('\n')

        print("Results for (U) Unbalanced Test:")
        print("Average Accuracy:", calcAverageStats(uAccuracyList, weightList))
        print("Average Recall:", calcAverageStats(uRecallList, weightList))
        print("Average Precision:", calcAverageStats(uPrecisionList, weightList))

        print("Results for (B) Balanced Test:")
        print("Average Accuracy:", calcAverageStats(bAccuracyList, weightList))
        print("Average Recall:", calcAverageStats(bRecallList, weightList))
        print("Average Precision:", calcAverageStats(bPrecisionList, weightList))
        time.sleep(10)