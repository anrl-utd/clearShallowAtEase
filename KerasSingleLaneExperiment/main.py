# set the RNG seeds
import numpy as np
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(2)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from keras.models import Model
import keras.backend as K
import datetime
import os

from KerasSingleLaneExperiment.deepFogGuardPlus import define_deepFogGuardPlus
from KerasSingleLaneExperiment.deepFogGuard import define_deepFogGuard
from KerasSingleLaneExperiment.Vanilla import define_vanilla_model
from KerasSingleLaneExperiment.random_guess import model_guess
from KerasSingleLaneExperiment.loadData import load_data

def fail_node(model,node_array):
    """fails node by making the specified node/nodes output 0
    ### Arguments
        model (Model): Keras model to have nodes failed
        node_array (list): bit list that corresponds to the node arrangement, 1 in the list represents to alive and 0 corresponds to failure 
    ### Returns
        return a boolean whether the model failed was a cnn or not
    """
    is_cnn = True
    # cnn failure
    if is_cnn:
        nodes = ["conv_pw_3","conv_pw_8"]
        for index,node in enumerate(node_array):
            # node failed
            if node == 0:
                layer_name = nodes[index]
                layer = model.get_layer(name=layer_name)
                layer_weights = layer.get_weights()
                # make new weights for the connections
                new_weights = np.zeros(layer_weights[0].shape)
                layer.set_weights([new_weights])
                print(layer_name, "was failed")
    # regular NN failure
    else:
        for index,node in enumerate(node_array):
            # node failed
            if node == 0:
                layer_name = "fog" + str(index + 1) + "_output_layer"
                layer = model.get_layer(name=layer_name)
                layer_weights = layer.get_weights()
                # make new weights for the connections
                new_weights = np.zeros(layer_weights[0].shape)
                #new_weights[:] = np.nan # set weights to nan
                # make new weights for biases
                new_bias_weights = np.zeros(layer_weights[1].shape)
                layer.set_weights([new_weights,new_bias_weights])
                print(layer_name, "was failed")
    return is_cnn

# function to return average of a list 
def average(list):
    if len(list) == 0:
        return 0
    else:
        return sum(list) / len(list)
        
def train_model(training_data,training_labels,model_type, survive_rates):
    """trains Keras model from training data and training labels
    ### Arguments
        training_data (numpy array): 2D array that contains the training data, assumes that each column is a variable and that each row is a training example
        training_labels (numpy array): 1D array that corresponds to each row in the training data with a class label
    ### Returns
        return trained Keras Model
    """
    # variable to save the model
    save_model = True

    # create model
    if model_type == 0:
        model = define_vanilla_model(num_vars,num_classes,250)
    elif model_type == 1:
        model = define_deepFogGuard(num_vars,num_classes,250,survive_rates)
    elif model_type == 2:
        model = define_deepFogGuardPlus(num_vars,num_classes,250,survive_rates)
    else:
        raise ValueError("Incorrect model type")

    # fit model on training data
    model.fit(training_data,training_labels, epochs=10, batch_size=1028,verbose=1,shuffle = True,callbacks=[])

    # get date when training occured 
    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    path = 'weights/' + date
    if not os.path.exists(path):
        os.mkdir(path)
    if save_model:
        model.save_weights(path + '/deepfoggguardTest_highconfig_withexperiment2activegaurd' + '.h5')
    return model

def load_model(num_vars, num_classes, hidden_units, weights_path,model_type,survive_rates):
    """creates specific type of model and loads weights for the model
    ### Arguments
        num_vars(int): specifies number of variables from the data, used to determine input size.
        num_classes (int): specifies number of classes to be outputted by the model
        hidden_units (int): specifies number of hidden units per layer in network
        weights_path (string): path where the weights are located
        model_type (int): determines what type of model is being loaded
        survive_rates (list): bit list that corresponds to the node arrangement, 1 in the list represents to alive and 0 corresponds to failure 
    ### Key for Model Type
      model type 0 = vanilla \\ 
      model type 1 = deepFogGuard \\ 
      model type 2 = deepFogPlus
    ### Returns
        return a loaded model
    ### Exception
        Throws a value error exception if incorrect model type is specified 
    """
   # create model
    if model_type == 0:
        model = define_vanilla_model(num_vars,num_classes,250)
    elif model_type == 1:
        model = define_deepFogGuard(num_vars,num_classes,250,survive_rates)
    elif model_type == 2:
        model = define_deepFogGuardPlus(num_vars,num_classes,250,survive_rates)
    else:
        raise ValueError("Incorrect model type")
    model.load_weights(weights_path)
    return model

def evaluate_withFailures(model,test_data,test_labels):
    """Evaluates model performance for all survival configurations
    ### Arguments
        model (Model): Keras model to have nodes failed
        test_data (numpy array): 2D array that contains the test data, assumes that each column is a variable and that each row is a test example
        test_labels (numpy array): 1D array that corresponds to each row in the test data with a class label
    ### Returns
        Prints model test accuracy
    """
    failure_list = [
        [1,1,1], # no failures
        [0,1,1], # fog node 1 fails
        [0,0,1], # fog node 1 and 2 fail
        [0,1,0], # fog node 1 and 3 fail
        [1,0,0], # fog node 2 and 3 fail
        [1,0,1], # fog node 2 fails
        [1,1,0], # fog node 3 fails
    ]
    # keep track of original weights so it does not get overwritten by fail_node
    original_weights = model.get_weights()
    for failure in failure_list:
        print(failure)
        fail_node(model,failure)
        acc = model.evaluate(test_data,test_labels)[1]
        print(acc)
        # reset the model with the original weights 
        model.set_weights(original_weights)

def test(survive_array):
    """ calculate expected value of network performance, meant to be called by outside function
    ### Arguments
        survive_array (list): Keras model to have nodes failed
    ### Returns
        return a tuple of accuracy as a float and whether there was total network failure as an integer
    """
    data,labels= load_data('mHealth_complete.log')
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .2, shuffle = True,stratify = labels)
    num_vars = len(training_data[0])
    num_classes = 13
    path = 'weights/2-26-2019/250 units 10 layers baseline [.7, .8, .85] adam 25 epochs .8 split stratify.h5'
    model_type = 0
    survive_rates = [.7,.8,.85]
    model = load_model(num_vars = num_vars, num_classes = num_classes, hidden_units = 250, weights_path = path, model_type = model_type,survive_rates=survive_rates)
    fail_node(model,survive_array)
    return model_guess(model,training_labels,test_data,test_labels)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # # load data
    data,labels= load_data('mHealth_complete.log')
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .2, shuffle = True, stratify = labels)

    # define number of classes and variables in the data
    num_vars = len(training_data[0])
    num_classes = 13

    # define model type
    model_type = 1

    load_weights = False
    if load_weights:
        path = 'weights/7-11-2019/deepfoggguardTest.h5'
        model = load_model(num_vars = num_vars, num_classes = num_classes, hidden_units = 250, weights_path = path, model_type = model_type, survive_rates=[.92,.96,.99])
    else:
        K.set_learning_phase(1)
        model = train_model(training_data,training_labels,model_type=model_type,survive_rates=[.92,.96,.99])
    K.set_learning_phase(0)
    evaluate_withFailures(model,test_data,test_labels)
   
