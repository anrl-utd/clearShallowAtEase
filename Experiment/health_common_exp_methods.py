import os
from Experiment.health_data_handler import load_data
from sklearn.model_selection import train_test_split

def init_data(use_GCP):
    if use_GCP == True:
        os.system('gsutil -m cp -r gs://anrl-storage/data/mHealth_complete.log ./')
        if not os.path.exists('models/'):
            os.mkdir('models/')
    data,labels= load_data('mHealth_complete.log')
    # split data into train, val, and test
    # 80/10/10 split
    training_data, test_data, training_labels, test_labels = train_test_split(data,labels,random_state = 42, test_size = .20, shuffle = True)
    val_data, test_data, val_labels, test_labels = train_test_split(test_data,test_labels,random_state = 42, test_size = .50, shuffle = True)
    return  training_data, test_data, training_labels, test_labels, val_data, val_labels

def init_common_experiment_params(training_data):
    num_vars = len(training_data[0])
    num_classes = 13
    survivability_settings = [
        [1,1,1],
        [.92,.96,.99],
        [.87,.91,.95],
        [.78,.8,.85],
    ]
    num_train_epochs = 25 
    hidden_units = 250
    batch_size = 1028
    num_iterations = 10
    return num_iterations, num_vars, num_classes, survivability_settings, num_train_epochs, hidden_units, batch_size

def write_n_upload(output_name, output_list, use_GCP):
    # write experiments output to file
    with open(output_name,'w') as file:
        file.writelines(output_list)
        file.flush()
        os.fsync(file)
    # upload file to GCP
    if use_GCP:
        os.system('gsutil -m -q cp -r {} gs://anrl-storage/results/'.format(output_name))
        os.system('gsutil -m -q cp -r *.h5 gs://anrl-storage/models')