import os

def make_results_folder():
    # makes folder for results and models (if they don't exist)
    if not os.path.exists('results/' ):
        os.mkdir('results/' )
    if not os.path.exists('models'):      
        os.mkdir('models/')

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

def convert_to_string(survivability_settings):
    # convert survivability settings into strings so it can be used in the dictionary as keys
    no_failure = str(survivability_settings[0])
    normal = str(survivability_settings[1])
    poor = str(survivability_settings[2])
    hazardous = str(survivability_settings[3])
    return no_failure, normal, poor, hazardous

def make_output_dictionary_average_accuracy(survivability_settings, num_iterations):
    no_failure, normal, poor, hazardous = convert_to_string(survivability_settings)

    # dictionary to store all the results
    output = {
        "ResiliNet":
        {
            hazardous:[0] * num_iterations,
            poor:[0] * num_iterations,
            normal:[0] * num_iterations,
            no_failure:[0] * num_iterations,
        }, 
        "deepFogGuard":
        {
            hazardous:[0] * num_iterations,
            poor:[0] * num_iterations,
            normal:[0] * num_iterations,
            no_failure:[0] * num_iterations,
        },
        "Vanilla": 
        {
            hazardous:[0] * num_iterations,
            poor:[0] * num_iterations,
            normal:[0] * num_iterations,
            no_failure:[0] * num_iterations,
        },
    }
    return output