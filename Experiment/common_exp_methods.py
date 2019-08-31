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

def make_output_dictionary_hyperconnection_weight(survivability_settings, num_iterations):
    no_failure, normal, poor, hazardous = convert_to_string(survivability_settings)

    # define weight schemes for hyperconnections
    one_weight_scheme = 1 # weighted by 1
    normalized_survivability_weight_scheme = 2 # normalized survivability
    survivability_weight_scheme = 3 # survivability
    random_weight_scheme = 4 # randomly weighted between 0 and 1
    random_weight_scheme2 = 5 # randomly weighted between 0 and 10
    fifty_weight_scheme = 6  # randomly weighted by .5

    weight_schemes = [
        one_weight_scheme,
        normalized_survivability_weight_scheme,
        survivability_weight_scheme,
        random_weight_scheme,
        random_weight_scheme2,
        fifty_weight_scheme,
    ]

    # dictionary to store all the results
    output = {
        "DeepFogGuard Hyperconnection Weight": 
        {
            one_weight_scheme:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            },
            normalized_survivability_weight_scheme:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            },
            survivability_weight_scheme:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            },
            random_weight_scheme:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            },
            random_weight_scheme2:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            },
            fifty_weight_scheme:
            {
                no_failure: [0] * num_iterations,
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
            }
        },
    }
    return output, weight_schemes

def make_output_dictionary_failout_rate(failout_survival_rates, survivability_settings, num_iterations):
    no_failure, normal, poor, hazardous = convert_to_string(survivability_settings)
    
    # convert dropout rates into strings
    failout_rate_05 =  str(failout_survival_rates[0])
    failout_rate_10 = str(failout_survival_rates[1])
    failout_rate_30 = str(failout_survival_rates[2])
    failout_rate_50 = str(failout_survival_rates[3])
    # dictionary to store all the results
    output = {
        "ResiliNet": 
        {
            failout_rate_05:
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            failout_rate_10 :
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            failout_rate_30:
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            failout_rate_50:
            {
                hazardous:[0] * num_iterations,
                poor:[0] * num_iterations,
                normal:[0] * num_iterations,
                no_failure:[0] * num_iterations,
            },
            "Variable Failout 1x": 
            {
                hazardous:[0] * num_iterations,
                poor :[0] * num_iterations,
                normal:[0] * num_iterations,
            },
            "Variable Failout 10x": 
            {
                hazardous:[0] * num_iterations,
                poor :[0] * num_iterations,
                normal:[0] * num_iterations,
            }
        }
    }
    return output