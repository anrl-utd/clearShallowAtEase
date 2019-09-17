import numpy as np
def parseNodeFailureHealth(file_name):
  
    report_dict = {
        "deepFogGuard Plus":
        {
            "[0, 0, 0]": [0] * 10,
            "[0, 0, 1]": [0] * 10,
            "[0, 1, 0]": [0] * 10,
            "[0, 1, 1]": [0] * 10,
            "[1, 0, 0]": [0] * 10,
            "[1, 0, 1]": [0] * 10,
            "[1, 1, 0]": [0] * 10,
            "[1, 1, 1]": [0] * 10,
        },
        "deepFogGuard":
        {
            "[0, 0, 0]": [0] * 10,
            "[0, 0, 1]": [0] * 10,
            "[0, 1, 0]": [0] * 10,
            "[0, 1, 1]": [0] * 10,
            "[1, 0, 0]": [0] * 10,
            "[1, 0, 1]": [0] * 10,
            "[1, 1, 0]": [0] * 10,
            "[1, 1, 1]": [0] * 10,
        }
        ,
        "Vanilla":
        {
            "[0, 0, 0]": [0] * 10,
            "[0, 0, 1]": [0] * 10,
            "[0, 1, 0]": [0] * 10,
            "[0, 1, 1]": [0] * 10,
            "[1, 0, 0]": [0] * 10,
            "[1, 0, 1]": [0] * 10,
            "[1, 1, 0]": [0] * 10,
            "[1, 1, 1]": [0] * 10,
        }
    }
    avg_dict = {
        "deepFogGuard Plus":
        {
            "[0, 0, 0]": 0,
            "[0, 0, 1]": 0,
            "[0, 1, 0]": 0,
            "[0, 1, 1]": 0,
            "[1, 0, 0]": 0,
            "[1, 0, 1]": 0,
            "[1, 1, 0]": 0,
            "[1, 1, 1]": 0,
        },
        "deepFogGuard":
        {
            "[0, 0, 0]": 0,
            "[0, 0, 1]": 0,
            "[0, 1, 0]": 0,
            "[0, 1, 1]": 0,
            "[1, 0, 0]": 0,
            "[1, 0, 1]": 0,
            "[1, 1, 0]": 0,
            "[1, 1, 1]": 0,
        }
        ,
        "Vanilla":
        {
            "[0, 0, 0]": 0,
            "[0, 0, 1]": 0,
            "[0, 1, 0]": 0,
            "[0, 1, 1]": 0,
            "[1, 0, 0]": 0,
            "[1, 0, 1]": 0,
            "[1, 1, 0]": 0,
            "[1, 1, 1]": 0,
        }
    }
    num_iterations = 1
    model_counter = 0
    counter = 0
    # goes from low survival config to high 
    with open(file_name) as file:

        for line in file:
            index = line.find("acc:")
            if index != -1:
                split_line = line.split()
                acc = float(split_line[-1])
                # extract the survival config from the file
                survival_config = line.split('n')[0]
                if model_counter % 3 == 0:
                    report_dict['deepFogGuard Plus'][survival_config][num_iterations-1] = acc
                    print(acc)
                elif model_counter % 3 == 1:
                    report_dict['deepFogGuard'][survival_config][num_iterations-1] = acc
                else:
                    report_dict['Vanilla'][survival_config][num_iterations-1] = acc
                counter+=1 
                if(counter % 96 == 0):
                    num_iterations+=1
                if(counter % 8 == 0):
                    model_counter+=1

    for model in report_dict:
        for survival_config in report_dict[model]:
            avg_dict[model][survival_config] = np.average(report_dict[model][survival_config])
    print(avg_dict)

def parseNodeFailureCifar(file_name):
    iteration_counter = 4 * 4
    report_dict = {
        "model":
        {
            "[0, 0]": [0] * 10,
            "[0, 1]": [0] * 10,
            "[1, 0]": [0] * 10,
            "[1, 1]": [0] * 10,
        }
    }
    avg_dict = {
        "model":
        {
            "[0, 0]": 0,
            "[0, 1]": 0,
            "[1, 0]": 0,
            "[1, 1]": 0,
        },
    }
    num_iterations = 1
    model_counter = 0
    counter = 0
    # goes from low survival config to high 
    with open(file_name) as file:

        for line in file:
            index = line.find("acc:")
            if index != -1:
                split_line = line.split()
                acc = float(split_line[-1])
                # extract the survival config from the file
                survival_config = line.split('n')[0]
                report_dict['model'][survival_config][num_iterations-1] = acc
                counter+=1 
                if(counter % iteration_counter == 0):
                    num_iterations+=1

    for model in report_dict:
        for survival_config in report_dict[model]:
            avg_dict[model][survival_config] = np.average(report_dict[model][survival_config])
            print(np.std(report_dict[model][survival_config]))
    print(avg_dict)

def calculate_cifar_report_stats(file_name):
    reliability_settings = [
        [.98,.96],
        [.95,.90],
        [.85,.80],
        [1,1],
    ]
    # convert reliability settings into strings so it can be used in the dictionary as keys
    no_failure = str(reliability_settings[0])
    normal = str(reliability_settings[1])
    poor = str(reliability_settings[2])
    hazardous = str(reliability_settings[3])
    report = {
        hazardous:[0] * 10,
        poor:[0] * 10,
        normal: [0] * 10,
        no_failure:[0] * 10,
    }
    # goes from low survival config to high 
    num_iterations = 1
    iteration_counter = 4 
    counter = 0
    with open(file_name) as file:
        for line in file:
            index = line.find("acc:")
            if index != -1:
                split_line = line.split()
                acc = float(split_line[-1])
                report[str(reliability_settings[counter - 1 % iteration_counter])][num_iterations-1] = acc
                counter+=1 
                if(counter % iteration_counter == 0):
                    num_iterations+=1

    for survival_config in report:
        print(survival_config)
        print(np.average(report[survival_config]))
        print(np.std(report[survival_config]))
if __name__ == "__main__":
    #parseFailureHealth("results_newsplit_normalHealthActivityExperiment.txt")
    #parseNodeFailureCifar("reports/"+ "results_experiment3_fixedsplit_normalDeepFogGuardExperiment_results.txt")
    calculate_cifar_report_stats("reports/"+ "results_experiment3_fixedsplit_normalDeepFogGuardExperiment_results.txt")
    