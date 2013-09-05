#!/usr/bin/env python

import argparse
from fresco.command_line_wrapper import command_line_argument_wrapper
from fresco.utils import get_list_accuracy

settings_filepath = "fresco.config"

def main():
    prediction_score_functions = {'ACCURACY_PROPORTION_SCORE':get_list_accuracy}

    parameters = {}

    # Input/output and resource allocation options
    parameters["mapping_file"] = {'key':'--mapping_file', 'type':str, 'help':'Filepath to the mapping file.'}
    parameters["group_map_files"] = {'key':'--group_map_files', 'type':str, 'help':'Filepath to a file containing one group map filepath per line. These filepaths should be in order from largest groups to smallest groups. For example, 61%%, 94%%, and 99%% OTU maps.'}
    parameters["output_dir"] = {'key':'--output_dir', 'type':str, 'help': 'Directory to write results to. Results will include information about the final prediction accuracy, as well as information about the finished feature vector.'}
    parameters["prediction_field"] = {'key':'--prediction_field', 'type':str, 'help':'The name of the field in the mapping file to predict.'}
    parameters["include_only"] = {'key':'--include_only', 'type':str, 'help': 'A string to specify the type of samples to be included in building the feature vector. Format should be a comma separated list of FIELD:VALUE pairs, where FIELD and VALUE correspond to a field name and value from samples in the mapping file', 'default':""}
    parameters["n_processes"] = {'key':'--n_procs', 'type':int, 'help':'The number of processes to use in the optimization process.', 'default':1}

    # Optimization options
    parameters["n_iterations"] = {'key':'--n_iterations', 'type':int, 'help':'Number of iterations to make in optimizing the feature vector', 'default':10}
    parameters["start_level"] = {'key':'--start_level', 'type':int, 'help':'The index of the group map file at which all features will start (uses zero-based indexing)', 'default':0}
    parameters["n_trials"] = {'key':'--n_trials', 'type':int, 'help':'Used for testing functionality, this parameter defines the number of cross folds to use. To omit testing and only generate the optimized feature vector, use value 0.', 'default':5}
    parameters["n_maintain"] = {'key':'--n_maintain', 'type':int, 'help':'The number of feature vectors to be saved every iteration. For example, for --n_maintain 2, the top 2 scoring models will be saved every iteration.', 'default':1}
    parameters["n_generate"] = {'key':'--n_generate', 'type':int, 'help':'The number of feature vectors to be generated from each vector every iteration. For example, for --n_maintain 2 and --n_generate 3, the maxiumum number of vectors to be generated and assessed every iteration is 6.', 'default':1}
    parameters["model"] = {'key':'--model', 'type':str, 'help':'String describing the classifier to use. Select from: \"lr\" (Logistic Regression), \"rf\" (Random Forest) \"sv\" (Linear Support Vector Machine)', 'default':"lr"}
    parameters["n_cross_folds"] = {'key':'--n_cross_folds', 'type':int, 'help':'The number of cross folds to use in measuring the effectiveness of a split/merge selection set internally.', 'default':5}
    
    # Functional options
    parameters["score_predictions_function"] = {'key':'--score_predictions_function_str', 'type':str, 'help':'A string to specify the scoring function for predicted vs real response variables. Current options: ' + str(prediction_score_functions.keys()), 'default':"ACCURACY_PROPORTION_SCORE"}

    # Conditional options
    parameters["split_score_coef"] = {'key':'--spliting_score_coef', 'type':float, 'help':'The coeficient on the deviation of the feature score in calculating split score, for use in the DEVIATION_SCORE split scoring function.', 'default':-2}
    parameters["split_abun_coef"] = {'key':'--spliting_abun_coef', 'type':float, 'help':'The coeficient on the deviation of the feature abundance in calculating split score, for use in the DEVIATION_SCORE split scoring function.', 'default':1}
    parameters["split_proportion"] = {'key':'--spliting_proportion', 'type':float, 'help':'The proportion of the feature vector to be split every iteration', 'default':.025}
    parameters["merge_score_coef"] = {'key':'--merging_score_coef', 'type':float, 'help':'The coeficient on the deviation of the feature score in calculating merge score, for use in the DEVIATION_SCORE merge scoring function.', 'default':-2}
    parameters["merge_abun_coef"] = {'key':'--merging_abun_coef', 'type':float, 'help':'The coeficient on the deviation of the feature abundance in calculating merge score, for use in the DEVIATION_SCORE merge scoring function.', 'default':-1}
    parameters["merge_proportion"] = {'key':'--merging_proportion', 'type':float, 'help':'The proportion of the feature vector to be merged every iteration', 'default':.025}
    parameters["delete_score_coef"] = {'key':'--deletion_score_coef', 'type':float, 'help':'The coeficient on the deviation of the feature score in calculating deletion score, for use in the DEVIATION_SCORE deletion scoring function.', 'default':0}
    parameters["delete_abun_coef"] = {'key':'--deletion_abun_coef', 'type':float, 'help':'The coeficient on the deviation of the feature abundance in calculating deletion score, for use in the DEVIATION_SCORE deletion scoring function.', 'default':0}
    parameters["delete_proportion"] = {'key':'--deletion_proportion', 'type':float, 'help':'The proportion of the feature vector to be deleted every iteration.', 'default':0}
    
    defaults_from_settings_file(settings_filepath, parameters)

    parameter_values = get_commandline_parameters(parameters, 
                                                  'Run predictive models on variable-sized features')

    lookup_pairs = [('score_predictions_function', prediction_score_functions)]
    evaluate_string_lookups(lookup_pairs, parameter_values)

    filepath_args = ["group_map_files", "mapping_file"] 
    for filepath_arg in filepath_args:
        parameter_values[filepath_arg] = open_file(parameter_values[filepath_arg])
    parameter_values['group_map_files'] = read_file_list(parameter_values['group_map_files'])

  

    parameter_values['include_only'] = parse_sample_rules(parameter_values['include_only'])

    command_line_argument_wrapper(**parameter_values)
    
def parse_sample_rules(rule_string):
    if rule_string == '':
        return None
    pairs = [tuple(pair_string.split(":")) for pair_string in rule_string.split(",")]
    return pairs

def evaluate_string_lookups(lookup_pairs, parameter_values):
    try:
        for fieldname, function_lookup in lookup_pairs:
            parameter_values[fieldname] = function_lookup[parameter_values[fieldname]]
    except KeyError:
        print "Invalid function identifier."
        exit(1)

def open_file(filepath):
    try:
        return open(filepath)
    except:
        print "Invalid filepath:", filepath
        exit(1)
    
def read_file_list(file_list_file):
    ret_list = []
    for line in file_list_file:
        ret_list.append(open_file(line.strip(' \n')))
    return ret_list

def get_commandline_parameters(parameters, description):
    parser = argparse.ArgumentParser(description=description)

    for key in parameters.keys():
        if 'default' in parameters[key].keys():
            parser.add_argument(parameters[key]['key'], help=parameters[key]['help'], type=parameters[key]['type'], default=parameters[key]['default'], dest=key)
        else:    
            parser.add_argument(parameters[key]['key'], help=parameters[key]['help'], type=parameters[key]['type'], dest=key, required=True)
    parser.add_argument('--generate_settings_file', help='Generate a blank settings file', action='store_true', dest='generate_settings_file')

    args = parser.parse_args()
    
    parameter_values = {}
    for key in parameters.keys():
        parameter_values[key] = eval("args."+key)

    generate_settings_file = args.generate_settings_file
    if generate_settings_file:
        make_blank_settings_file(parameters)
        exit(0)

    return parameter_values

def defaults_from_settings_file(settings_filepath, parameters):
    parameter_values = read_settings_file(settings_filepath, parameters)
    for key in parameter_values:
        parameters[key]['default'] = parameter_values[key]

def make_blank_settings_file(parameters):
    f = open(settings_filepath, 'w')
    
    for parameter in parameters:
        f.write(parameter + ":\n")
    f.close()

def read_settings_file(settings_filepath, parameters):
    parameter_values = {}

    try:
        settings_file = open(settings_filepath)
    except IOError:
        return parameter_values
    for line in settings_file:
        parts = line.split(":")

        parts = [part.strip(' \n') for part in parts]

        if len(parts) != 2 or len(parts[1]) == 0:
            continue
        if not parts[0] in parameters.keys():
            continue

        try:
            parameter_values[parts[0]] = parameters[parts[0]]['type'](parts[1])
        except ValueError:
            print "The value in the settings file for the parameter", parts[0], "is not of correct type."
            continue

    return parameter_values


if __name__ == "__main__":
    main()
