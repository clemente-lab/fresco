import scope_optimization
import analysis
import argparse

settings_filepath = "fresco.config"

def main():
    score_functions = {'DEVIATION_SCORE':scope_optimization.deviation_feature_action_scores}
    prediction_score_functions = {'ACCURACY_PROPORTION_SCORE':analysis.get_average_list_accuracy}

    parameters = {}

    # Input/output and resource allocation options
    parameters["mapping_file"] = {'key':'--mapping_file', 'type':str, 'help':'Filepath to the mapping file.'}
    parameters["group_map_files"] = {'key':'--group_map_files', 'type':str, 'help':'Filepath to a file containing one group map filepath per line. These filepaths should be in order from largest groups to smallest groups. For example, 61%%, 94%%, and 99%% OTU maps.'}
    parameters["prediction_field"] = {'key':'--prediction_field', 'type':str, 'help':'The name of the field in the mapping file to predict.'}
    parameters["include_only"] = {'key':'--include_only', 'type':str, 'help': 'A string to specify the type of samples to be included in building the feature vector. Format should be a comma separated list of FIELD:VALUE pairs, where FIELD and VALUE correspond to a field name and value from samples in the mapping file', 'default':""}
    parameters["n_procs"] = {'key':'--n_procs', 'type':int, 'help':'The number of processes to use in the optimization process.', 'default':1}
    parameters["prediction_testing_output"] = {'key':'--prediction_testing_output', 'type':str, 'help': 'Filepath to output information about the final prediction accuracy.', 'default':"prediction_testing_output.txt"}
    parameters["feature_vector_output"] = {'key':'--feature_vector_output', 'type':str, 'help': 'Filepath to output information about the finished feature vector.', 'default':"feature_vector_output.txt"}

    # Optimization options
    parameters["n_iterations"] = {'key':'--n_iterations', 'type':int, 'help':'Number of iterations to make in optimizing the feature vector', 'default':10}
    parameters["start_level"] = {'key':'--start_level', 'type':int, 'help':'The index of the group map file at which all features will start (uses zero-based indexing)', 'default':0}
    parameters["n_trials"] = {'key':'--n_trials', 'type':int, 'help':'The number of random trials to be compared every iteration, per feature vector saved.', 'default':10}
    parameters["n_keep"] = {'key':'--n_keep', 'type':int, 'help':'The number of feature vectors to be saved every iteration. For example, for --n_keep 2, the top 2 scoring models will be saved every iteration.', 'default':1}
    parameters["model"] = {'key':'--model', 'type':str, 'help':'String describing the classifier to use. Select from: \"lr\" (Logistic Regression), \"rf\" (Random Forest) \"sv\" (Linear Support Vector Machine)', 'default':"lr"}
    parameters["n_cross_folds"] = {'key':'--n_cross_folds', 'type':int, 'help':'The number of cross folds to use in measuring the effectiveness of a split/merge selection set internally.', 'default':5}
    parameters["test_partition_size"] = {'key':'--test_partition_size', 'type':float, 'help':'The proportion of samples to be held out for measuring the effectiveness of the final feature vector.', 'default':0.0}
    parameters["test_holdout"] = {'key':'--test_holdout', 'type':float, 'help':'The proportion of the data to be held out for testing.', 'default':0.0}

    # Functional options
    parameters["score_function"] = {'key':'--score_function_str', 'type':str, 'help': 'A string to specify the scoring function for splitting, merging or deleting features. Current options: ' + str(score_functions.keys()), 'default':"DEVIATION_SCORE"}
    parameters["score_predictions_function"] = {'key':'--score_predictions_function_str', 'type':str, 'help':'A string to specify the scoring function for predicted vs real response variables. Current options: ' + str(prediction_score_functions.keys()), 'default':"ACCURACY_PROPORTION_SCORE"}

    # Conditional options
    parameters["spliting_score_coef"] = {'key':'--spliting_score_coef', 'type':float, 'help':'The coeficient on the deviation of the feature score in calculating split score, for use in the DEVIATION_SCORE split scoring function.', 'default':-2}
    parameters["spliting_abun_coef"] = {'key':'--spliting_abun_coef', 'type':float, 'help':'The coeficient on the deviation of the feature abundance in calculating split score, for use in the DEVIATION_SCORE split scoring function.', 'default':1}
    parameters["spliting_proportion"] = {'key':'--spliting_proportion', 'type':float, 'help':'The proportion of the feature vector to be split every iteration', 'default':.025}
    parameters["merging_score_coef"] = {'key':'--merging_score_coef', 'type':float, 'help':'The coeficient on the deviation of the feature score in calculating merge score, for use in the DEVIATION_SCORE merge scoring function.', 'default':-2}
    parameters["merging_abun_coef"] = {'key':'--merging_abun_coef', 'type':float, 'help':'The coeficient on the deviation of the feature abundance in calculating merge score, for use in the DEVIATION_SCORE merge scoring function.', 'default':-1}
    parameters["merging_proportion"] = {'key':'--merging_proportion', 'type':float, 'help':'The proportion of the feature vector to be merged every iteration', 'default':.025}
    parameters["deletion_score_coef"] = {'key':'--deletion_score_coef', 'type':float, 'help':'The coeficient on the deviation of the feature score in calculating deletion score, for use in the DEVIATION_SCORE deletion scoring function.', 'default':0}
    parameters["deletion_abun_coef"] = {'key':'--deletion_abun_coef', 'type':float, 'help':'The coeficient on the deviation of the feature abundance in calculating deletion score, for use in the DEVIATION_SCORE deletion scoring function.', 'default':0}
    parameters["deletion_proportion"] = {'key':'--deletion_proportion', 'type':float, 'help':'The proportion of the feature vector to be deleted every iteration.', 'default':0}
    
    defaults_from_settings_file(settings_filepath, parameters)

    parameter_values = get_commandline_parameters(parameters, 
                                                  'Run predictive models on variable-sized features')

    lookup_pairs = [('score_function', score_functions),
                    ('score_predictions_function', prediction_score_functions)]
    evaluate_string_lookups(lookup_pairs, parameter_values)

    filepath_args = ["group_map_files", "mapping_file"] 
    for filepath_arg in filepath_args:
        parameter_values[filepath_arg] = open_file(parameter_values[filepath_arg])
    parameter_values['group_map_files'] = read_file_list(parameter_values['group_map_files'])

    parameter_values['score_function_parameters'] = (parameter_values.pop('spliting_abun_coef'), 
                                                     parameter_values.pop('spliting_score_coef'),
                                                     parameter_values.pop('merging_abun_coef'),
                                                     parameter_values.pop('merging_score_coef'),
                                                     parameter_values.pop('deletion_abun_coef'),
                                                     parameter_values.pop('deletion_score_coef'))

    parameter_values['include_only'] = parse_sample_rules(parameter_values['include_only'])

    scope_optimization.feature_scope_optimization(**parameter_values)
    
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

        key = parts[0]
        try:
            parameter_values[parts[0]] = parameters[parts[0]]['type'](parts[1])
        except ValueError:
            print "The value in the settings file for the parameter", parts[0], "is not of correct type."
            continue

    return parameter_values

main()
