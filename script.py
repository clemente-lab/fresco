from otu_trees import *

def main():
    parameters = {}
    parameters["mapping_filepath"] = {'key':'--mapping_file', 'type':str, 'help':'Filepath to the mapping file.', 'default':None}
    parameters["otu_map_files_filepath"] = {'key':'--otu_map_files_filepath', 'type':str, 'help':'Filepath to a file containing one otu_map filepath per line.', 'default':None}
    parameters["prediction_field"] = {'key':'--prediction_field', 'type':str, 'help':'The name of the field in the mapping file to predict.', 'default':None}
    parameters["n_of_procs"] = {'key':'--n_procs', 'type':int, 'help':'The number of processes to use in the optimization process.', 'default':1}
    #Optimization options
    parameters["n_of_iterations"] = {'key':'--n_iterations', 'type':int, 'help':'Number of iterations to make in optimizing the feature vector', 'default':10}
    parameters["model"] = {'key':'--model', 'type':float, 'help':'String describing the classifier to use. Select from: \"lr\" (Logistic Regression), \"rf\" (Random Forest) \"sv\" (Linear Support Vector Machine)', 'default':"lr"}
    parameters["n_of_cross_folds"] = {'key':'--n_cross_folds', 'type':int, 'help':'The number of cross folds to use in measuring the effectiveness of a split/merge selection set internally.', 'default':5}
    parameters["test_partition_size"] = {'key':'--test_partition_size', 'type':float, 'help':'The proportion of samples to be held out for measuring the effectiveness of the final feature vector.', 'default':0.0}
    #Functional and conditional parameters
    parameters["split_score_function_str"] = {'key':'--split_score_function_str', 'type':str, 'help': 'A string to specify the scoring function for splitting features. Current options: ' + split_score_functions.keys(), 'default':"DEVIATION_SPLIT_SCORE"}
    parameters["merge_score_function_str"] = {'key':'--merge_score_function_str', 'type':str, 'help':'A string to specify the scoring function for merging features. Current options: ' + merge_score_functions.keys(), 'default':"DEVIATION_MERGE_SCORE"}
    parameters["delete_score_function_str"] = {'key':'--delete_score_function_str', 'type':str, 'help':'A string to specify the scoring function for deleting features. Current options: ' + delete_score_functions.keys(), 'default':"DEVIATION_DELETE_SCORE"}
    parameters["score_predictions_function_str"] = {'key':'--mapping_file', 'type':str, 'help':'A string to specify the scoring function for predicted vs real response variables. Current options: ' + prediction_score_functions.keys(), 'default':"ACCURACY_PROPORTION_SCORE"}
    #conditional
    parameters["spliting_score_coef"] = {'key':'--spliting_score_coef', 'type':float, 'help':'The coeficient on the deviation of the feature score in calculating split score, for use in the DEVIATION_SPLIT_SCORE split scoring function.', 'default':-2}
    parameters["spliting_abun_coef"] = {'key':'--spliting_abun_coef', 'type':float, 'help':'The coeficient on the deviation of the feature abundance in calculating split score, for use in the DEVIATION_SPLIT_SCORE split scoring function.', 'default':1}
    parameters["spliting_proportion"] = {'key':'--spliting_proportion', 'type':float, 'help':'The proportion of the feature vector to be split every iteration', 'default':.025}
    parameters["merging_score_coef"] = {'key':'--merging_score_coef', 'type':float, 'help':'The coeficient on the deviation of the feature score in calculating merge score, for use in the DEVIATION_MERGE_SCORE merge scoring function.', 'default':-2}
    parameters["merging_abun_coef"] = {'key':'--merging_abun_coef', 'type':float, 'help':'The coeficient on the deviation of the feature abundance in calculating merge score, for use in the DEVIATION_MERGE_SCORE merge scoring function.', 'default':-1}
    parameters["merging_proportion"] = {'key':'--merging_proportion', 'type':float, 'help':'The proportion of the feature vector to be merged every iteration', 'default':.025}
    parameters["deletion_score_coef"] = {'key':'--deletion_score_coef', 'type':float, 'help':'The coeficient on the deviation of the feature score in calculating deletion score, for use in the DEVIATION_DELETE_SCORE deletion scoring function.', 'default':0}
    parameters["deletion_abun_coef"] = {'key':'--deletion_abun_coef', 'type':float, 'help':'The coeficient on the deviation of the feature abundance in calculating deletion score, for use in the DEVIATION_DELETE_SCORE deletion scoring function.', 'default':0}
    parameters["deletion_proportion"] = {'key':'--deletion_abun_coef', 'type':float, 'help':'The proportion of the feature vector to be deleted every iteration.', 'default':0}

    parser = argparse.ArgumentParser(description=
                                      'Run predictive models on otu tables/sample tables')
    for key in parameters.keys():
        parser.add_argument(parameters[key]['key'], help=parameters[key]['help'], type=parameters[key]['type'], default=parameters[key]['default'], dest=key)

        
    args = parser.parse_args()
    
    
