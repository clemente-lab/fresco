import setup_problems
import numpy as np
import argparse
from utils import *
import analysis
import copy
from feature_vector import *
import parse

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Main interface function for scope optimization library.
#Initializes values for the algorithm
def feature_scope_optimization(model, n_iterations, group_map_files, start_level, mapping_file, prediction_field, include_only, n_trials, n_keep, score_predictions_function, score_function, score_function_parameters, spliting_proportion, merging_proportion, deletion_proportion, test_partition_size, n_cross_folds, n_procs, feature_vector_output, prediction_testing_output, test_holdout):
    #'model' will be passed in as a short identifier string, such as lr or rf
    #parse it to a scikit_learn classifier
    skl_model = parse_model_string(model)

    #props defines the proportions of the features the will be acted on by each action, each trial
    #to add flexibility, different proportions could be used for each trial
    props = [(spliting_proportion, merging_proportion, deletion_proportion) for i in range(n_trials)]
    
    #For each scope, build a map from group to object and vice versa
    group_to_object = []
    object_to_group = []
    for map_file in group_map_files:
        g_to_o, o_to_g = parse.read_split_file(map_file)
        group_to_object.append(g_to_o)
        object_to_group.append(o_to_g)

    #get a map of sample name to it's properties
    samplemap = parse.read_mapping_file(mapping_file)
    
    #get a map of class strings from the samplemap to a numerical class
    classmap = setup_problems.build_classmap(samplemap, prediction_field)

    #Find a list of sample names from our group names
    #An alternative is 'samplenames = samplemap.keys()', but that may have records without features
    samplenames = set()
    for grp in group_to_object[start_level]:
        l = group_to_object[start_level][grp]
        for obj in l:
            samplenames.add(parse.parse_object_name(obj))
    samplenames = list(samplenames)

    samplenames = prune_samplenames(samplenames, samplemap, include_only)

    #build the vector of response variables
    y = np.array([classmap[samplemap[sample][prediction_field]] for sample in samplenames if sample in samplemap.keys()])

    #build a map of samplename -> index of samplename for building feature matrix
    samplename_map = dict( [(samplenames[index], index) 
                            for index in range(len(samplenames))] )

    misc = {}

    #test_indeces = get_test_indecies(len(y), test_holdout)

    #build a list of features to start iterating on
    rec_list = [FeatureRecord(group, start_level,
                          len(group_to_object[start_level][group]))
                for group in group_to_object[start_level].keys()]

    feature_vectors, outcomes = feature_optimization(samplename_map, group_to_object, object_to_group, y,
                                             rec_list, n_cross_folds, skl_model, misc, n_procs, 
                                             n_iterations, props, n_keep, score_predictions_function,
                                             score_function, score_function_parameters, saveall=True)

    print_features_to_file(feature_vectors[-1], outcomes[-1], feature_vector_output)

    print_outcome_to_file(outcomes, prediction_testing_output)

def get_test_indecies(n_records, prop):
    sample = np.random.choice(n_records, int(n_records*prop), replace=False)
    return sample

#Return a list of every sample name that has at least one field/value pair in include_only
#If include_only is None, return the samplenames
def prune_samplenames(samplenames, samplemap, include_only):
    if include_only == None:
        return samplenames
    
    def is_included(name):
        for rule in include_only:
            if samplemap[name][rule[0]] == rule[1]:
                return True
        return False

    return [name for name in samplenames if is_included(name)]

def print_outcome_to_file(outcomes, filepath):
    f = open(filepath, 'w')
    
    header = ("MODEL_NAME", "ACCURACY", "DISTRIBUTION")
    lines = [header]
    for outcome in outcomes:
        rec_list = outcome['feature_vector'].get_record_list()
        t_list = [record.get_threshold() for record in rec_list]
        t_set = set(t_list)
        dist = [(thresh, t_list.count(thresh)) for thresh in sorted(list(t_set))]

        lines.append( (outcome['model_name'], outcome["accuracy"]) )

    for prop in lines:
        line = ""
        for i in range(len(prop)):
            line += str(prop[i])
            if i != len(prop) - 1:
                line += "\t"
        f.write(line+"\n")
    f.close()

#Parse a string describing a classifier (lr, sv, rf) into a Model object
def parse_model_string(model_str):
    if model_str == 'rf':
        return Model(model=RandomForestClassifier(n_estimators=10),
                     name="Random Forest, n=10")
    if model_str == 'sv':
        return Model(model=LinearSVC(), name="Linear SVC")
    else:
        return Model(model=LogisticRegression(penalty='l2'), name="Logistic Regression L2")

#Print the properties of each feature in a feature vector, including it's score in the outcome
def print_features_to_file(feature_vector, outcome, filepath):
    f = open(filepath, 'w')

    rec_list = feature_vector.get_record_list()

    populations = np.array([record.get_pop() for record in rec_list])    
    pred_scores = outcome['scores']

    properties = [(rec_list[index].get_ID(), rec_list[index].get_threshold(), pred_scores[index],
                   populations[index]) for index in range(len(rec_list))] 
    sorted_properties = sorted(properties, key=lambda prop:prop[2], reverse=True)

    header = ("GROUP_ID", "GROUP_SCOPE", "GROUP_SCORE", "GROUP_ABUNDANCE")
    sorted_properties[:0] = [header]

    for prop in sorted_properties:
        line = ""
        for i in range(len(prop)):
            line += str(prop[i])
            if i != len(prop) - 1:
                line += "\t"
        f.write(line+"\n")
    f.close()

def feature_optimization(samplename_map, group_to_object, object_to_group, y, rec_list, n_cross_folds, 
                     model, misc, procs, niterations, props, n_keep, score_predictions_function,
                     score_function, score_function_parameters, saveall=False):

    #build new feature matrix and feature vector
    initial_sample_matrix = setup_problems.build_sample_matrix(samplename_map, rec_list, 
                                                               group_to_object, None)
    initial_feature_vector = FeatureVector(rec_list, initial_sample_matrix, samplename_map)
    nclasses = len(set(y))

    #if we should save all of the iterations' outcomes and feature vectors, initialize lists
    if saveall:
        save_vectors = []
        save_outcomes = []

    #Run an initial classifier to score features

    #add the current feature vector to the properties for the run
    nmisc = {}
    nmisc.update(misc)
    nmisc['feature_vector'] = initial_feature_vector
    #build a function to perform the analysis
    functions = [function_from_vector(initial_feature_vector, y, "initial run", nclasses, model,
                                      n_cross_folds, nmisc, score_predictions_function)]
    #handle the results of the function by adding it to a list
    current_outcomes = []
    outcome_handler = analysis.OutcomeHandler(current_outcomes.append, ())
    #run the analysis
    analysis.paralel_mp_limited(functions, procs, outcome_handler)
    
    if saveall:
        save_vectors.append(current_outcomes[0]['feature_vector'])
        save_outcomes.append(current_outcomes[0])

    #main iteration loop
    for iteration in range(niterations):
        #get a list of functions to try out in paralel
        functions = []
        for vector_index in range(len(current_outcomes)):
            feature_vector = current_outcomes[vector_index]['feature_vector']
            
            #score each feature for splitting, merging and deletion
            split_scores, merge_scores, kill_scores = \
                score_function(current_outcomes[vector_index], feature_vector.get_record_list(), 
                               *score_function_parameters)
            
            #for each entry in the list of proportions, create a new function for performing
            #the actions and running the model
            for split_prop, merge_prop, kill_prop in props:
                functions.append( (process_score_model, (feature_vector, y, nclasses, model, 
                                                        n_cross_folds, 
                                                        nmisc, split_scores, merge_scores, 
                                                        kill_scores, split_prop, merge_prop, 
                                                        kill_prop, group_to_object, 
                                                        object_to_group, score_predictions_function)) )

        
        
        #run the functions in paralel, write the outcomes to a list
        outcomes = []
        outcome_handler = analysis.OutcomeHandler(outcomes.append, ())
        analysis.paralel_mp_limited(functions, procs, outcome_handler)

        #when picking the leads to follow next, we should consider those we already have
        outcomes += current_outcomes
        #sort outcomes by which performed best on the 'score_predictions_function'
        best_outcomes = sorted(outcomes, key = lambda outcome: outcome['accuracy'], reverse=True)
        #keep the 'n_keep' best outcomes to iterate on later
        current_outcomes = best_outcomes[:n_keep]
            
        if saveall:
            save_outcomes.append(current_outcomes[0])
            save_vectors.append(current_outcomes[0]['feature_vector'])

    if saveall:
        return save_vectors, save_outcomes
    else:
        return current_outcomes[0]['feature_vector'], current_outcomes[0]

#Select features to split/merge/delete, run a classifier using those choices, return 
#information about the results
def process_score_model(feature_vector, y, nclasses, model, n_cross_folds, misc,
                        split_scores, merge_scores, kill_scores, split_prop,
                        merge_prop, kill_prop, group_to_object, object_to_group,
                        score_predictions_function):

    #Make random action selections from the scores
    rec_list = feature_vector.get_record_list()
    split_indecies, merge_indecies, kill_indecies = choose_indecies(rec_list, split_scores, 
                                                                    merge_scores, kill_scores, 
                                                                    split_prop, merge_prop,
                                                                    kill_prop, len(group_to_object))
    
    #Build a new feature vector from applying the actions
    new_vector = apply_split_merge_kill(feature_vector, split_indecies, merge_indecies, 
                                        kill_indecies, group_to_object, object_to_group)
    
    #Using the new vector, build and run a new classifier
    nmisc = {}
    nmisc.update(misc)
    nmisc['feature_vector'] = new_vector
    nmisc['props'] = (split_prop, merge_prop, kill_prop)
    function = function_from_vector(new_vector, y, "score model", nclasses, model, n_cross_folds, nmisc, score_predictions_function)

    #Run the classifier and return the results
    return function[0](*function[1])

#Given a set of indecies to perform actions on, create a new feature vector
#Note: If performed in place, might not be save do in the same process as other trials
def apply_split_merge_kill(feature_vector, split_indecies, merge_indecies, kill_indecies, 
                           group_to_object, object_to_group, in_place=True):

    #since applying an action changes the indecies of the features, make a complete list of actions first
    #make a list of tuples, where each is (index of feature, change in scope level) and None is deletion
    pop_list = []
    pop_list += [(i, 1) for i in split_indecies]
    pop_list += [(i, -1) for i in merge_indecies]
    pop_list += [(i, None) for i in kill_indecies]
    #sort the list of actions so that changing elements along the way won't mess up iteration
    pop_list = sorted(pop_list, reverse=True)
    
    if not in_place:
        new_vector = feature_vector.get_copy()
    else:
        new_vector = feature_vector
    
    #pop each action feature out of the list and remember what action to perform on it
    pop_features = [(new_vector.pop_feature(t[0]), t[1]) for t in pop_list]
    #apply the action for each poped feature
    for feature_rec, t_change in pop_features:
        if t_change != None:
            #splitting changes the scope level by +1, merging by -1
            split_feature_rec(feature_rec, new_vector, t_change, group_to_object,
                          object_to_group, partial = False)
            
    return new_vector

#Randomly select indecies for actions based on scores
#To avoid overlapping selections, actions are given priority: split gets first pick, then merge, delete
def choose_indecies(rec_list, split_scores, merge_scores, kill_scores, split_prop, merge_prop, 
                    kill_prop, n_levels):
    #For splitting, exclude any features that are already at max level
    split_exclude_list = [i for i in range(len(rec_list)) if rec_list[i].get_threshold()
                          == n_levels-1]
    split_indecies = sample_from_scores(split_scores, split_prop, split_exclude_list)
    
    #For merging, exclude any features that are already at min level or have been picked
    merge_exclude_list = [i for i in range(len(rec_list)) if rec_list[i].get_threshold() == 0
                          ] + list(split_indecies)
    merge_indecies = sample_from_scores(merge_scores, merge_prop, merge_exclude_list)
    
    #For deletion, exclude features that have already been picked
    kill_exclude_list = list(split_indecies) + list(merge_indecies)
    kill_indecies = sample_from_scores(kill_scores, kill_prop, kill_exclude_list)
    
    return split_indecies, merge_indecies, kill_indecies

#A function for generating action scores for features.
#Scores are generated by taking a weighted sum of the deviations of the population and prediction scores
def deviation_feature_action_scores(outcome, feature_list, split_abun_coef, split_score_coef,
                                    merge_abun_coef, merge_score_coef, delete_abun_coef, 
                                    delete_score_coef):
    abudances = np.array([feature.get_pop() for feature in feature_list])
    pred_scores = outcome['scores']

    value_lists = [pred_scores, abudances]
    value_coefs = [[split_score_coef, split_abun_coef],
                   [merge_score_coef, merge_abun_coef],
                   [delete_score_coef, delete_abun_coef]]
    split_scores, merge_scores, delete_scores = deviation_scores(value_lists, value_coefs)

    return split_scores, merge_scores, delete_scores
    
#Generates one list of scores for each set of coeficients in value_coefs.
#Each list of scores is an inner product of each element of value_coefs and the deviations of value_lists
def deviation_scores(value_lists, value_coefs):
    #Helper function for generating the deviation list from a list of floats
    #Deviation is used here to mean the distance each element in a list is from the list's average, in
    #    units of standard deviation
    def std_dev_dists(value_list):
        value_list = np.array(value_list)
        avg = np.mean(value_list)
        std = np.std(value_list)
        
        dists = [(s-avg)/std for s in value_list]
        return dists

    deviation_lists = [std_dev_dists(value_list) for value_list in value_lists]
    
    ret_value_lists = []
    #generate a list of scores for each set of coeficients
    #we do this in a single function to avoid regenerating deviation_lists
    for coefs in value_coefs:
        ret_value_list = []
        
        for value_index in range(len(deviation_lists[0])):
            #each value is the linear combination of elements in deviation_lists, weighted by coefs
            value = sum([coefs[i] * deviation_lists[i][value_index] for i in range(len(coefs))])
            ret_value_list.append(value)
        ret_value_lists.append(ret_value_list)

    return ret_value_lists

#Randomly sample a proportion of a list of (not necessarily normalized) probability scores
#exclude_list is a list of indecies to avoid picking
#To avoid negative values, the whole list of scores is shifted up by the minimum value (if negative)
def sample_from_scores(scores, prop, exclude_list):
    #make positive and normalize
    mini = min(scores)
    if float(mini) < 0:
        scores -= mini

    for e in exclude_list:
        scores[e] = 0
        
    sumi = sum(scores)
    if float(sumi) != 0:
        scores /= sumi
    else: #since all elements are non-negative, all elements must be 0
        return []

    np.random.seed() #necessary for mutlithreaded code
    sample = np.random.choice(len(scores), int(len(scores)*prop), p=scores, replace=False)

    return sample

#Generate a (function, arguments) tuple that will run a classifier, given a feature_vector
def function_from_vector(feature_vector, y, name, nclasses, model, n_cross_folds, misc, prediction_score_function):
    nmisc = {}
    nmisc.update(misc)
    nmisc['feature_vector'] = feature_vector

    rarify = 500.0
    cp = np.copy(feature_vector.sampletable_arr)
    for r_index in range(len(cp)):
        row = cp[r_index]
        s = rarify / np.sum(row)
        cp[r_index] /= s

    problem = Problem(feature_vector.sampletable_arr, y, nclasses, name, nmisc)
    function = (analysis.process, (model, problem, n_cross_folds, outcome_maker, (prediction_score_function,)))
    return function

def split_feature_rec(feature_rec, feature_vector, threshold_change, group_to_object, object_to_group,
                  partial = False):
    feature_id = feature_rec.get_ID()
    old_threshold = feature_rec.get_threshold()
    
    #assume that we haven't been told to split a max-threshold feature?
    new_threshold = old_threshold+threshold_change #1 to split, -1 to merge
    sequences = group_to_object[old_threshold][feature_id]

    new_groups = groups_from_objects(sequences, object_to_group[new_threshold], group_to_object[new_threshold], partial = partial)

    for new_group in new_groups.keys():
        seq_list = new_groups[new_group]
        new_feature_rec = FeatureRecord(new_group, new_threshold, len(seq_list))
        feature_vector.add_feature(new_feature_rec, seq_list, partial = partial)

#gets a map of each feature found to a list of that feature's sequences.
#if partial=True, only sequences included in the original sequence list are included.
def groups_from_objects(sequences, object_to_group, group_to_object, partial = False):
    features = {}
    for sequence in sequences:
        #there might have been a clustering failure at this threshold level, in which case the
        #sequence might not be in our lookup table
        try:
            feature = object_to_group[sequence]
            if partial:
                #only add the sequence that found the feature
                if not feature in features.keys():
                    features[feature] = []
                features[feature].append(sequence)
            else:        
                #add all of the feature's sequences
                if not feature in features.keys():
                    features[feature] = group_to_object[feature]
        except KeyError:
            continue
    return features

#Function to take information from analysis.process and produce a useful dictionary of values
def outcome_maker(prediction_score_function, problem, model, n_cross_folds, predictions, duration = None,
                  importances = None):
    result = {}

    result['model_name'] = model.getName()
    result['problem_name'] = problem.getName()
    if duration != None:
        result['duration'] = duration
    if importances != None:
        result['importances'] = importances
    result['scores'] = scores_from_model(model.getModel())
    result['accuracy'] = prediction_score_function(problem.getY(), predictions)
    result['predictions'] = predictions
    if 'feature_vector' in problem.getMisc():
        result['feature_vector'] = problem.getMisc()['feature_vector']
    result.update(problem.getMisc())

    return result

#Get a list of score values from a model for each feature, where the scores grow with feature importance
def scores_from_model(model):
    if hasattr(model, 'coef_'):
        #Some models report the coeficients of each feature instead of the importances
        #For these, multiclass problems will generate a score lists for each class
        #To get a roughly meaningful single score list, we average these score lists
        coefs = np.absolute(model.coef_)
        if len(coefs.shape) == 2:
            avgs = coefs[0]
            for c in coefs[1:]:
                avgs += c
            avgs /= len(avgs)
            return avgs
        else:
            return coefs
    elif hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    return None

def print_progress_report(current_outcome):
    feature_vector = current_outcome['feature_vector']
    rec_list = feature_vector.get_record_list()
    t_list = [record.get_threshold() for record in rec_list]
    t_set = set(t_list)
    dist = [(thresh, t_list.count(thresh)) for thresh in sorted(list(t_set))]
    print '\t Accuracy:', current_outcome['accuracy']
    print '\t Distribution:', dist
    if 'props' in current_outcome.keys():
        print '\t Action proportions:', current_outcome['props']
