import setup_problems
import numpy as np
import argparse
import sys
from utils import *
import analysis
import copy
import time
from feature_vector import *
import cPickle as pickle
from parse import parse_object_name

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def feature_scope_optimization(model, n_iterations, group_map_files, start_level, mapping_file, prediction_field, score_predictions_function, score_function, score_function_parameters, spliting_proportion, merging_proportion, deletion_proportion, test_partition_size, n_cross_folds, n_procs):

    settings = {'cross_folds':n_cross_folds}

    if model == 'rf':
        model = Model(model=RandomForestClassifier(n_estimators=10, compute_importances=True), name="RF 500")
    elif model == 'sv':
        model = Model(model=LinearSVC(), name="Linear SVC")
    else:
        model = Model(model=LogisticRegression(penalty='l2'), name="Logistic Regression L2")
    props = [(spliting_proportion, merging_proportion, deletion_proportion) for i in range(10)]

    #Build maps for group/sequence relationships for each threhold
    group_to_object = []
    object_to_group = []
    for map_file in group_map_files:
        g_to_o, o_to_g = setup_problems.read_split_file(map_file)
        group_to_object.append(g_to_o)
        object_to_group.append(o_to_g)
    samplenames = set()
    for grp in group_to_object[start_level]:
        l = group_to_object[start_level][grp]
        for obj in l:
            samplenames.add(parse_object_name(obj))
    samplenames = list(samplenames)

    samplemap = setup_problems.readMappingFile(mapping_file)
    classmap = setup_problems.build_classmap(samplemap, prediction_field)

    n_keep = 1
    y = np.array([classmap[samplemap[sample][prediction_field]] for sample in samplenames if sample in samplemap.keys()])

    samplename_map = dict( [(samplenames[index], index) 
                            for index in range(len(samplenames))] )

    misc = {}
    rec_list = [FeatureRecord(group, start_level, 
                          misc={'pop':len(group_to_object[start_level][group])})
                for group in group_to_object[start_level].keys()]

    feature_vectors, outcomes = feature_optimization(samplename_map, group_to_object, object_to_group, y,
                                             rec_list, settings, model, misc, n_procs, 
                                             n_iterations, props, n_keep, score_predictions_function,
                                             score_function, score_function_parameters)

    print_features(feature_vectors, outcomes)

def print_features(feature_vector, outcome):
    rec_list = feature_vector.get_record_list()

    populations = np.array([record.get_misc()['pop'] for record in rec_list])
    
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
        print line

def get_test_indecies(y, prop):
    n_records = len(y)
    sample = np.random.choice(n_records, int(n_records*prop), replace=False)
    return sample


def feature_optimization(samplename_map, group_to_object, object_to_group, y, rec_list, settings, 
                     model, misc, procs, niterations, props, n_keep, score_predictions_function,
                     score_function, score_function_parameters, saveall=False):

    initial_sample_matrix = setup_problems.build_sample_matrix(samplename_map, rec_list, 
                                                               group_to_object, None)
    current_feature_vectors = [FeatureVector(rec_list, initial_sample_matrix, samplename_map)]
    
    nclasses = len(set(y))
    if saveall:
        save_vectors = []
        save_outcomes = []

    #get the initial model outcome for the initial splitting scores
    nmisc = {}
    nmisc.update(misc)
    nmisc['feature_vector'] = current_feature_vectors[0]
    functions = [function_from_vector(current_feature_vectors[0], y, "initial run", nclasses, model,
                                      settings, nmisc, score_predictions_function)]
    current_outcomes = []
    outcome_handler = analysis.OutcomeHandler(current_outcomes.append, ())
    analysis.paralel_mp_limited(functions, procs, outcome_handler)
    
    if saveall:
        save_vectors.append(current_feature_vectors[0])
        save_outcomes.append(current_outcomes[0])

    for iteration in range(niterations):
        print "iteration", iteration
        print_progress_report(current_outcomes[0])

        #get a list of functions to try out in paralel
        functions = []
        for vector_index in range(len(current_feature_vectors)):
            feature_vector = current_outcomes[vector_index]['feature_vector']
            #score the current featuress for splitting, the larger the better for splitting
            split_scores, merge_scores, kill_scores = \
                score_function(current_outcomes[vector_index], feature_vector.get_record_list(), *score_function_parameters)
            
            for split_prop, merge_prop, kill_prop in props:
                functions.append(function_from_scores( (feature_vector, y, nclasses, model, settings, 
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
        
        best_outcomes = sorted(outcomes, key = lambda outcome: outcome['accuracy'], reverse=True)
        current_outcomes = best_outcomes[:n_keep]
        current_feature_vectors = [outcome['feature_vector'] for outcome in current_outcomes]
            
        if saveall:
            save_outcomes.append(current_outcomes[0])
            save_vectors.append(current_feature_vectors[0])

    if saveall:
        return save_vectors, save_outcomes
    else:
        return current_feature_vectors[0], current_outcomes[0]

def function_from_scores(params):
    return (process_score_model, params)

def process_score_model(feature_vector, y, nclasses, model, settings, misc,
                        split_scores, merge_scores, kill_scores, split_prop,
                        merge_prop, kill_prop, group_to_object, object_to_group, score_predictions_function):
    
    rec_list = feature_vector.get_record_list()
    split_indecies, merge_indecies, kill_indecies = choose_indecies(rec_list, split_scores, 
                                                                    merge_scores, kill_scores, 
                                                                    split_prop, merge_prop,
                                                                    kill_prop, len(group_to_object))
    
    new_vector = apply_split_merge_kill(feature_vector, split_indecies, merge_indecies, 
                                        kill_indecies, group_to_object, object_to_group)
    
    #using the new vector, make a new job to run in paralel
    nmisc = {}
    nmisc.update(misc)
    nmisc['feature_vector'] = new_vector
    nmisc['props'] = (split_prop, merge_prop, kill_prop)
    function = function_from_vector(new_vector, y, "score model", nclasses, model, settings, nmisc, score_predictions_function)

    return function[0](*function[1])

def apply_split_merge_kill(feature_vector, split_indecies, merge_indecies, kill_indecies, 
                           group_to_object, object_to_group):
    pop_list = []
    pop_list += [(i, 1) for i in split_indecies]
    pop_list += [(i, -1) for i in merge_indecies]
    pop_list += [(i, None) for i in kill_indecies]
    pop_list = sorted(pop_list, reverse=True)
    
    #make a new feature vector to alter, split the features
    new_vector = feature_vector.get_copy()
    
    #since splitting involves poping the features off, we need to do so in reverse order to
    #preserve indecies
    pop_features = [(new_vector.pop_feature(t[0]), t[1]) for t in pop_list]
    for feature_rec, t_change in pop_features:
        if t_change != None:
            split_feature_rec(feature_rec, new_vector, t_change, group_to_object,
                          object_to_group, partial = False)
            
    return new_vector

def choose_indecies(rec_list, split_scores, merge_scores, kill_scores, split_prop, merge_prop, 
                    kill_prop, n_levels):
    #get a random sampling of features to split using the scores
    split_exclude_list = [i for i in range(len(rec_list)) if rec_list[i].get_threshold()
                          == n_levels-1]
    split_indecies = sample_from_scores(split_scores, split_prop, split_exclude_list)
    
    merge_exclude_list = [i for i in range(len(rec_list)) if rec_list[i].get_threshold() == 0
                          ] + list(split_indecies)
    merge_indecies = sample_from_scores(merge_scores, merge_prop, merge_exclude_list)
    
    kill_exclude_list = list(split_indecies) + list(merge_indecies)
    kill_indecies = sample_from_scores(kill_scores, kill_prop, kill_exclude_list)
    
    return split_indecies, merge_indecies, kill_indecies

def get_scores(outcome, feature_list, score_split_function, score_split_params, score_merge_function, score_merge_params, score_delete_function, score_delete_params):
    populations = np.array([record.get_misc()['pop'] for record in rec_list])
    pred_scores = outcome['scores']

    p_scores = std_dev_dists(populations)
    c_scores = std_dev_dists(pred_scores)
    
    scores = [(p_scores[i], c_scores[i]) for i in range(len(rec_list))]
    
    #return a list of the dot products of scores and coefs, unless the record is excluded in which
    #case fill in None
    def dot_prod(coefs, scores, rec_list):
        ret = []
        for x in range(len(rec_list)):
            s = sum([coefs[i] * scores[x][i] for i in range(len(coefs))])
            ret.append(s)
        return ret

    split_scores = dot_prod(split_coefs, scores, rec_list)
    merge_scores = dot_prod(merge_coefs, scores, rec_list)
    kill_scores = dot_prod(kill_coefs, scores, rec_list)

    return split_scores, merge_scores, kill_scores

def deviation_feature_action_scores(outcome, feature_list, split_abun_coef, split_score_coef,
                                    merge_abun_coef, merge_score_coef, delete_abun_coef, 
                                    delete_score_coef):
    abudances = np.array([feature.get_misc()['pop'] for feature in feature_list])
    pred_scores = outcome['scores']

    value_lists = [pred_scores, abudances]
    value_coefs = [[split_score_coef, split_abun_coef],
                   [merge_score_coef, merge_abun_coef],
                   [delete_score_coef, delete_abun_coef]]
    split_scores, merge_scores, delete_scores = deviation_scores(value_lists, value_coefs)

    return split_scores, merge_scores, delete_scores
    
def deviation_scores(value_lists, value_coefs):
    def std_dev_dists(value_list):
        value_list = np.array(value_list)
        avg = np.mean(value_list)
        std = np.std(value_list)
        
        dists = [(s-avg)/std for s in value_list]
        return dists
    deviation_lists = [std_dev_dists(value_list) for value_list in value_lists]
    
    ret_value_lists = []
    for coefs in value_coefs:
        ret_value_list = []
        
        for value_index in range(len(deviation_lists[0])):
            value = sum([coefs[i] * deviation_lists[i][value_index] for i in range(len(coefs))])
            ret_value_list.append(value)
        ret_value_lists.append(ret_value_list)

    return ret_value_lists

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
    else: #since positive, all elements must be 0
        return []

    np.random.seed() #necessary for mutlithreaded code
    sample = np.random.choice(len(scores), int(len(scores)*prop), p=scores, replace=False)

    return sample

def function_from_vector(feature_vector, y, name, nclasses, model, settings, misc, prediction_score_function):
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
    function = (analysis.process, (model, problem, settings, outcome_maker, (prediction_score_function,)))
    return function

def split_feature_rec(feature_rec, feature_vector, threshold_change, group_to_object, object_to_group,
                  partial = False):
    feature_id = feature_rec.get_ID()
    old_threshold = feature_rec.get_threshold()
    
    #assume that we haven't been told to split a max-threshold feature?
    new_threshold = old_threshold+threshold_change #1 to split, -1 to merge
    sequences = group_to_object[old_threshold][feature_id]

    new_features = features_from_sequences(sequences, object_to_group[new_threshold], group_to_object[new_threshold], partial = partial)

    for new_feature in new_features.keys():
        seq_list = new_features[new_feature]
        new_feature_rec = FeatureRecord(new_feature, new_threshold, parents = [feature_rec],
                                misc={'pop':len(seq_list)})
        feature_rec.get_children().append(new_feature_rec)
        #99.5% of time being spent in add_feature
        feature_vector.add_feature(new_feature_rec, seq_list, partial = partial)

        
#gets a map of each feature found to a list of that feature's sequences.
#if partial=True, only sequences included in the original sequence list are included.
def features_from_sequences(sequences, object_to_group, group_to_object, partial = False):
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

def outcome_maker(prediction_score_function, problem, model, settings, predictions, duration = None,
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

def scores_from_model(model):
    if hasattr(model, 'coef_'):
        coefs = np.absolute(model.coef_)
        avgs = coefs[0]
        for c in coefs[1:]:
            avgs += c
        avgs /= len(avgs)
        return avgs
    elif hasattr(model.getModel(), 'feature_importances_'):
        return model.feature_importances_
    print "SCORES COULD NOT BE FOUND"
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
