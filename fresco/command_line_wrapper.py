import logging
from os import makedirs
from os.path import exists, join
from time import time
from sklearn.cross_validation import KFold

from fresco.scope_optimization import scope_optimization
from fresco.utils import parse_model_string, parse_object_string_sample
from fresco.parse_input_files import read_split_file, read_mapping_file
from fresco.feature_vector import FeatureRecord, FeatureVector
from fresco.group_problem_data import ProblemData
from fresco.group_vector_model import GroupVectorModel
from fresco.score_group_vector import CrossValidationGroupVectorScorer
from fresco.action_vector_generator import (ActionVectorGenerator, SplitAction,
                                            MergeAction, DeleteAction)
from fresco.write_results import (write_to_file, testing_output_lines,
                                  feature_output_lines)
from fresco.parallel_processing import multiprocess_functions
from fresco.model_outcome import ModelOutcome

def command_line_argument_wrapper(model, n_iterations, group_map_files,
        start_level, mapping_file, prediction_field, include_only, negate,
        n_maintain, n_generate, score_predictions_function, split_abun_coef,
        split_score_coef, merge_abun_coef, merge_score_coef, delete_abun_coef,
        delete_score_coef, split_proportion, merge_proportion,
        delete_proportion, n_cross_folds, n_processes, output_dir, n_trials):

    """
    Sets up and executes scope optimization on a given problem, runs testing, and writes to files.
    
    Builds the data structures and objects that are used by fresco.scope_optimization from
    command line friendly arguments. Then runs scope optimization, performs cross fold testing
    on the results if instructed, and writes the results to the files.
    
    Args:
        model: A string representing a machine learning classification model to be used
            both for testing in within the optimization process.
        n_iterations: The number of iterations to be completed by the optimization process.
        group_map_files: A list of open files containing tab-separated lines mapping from
            groups to objects. For example:
                Group1    Object1    Object2    Object3
                Group2    Object4    Object5
            Map files should be ordered by decreasing level of scope, i.e., most general to least
            general map files.
        start_level: The starting scope level (map file index) for the optimization process.
        mapping_file: An open file with tab separated lines mapping from a sample ID to its
            properties. The first line should be tab separated identifiers for the properties.
            For example:
                SAMPLE_ID    COLOR    TASTE
                APPLE_g    GREEN    AMAZING
                APPLE_r    RED    AWEFUL
        n_maintain: The number of vectors to be kept after every iteration of optimization.
        n_generate: The number of vectors to be generated for each input vector by the
            vector generator.
        score_predictions_function: A function which takes two lists of class predictions of
            equal length and returns a numerical score. For example:
                def score_predictions(real_classes, predicted_classes):
                    return sum([1 if read_classes[i] != predicted_classes[i] else 0
                                for i in range(len(real_classes))])
        split_abun_coef: The abundance deviation coefficient for the splitting heuristic.
        split_score_coef: The prediction score deviation coefficient for the splitting heuristic.
        merge_abun_coef: The abundance deviation coefficient for the merging heuristic.
        merge_score_coef: The prediction score deviation coefficient for the merging heuristic.
        delete_abun_coef: The abundance deviation coefficient for the deletion heuristic.
        delete_score_coef: The prediction score deviation coefficient for the deletion heuristic.
        split_proportion: The proportion of total features to be split each iteration.
        merge_proportion: The proportion of total features to be split each iteration.
        delete_proportion: The proportion of total features to be split each iteration.
        n_cross_folds: The number of cross folds to use in scoring the vectors for selection.
        n_processes: The number of additional processes to spawn, at maximum.
        output_dir: The directory that the output files will be put in.
        n_trials: The number of cross folds to use in scoring the vectors returned by the
            optimization process. If 0, no testing will be performed.
    """
    if not exists(output_dir):
        makedirs(output_dir)

    log_fp = join(output_dir, 'info.log')
    logging.basicConfig(filename=log_fp, filemode='w', level=logging.DEBUG,
                        format='%(asctime)s\t%(levelname)s\t%(message)s')
    logging.info('Started feature vector optimization process for \'%s\' '
                 'model' % model)
    start_time = time()

    feature_vector_output_fp = join(output_dir,
                                    'feature_vector_output.txt')

    vector_model = GroupVectorModel(parse_model_string(model))
    group_vector_scorer = CrossValidationGroupVectorScorer(score_predictions_function, vector_model, n_cross_folds)
    problem_data, initial_feature_vector = build_problem_data(group_map_files, mapping_file, prediction_field, start_level, include_only, negate, n_processes)
    group_actions = [SplitAction(problem_data, split_proportion, split_abun_coef, split_score_coef),
                     MergeAction(problem_data, merge_proportion, merge_abun_coef, merge_score_coef),
                     DeleteAction(problem_data, delete_proportion, delete_abun_coef, delete_score_coef)]
    vector_generator = ActionVectorGenerator(group_actions, n_generate)

    if n_trials > 0:
        xfold_feature_vectors = [[] for i in range(n_iterations)]
        masks = [(train, test) for train, test in KFold(problem_data.get_n_samples(), n_folds=n_trials, indices=False)]
        for train_mask, test_mask in masks:
            problem_data.set_mask(train_mask)
            iteration_outcomes = scope_optimization(initial_feature_vector, problem_data, group_vector_scorer, vector_generator, n_iterations, n_processes, n_maintain, True)
            for iteration in range(len(iteration_outcomes)):
                xfold_feature_vectors[iteration].append(iteration_outcomes[iteration].feature_vector)
        functions = []
        mask_results = []
        for iteration in range(len(iteration_outcomes)):
            for mask_index in range(len(masks)):
                functions.append( (mask_testing, (problem_data, masks[mask_index], vector_model, score_predictions_function, xfold_feature_vectors[iteration][mask_index], (iteration, mask_index))) )
        multiprocess_functions(functions, mask_results.append, n_processes)
        test_outcomes = [[None for x in range(len(masks))] for i in range(n_iterations)]
        for tag, mask_result in mask_results:
            iteration, mask_index = tag
            test_outcomes[iteration][mask_index] = mask_result

        prediction_testing_output_fp = join(output_dir,
                                            'prediction_testing_output.txt')
        write_to_file(testing_output_lines(test_outcomes),
                      prediction_testing_output_fp)
        
        avg_outcome = stitch_avg_outcome(test_outcomes[-1], masks)

        write_to_file(feature_output_lines(avg_outcome),
                      feature_vector_output_fp)
    else:
        outcome = scope_optimization(initial_feature_vector, problem_data, group_vector_scorer, vector_generator, n_iterations, n_processes, n_maintain, False)
        write_to_file(feature_output_lines(outcome), feature_vector_output_fp)

    end_time = time()
    elapsed_time = end_time - start_time
    logging.info('Finished feature vector optimization process for \'%s\' '
                 'model' % model)
    logging.info('Total elapsed time (in seconds): %d' % elapsed_time)

def mask_testing(problem_data, masks, vector_model, score_predictions_function, feature_vector, ordering_tag=None):
    train_mask, test_mask = masks
    problem_data.set_mask(train_mask)
    vector_model.fit(problem_data, feature_vector)
    problem_data.set_mask(test_mask)
    test_predictions = vector_model.predict(problem_data, feature_vector)
    prediction_quality = score_predictions_function(problem_data.get_response_variables(), test_predictions)
    feature_scores = vector_model.get_feature_scores()

    outcome = ModelOutcome(feature_vector, feature_scores, prediction_quality, test_predictions)

    if ordering_tag != None:
        return (ordering_tag, outcome)
    else:
        return outcome
   
def stitch_avg_outcome(outcome_list, masks):
    n_features = len(outcome_list[0].feature_vector.get_record_list())
    n_samples = len(masks[0][0])
    n_outcomes = len(outcome_list)
    
    feature_vector = outcome_list[0].feature_vector    
    avg_prediction_score = sum([outcome.prediction_quality for outcome in outcome_list])/float(n_outcomes)
    average_feature_scores = []
    for i in range(n_features):
        avg_feature_score = sum([outcome.feature_scores[i] for outcome in outcome_list])/float(n_outcomes)
        average_feature_scores.append(avg_feature_score)
    
    predictions = [None for i in range(n_samples)]
    for outcome_index in range(len(outcome_list)):
        train_mask, test_mask = masks[outcome_index]
        p_index = 0
        for m_index in range(len(test_mask)):
            if test_mask[m_index]:
                predictions[m_index] = outcome_list[outcome_index].predictions[p_index]
                p_index += 1
                
    avg_outcome = ModelOutcome(feature_vector, average_feature_scores, avg_prediction_score, predictions)
    return avg_outcome
   
def build_problem_data(group_map_files, mapping_file, prediction_field,
                       start_level, include_only, negate, n_processes):
    #For each scope, build a map from group to object and vice versa
    group_to_object = []
    object_to_group = []
    for map_file in group_map_files:
        g_to_o, o_to_g = read_split_file(map_file)
        group_to_object.append(g_to_o)
        object_to_group.append(o_to_g)

    #Find a list of sample names from our group names
    #An alternative is 'samplenames = samplemap.keys()', but that may have records without features
    samplenames = set()
    for grp in group_to_object[start_level]:
        l = group_to_object[start_level][grp]
        for obj in l:
            samplenames.add(parse_object_string_sample(obj))
    samplenames = list(samplenames)

    #get a map of sample name to it's properties
    samplemap = read_mapping_file(mapping_file)

    sample_to_response = {}
    for samplename in samplenames:
        if (include_only is None or
            ((samplemap[samplename][include_only[0]] in include_only[1]) ^ negate)):
            sample_to_response[samplename] = samplemap[samplename][prediction_field]

    problem_data = ProblemData(group_to_object, object_to_group, sample_to_response, n_processes)

    feature_vector = FeatureVector([FeatureRecord(group, start_level,
                                                  len(group_to_object[start_level][group]))
                                    for group in group_to_object[start_level].keys()])

    return problem_data, feature_vector
