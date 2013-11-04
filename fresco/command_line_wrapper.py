import logging
from os import makedirs
from os.path import exists, join, isfile
from time import time
import types

from fresco.scope_optimization import ScopeOptimization
from fresco.utils import parse_model_string, check_input_type, InputTypeError
from fresco.feature_vector import FeatureVector
from fresco.group_problem_data import ProblemData, build_problem_data
from fresco.group_vector_model import GroupVectorModel
from fresco.score_group_vector import CrossValidationGroupVectorScorer
from fresco.action_vector_generator import (ActionVectorGenerator, SplitAction,
                                            MergeAction, DeleteAction)
from fresco.write_results import (write_to_file, testing_output_lines,
                                  feature_output_lines, fold_features_output_lines)
from fresco.model_outcome import ModelOutcome
from score_scope_optimization import scope_optimization_cross_validation
import inspect

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
    simple_var_types = [
                 ("n_iterations", types.IntType),
                 ("n_maintain", types.IntType),
                 ("n_generate", types.IntType),
                 ("n_processes", types.IntType),
                 ("n_cross_folds", types.IntType),
                 ("n_trials", types.IntType),
                 ("start_level", types.IntType),
                 ("split_score_coef", (types.IntType, types.FloatType)),
                 ("split_abun_coef", (types.IntType, types.FloatType)),
                 ("merge_score_coef", (types.IntType, types.FloatType)),
                 ("merge_abun_coef", (types.IntType, types.FloatType)),
                 ("delete_score_coef", (types.IntType, types.FloatType)),   
                 ("delete_abun_coef", (types.IntType, types.FloatType)),
                 ("split_proportion", (types.IntType, types.FloatType)),
                 ("merge_proportion", (types.IntType, types.FloatType)),
                 ("delete_proportion", (types.IntType, types.FloatType)),
                 ("n_cross_folds", types.IntType),
                 ("mapping_file", types.FileType),
                 ("negate", types.BooleanType),
                 ("output_dir", types.StringType),
                 ("prediction_field", types.StringType),
                 ("include_only", (types.NoneType, types.ListType,
                                   types.TupleType)),
                 ("group_map_files", types.ListType)
                ]
    for var_name, var_type in simple_var_types:
        check_input_type(var_name, locals()[var_name], var_type)
    if len(inspect.getargspec(score_predictions_function)[0]) < 2:
        raise InputTypeError("scope_predictions_function should take at least two parameters")
    if not all([isinstance(f, types.FileType) for f in group_map_files]):
        raise InputTypeError("group_map_files should be a list of open files")
    if include_only != None:
        if not isinstance(include_only[0], types.StringType):
            raise InputTypeError("include_only[0] should be of type string")
        if not isinstance(include_only[1], types.ListType) or \
                not all([isinstance(value, types.StringType) for value in include_only[1]]):
            raise InputTypeError("include_only[1] should be a list of strings")

    if not exists(output_dir):
        makedirs(output_dir)

    log_fp = join(output_dir, 'info.log')
    logging.basicConfig(filename=log_fp, filemode='w', level=logging.DEBUG,
                        format='%(asctime)s\t%(levelname)s\t%(message)s')
    logging.info('Started feature vector optimization process for \'%s\' '
                 'model' % model)
    start_time = time()

    vector_model = GroupVectorModel(parse_model_string(model))
    group_vector_scorer = CrossValidationGroupVectorScorer(score_predictions_function, vector_model, n_cross_folds)
    
    problem_data = build_problem_data(group_map_files, mapping_file, prediction_field, start_level, include_only, negate, n_processes)
    assert isinstance(problem_data, ProblemData),\
        "build_problem_data did not return a GroupProblemData"
    
    initial_feature_ids = problem_data.get_group_ids(start_level)
    initial_feature_vector = FeatureVector([problem_data.get_feature_record(start_level, feature_id) for feature_id in initial_feature_ids])
    assert isinstance(initial_feature_vector, FeatureVector),\
        "build_problem_data did not return a FeatureVector"
    
    group_actions = [SplitAction(problem_data, split_proportion, split_abun_coef, split_score_coef),
                     MergeAction(problem_data, merge_proportion, merge_abun_coef, merge_score_coef),
                     DeleteAction(problem_data, delete_proportion, delete_abun_coef, delete_score_coef)]
    vector_generator = ActionVectorGenerator(group_actions, n_generate, problem_data)

    scope_optimization = ScopeOptimization(group_vector_scorer, vector_generator, n_processes, n_iterations, n_maintain)

    logging.info('Completed optimization initialization')
    if n_trials > 0:
        test_outcomes = scope_optimization_cross_validation(scope_optimization, initial_feature_vector, problem_data, vector_model, score_predictions_function, n_trials, n_processes)
        
        assert len(test_outcomes) == n_iterations, "test outcomes isn't returning an entry for each iteration"
        assert all([len(mask_outcomes) == n_trials for mask_outcomes in test_outcomes]), \
            "test outcomes isn't returning an outcome for each mask for each iteration"
        assert all([all([isinstance(outcome, ModelOutcome) for outcome in mask_outcomes])
                    for mask_outcomes in test_outcomes]), \
                    "test outcomes are not all of type ModelOutcome"
        
        prediction_testing_output_fp = join(output_dir,
                                            'prediction_testing_output.txt')
        write_to_file(testing_output_lines(test_outcomes),
                      prediction_testing_output_fp)
        assert isfile(prediction_testing_output_fp), \
            "Output file \"%s\"was not created" %prediction_testing_output_fp
            
        feature_vector_output_fp = join(output_dir,
                                        'fold_feature_vectors_output.txt')
        write_to_file(fold_features_output_lines(test_outcomes[-1], problem_data),
                      feature_vector_output_fp)
        assert isfile(feature_vector_output_fp), \
            "Output file \"%s\"was not created" %feature_vector_output_fp
    else:
        outcome = scope_optimization.optimize_vector(initial_feature_vector, problem_data, False)
        assert isinstance(outcome, ModelOutcome), "scope_optimization outcome is not a ModelOutcome"
        
        feature_vector_output_fp = join(output_dir,
                                        'feature_vector_output.txt')
        write_to_file(feature_output_lines(outcome, problem_data), feature_vector_output_fp)
        assert isfile(feature_vector_output_fp), \
            "Output file \"%s\"was not created" %feature_vector_output_fp

    end_time = time()
    elapsed_time = end_time - start_time
    logging.info('Finished feature vector optimization process for \'%s\' '
                 'model' % model)
    logging.info('Total elapsed time (in seconds): %d' % elapsed_time)
