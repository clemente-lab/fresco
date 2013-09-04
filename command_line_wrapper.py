import scope_optimization
import utils
import parse_input_files
from feature_vector import FeatureRecord, FeatureVector
from group_problem_data import ProblemData
from group_vector_model import GroupVectorModel
from score_group_vector import CrossValidationGroupVectorScorer
from action_vector_generator import ActionVectorGenerator, SplitAction, MergeAction, DeleteAction
import write_results
import parallel_processing
from sklearn.cross_validation import KFold


def command_line_argument_wrapper(model, n_iterations, group_map_files, start_level, mapping_file, prediction_field, include_only,
                                  n_maintain, n_generate, score_predictions_function, split_abun_coef, split_score_coef, merge_abun_coef,
                                  merge_score_coef, delete_abun_coef, delete_score_coef,
                                  split_proportion, merge_proportion, delete_proportion, n_cross_folds,
                                  n_processes, feature_vector_output, prediction_testing_output, n_trials):
    vector_model = GroupVectorModel(utils.parse_model_string(model))
    group_vector_scorer = CrossValidationGroupVectorScorer(score_predictions_function, vector_model, n_cross_folds)
    problem_data, initial_feature_vector = build_problem_data(group_map_files, mapping_file, prediction_field, start_level, include_only, n_processes)
    group_actions = [SplitAction(problem_data, split_proportion, split_abun_coef, split_score_coef),
                     MergeAction(problem_data, merge_proportion, merge_abun_coef, merge_score_coef),
                     DeleteAction(problem_data, delete_proportion, delete_abun_coef, delete_score_coef)]
    vector_generator = ActionVectorGenerator(group_actions)

    if n_trials > 0:
        xfold_feature_vectors = [[] for i in range(n_iterations)]
        masks = [(train, test) for train, test in KFold(problem_data.get_n_samples(), n_folds=n_trials, indices=False)]
        for train_mask, test_mask in masks:
            problem_data.set_mask(train_mask)
            iteration_outcomes = scope_optimization.scope_optimization(initial_feature_vector, problem_data, group_vector_scorer, vector_generator, n_iterations, n_processes, n_maintain, n_generate, True)
            for iteration in range(len(iteration_outcomes)):
                xfold_feature_vectors[iteration].append(iteration_outcomes[iteration].feature_vector)
        functions = []
        mask_results = []
        for iteration in range(len(iteration_outcomes)):
            for mask_index in range(len(masks)):
                functions.append( (mask_testing, (problem_data, masks[mask_index], vector_model, score_predictions_function, xfold_feature_vectors[iteration][mask_index], iteration)) )
        parallel_processing.multiprocess_functions(functions, mask_results.append, n_processes)
        test_outcomes = [[] for i in range(n_iterations)]
        for iteration, mask_result in mask_results:
            test_outcomes[iteration].append(mask_result)

        write_results.write_to_file(write_results.testing_output_lines(test_outcomes), prediction_testing_output)
    else:
        outcome = scope_optimization.scope_optimization(initial_feature_vector, problem_data, group_vector_scorer, vector_generator, n_iterations, n_processes, n_maintain, n_generate, False)
        write_results.write_to_file(write_results.feature_output_lines(outcome), feature_vector_output)
    

def mask_testing(problem_data, masks, vector_model, score_predictions_function, feature_vector, ordering_tag=None):
    train_mask, test_mask = masks
    problem_data.set_mask(train_mask)
    vector_model.fit(problem_data, feature_vector)
    problem_data.set_mask(test_mask)
    test_predictions = vector_model.predict(problem_data, feature_vector)
    test_quality_score = score_predictions_function(problem_data.get_response_variables(), test_predictions)
    if ordering_tag != None:
        return (ordering_tag, test_quality_score)
    else:
        return test_quality_score
   
def build_problem_data(group_map_files, mapping_file, prediction_field, start_level, include_only, n_processes):
        #For each scope, build a map from group to object and vice versa
    group_to_object = []
    object_to_group = []
    for map_file in group_map_files:
        g_to_o, o_to_g = parse_input_files.read_split_file(map_file)
        group_to_object.append(g_to_o)
        object_to_group.append(o_to_g)

    #Find a list of sample names from our group names
    #An alternative is 'samplenames = samplemap.keys()', but that may have records without features
    samplenames = set()
    for grp in group_to_object[start_level]:
        l = group_to_object[start_level][grp]
        for obj in l:
            samplenames.add(utils.parse_object_string_sample(obj))
    samplenames = list(samplenames)

    #get a map of sample name to it's properties
    samplemap = parse_input_files.read_mapping_file(mapping_file)
    sample_to_response = dict([(samplename, samplemap[samplename][prediction_field]) for samplename in samplenames
              if include_only == None or all([samplemap[samplename][include_only[i][0]] == [include_only[i][1]] for i in range(len(include_only))])])
    
    problem_data = ProblemData(group_to_object, object_to_group, sample_to_response, n_processes)

    feature_vector = FeatureVector([FeatureRecord(group, start_level,
                                                  len(group_to_object[start_level][group]))
                                    for group in group_to_object[start_level].keys()])
    
    return problem_data, feature_vector
