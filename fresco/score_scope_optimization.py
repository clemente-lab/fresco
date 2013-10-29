from sklearn.cross_validation import KFold
from fresco.parallel_processing import multiprocess_functions, ProcessDefinition
from score_group_vector import build_model_outcome_with_matrices
from group_vector_model import build_sample_matrix

def scope_optimization_cross_validation(scope_optimization, initial_feature_vector, problem_data, vector_model, prediction_scoring_function, n_cross_folds, n_processes):
    masks = [(train, test) for train, test in KFold(problem_data.get_n_unmasked_samples(), n_folds=n_cross_folds)]
    
    #Perform the optimization process for each fold.
    n_iterations = scope_optimization.n_iterations
    training_outcomes = [[] for i in range(n_iterations)]
    for train_mask, test_mask in masks:
        problem_data.push_mask(test_mask)
        iteration_outcomes = scope_optimization.optimize_vector(initial_feature_vector, problem_data, True)
        problem_data.pop_mask()
        for iteration in range(n_iterations):
            training_outcomes[iteration].append(iteration_outcomes[iteration])
    
    #Score the resulting feature vectors for each fold in parallel.
    process_definitions = []
    for iteration in range(n_iterations):
        for mask_index in range(len(masks)):
            X = build_sample_matrix(problem_data, training_outcomes[iteration][mask_index].feature_vector)
            y = problem_data.get_response_variables()
            train_mask, test_mask = masks[mask_index]
            
            process_definitions.append(ProcessDefinition(build_model_outcome_with_matrices, positional_arguments=
                                                         (training_outcomes[iteration][mask_index].feature_vector, vector_model,
                                                          prediction_scoring_function, X[train_mask], y[train_mask], X[test_mask], y[test_mask]),
                                                         tag=(iteration, mask_index)))
    
    test_results = []
    multiprocess_functions(process_definitions, test_results.append, n_processes)

    test_outcomes = [[None] * len(masks) for i in range(scope_optimization.n_iterations)]
    for tag, mask_result in test_results:
        iteration, mask_index = tag
        test_outcomes[iteration][mask_index] = mask_result

    return test_outcomes
