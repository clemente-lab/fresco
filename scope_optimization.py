import parallel_processing

def scope_optimization(initial_feature_vector, problem_data, group_vector_scorer, vector_generator, n_iterations, n_processes, n_maintain, n_generate, save_iterations):
    outcome_set = [group_vector_scorer.score_feature_vector(problem_data, initial_feature_vector)]
    return_set = [outcome_set[0]]
    
    for iteration in range(1, n_iterations):
        new_vector_set = vector_generator.generate_vectors(outcome_set, n_generate)
     
        testing_functions = [(group_vector_scorer.score_feature_vector, (problem_data, new_vector)) for new_vector in new_vector_set]
        result_handler = outcome_set.append
        parallel_processing.multiprocess_functions(testing_functions, result_handler, n_processes)
        
        outcome_set.sort(key=lambda outcome:outcome.prediction_quality, reverse=True)
        outcome_set = outcome_set[:n_maintain]
        return_set.append(outcome_set[0])
       
    return return_set if save_iterations else return_set[-1][1]
        
    
