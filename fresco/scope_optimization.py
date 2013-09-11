from fresco.parallel_processing import multiprocess_functions, ProcessDefinition

class ScopeOptimization():
    def __init__(self, group_vector_scorer, vector_generator, n_processes, n_iterations, n_maintain):
        self.group_vector_scorer = group_vector_scorer
        self.vector_generator = vector_generator
        self.n_processes = n_processes
        self.n_iterations = n_iterations
        self.n_maintain = n_maintain
    
    def optimize_vector(self, initial_feature_vector, problem_data, save_iterations = False):
        outcome_set = [self.group_vector_scorer.score_feature_vector(problem_data, initial_feature_vector)]
        return_set = [outcome_set[0]]
    
        for iteration in range(1, self.n_iterations):
            new_vector_set = self.vector_generator.generate_vectors(outcome_set)
     
            testing_functions = [ProcessDefinition(self.group_vector_scorer.score_feature_vector, positional_arguments=(problem_data, new_vector)) for new_vector in new_vector_set]
            result_handler = outcome_set.append
            multiprocess_functions(testing_functions, result_handler, self.n_processes)
        
            outcome_set.sort(key=lambda outcome:outcome.prediction_quality, reverse=True)
            outcome_set = outcome_set[:self.n_maintain]
            return_set.append(outcome_set[0])

        return return_set if save_iterations else return_set[-1]