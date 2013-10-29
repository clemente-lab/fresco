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
            #Note: Vector generation not currently parallel
            new_vector_set = self.vector_generator.generate_vectors(outcome_set)
     
            #Note: The entire problem_data object does not have to be copied to test a vector, only the column vector.
            outcome_set += self.group_vector_scorer.score_feature_vector_set(problem_data, new_vector_set, self.n_processes)
   
            outcome_set.sort(key=lambda outcome:outcome.prediction_quality, reverse=True)

            outcome_set = outcome_set[:self.n_maintain]
            return_set.append(outcome_set[0])

        return return_set if save_iterations else return_set[-1]
