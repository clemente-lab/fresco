from sklearn.cross_validation import KFold
from fresco.model_outcome import ModelOutcome
from parallel_processing import ProcessDefinition, multiprocess_functions
from group_vector_model import build_sample_matrix
import numpy as np

class GroupVectorScorer:
    def score_feature_vector(self, problem_data, feature_vector):
        pass
    
class CrossValidationGroupVectorScorer(GroupVectorScorer):
    def __init__(self, prediction_scoring_function, vector_model, n_cross_folds = 5):
        self.prediction_scoring_function = prediction_scoring_function
        self.vector_model = vector_model
        self.n_cross_folds = n_cross_folds

    def score_feature_vector_set(self, problem_data, feature_vector_set, n_processes):
        outcome_set = []
        testing_functions = []

        for new_vector in feature_vector_set:
            X = build_sample_matrix(problem_data, new_vector) 
            y = problem_data.get_response_variables()
            testing_functions.append(ProcessDefinition(self.score_feature_vector, positional_arguments=(None, new_vector, X, y)))

        result_handler = outcome_set.append
        multiprocess_functions(testing_functions, result_handler, n_processes)
        return outcome_set

    def score_feature_vector(self, problem_data, feature_vector, X=None, y=None):        
        n_samples = len(y) if y != None else len(problem_data.get_response_variables())
        masks = [(mask[0], mask[1]) for mask in KFold(n_samples, n_folds=self.n_cross_folds)]

        fold_outcomes = []
        for i in range(len(masks)):
            #We use this order because we want exclusion masks, not inclusion masks
            train_mask, test_mask = masks[i]

            if problem_data != None:
                fold_outcomes.append(build_model_outcome(problem_data, feature_vector, self.vector_model, self.prediction_scoring_function, train_mask, test_mask))
            elif X != None and y != None:
                fold_outcomes.append(build_model_outcome_with_matrices(feature_vector, self.vector_model, self.prediction_scoring_function, X[train_mask], y[train_mask], X[test_mask], y[test_mask]))
            else:
                raise TypeError('Neither problem_data nor X and y were provided to the scoring process.')
                
        feature_vector = fold_outcomes[0].feature_vector
        avg_prediction_quality = sum([outcome.prediction_quality for outcome in fold_outcomes])/float(self.n_cross_folds)
        avg_feature_scores = [sum([outcome.feature_scores[i] for outcome in fold_outcomes])/float(len(fold_outcomes)) for i in range(len(outcome.feature_vector.get_record_list()))]
        predictions = np.hstack([outcome.predictions for outcome in fold_outcomes])
        
        return CrossValidationModelOutcome(feature_vector, avg_feature_scores, avg_prediction_quality, predictions, fold_outcomes)

class CrossValidationModelOutcome(ModelOutcome):
    def __init__(self, feature_vector, feature_scores, prediction_quality, predictions, fold_outcome_list):
        ModelOutcome.__init__(self, feature_vector, feature_scores, prediction_quality, predictions)
        self.fold_outcome_list = fold_outcome_list
        
def build_model_outcome(problem_data, feature_vector, vector_model, prediction_scoring_function, train_mask, test_mask):
    problem_data.push_mask(test_mask)
    vector_model.fit(problem_data, feature_vector)
    problem_data.pop_mask()
    
    feature_scores = vector_model.get_feature_scores()
    
    problem_data.push_mask(train_mask)
    predictions = vector_model.predict(problem_data, feature_vector)
    prediction_quality = prediction_scoring_function(problem_data.get_response_variables(), predictions)
    problem_data.pop_mask()
                      
    return ModelOutcome(feature_vector, feature_scores, prediction_quality, predictions)
    
def build_model_outcome_with_matrices(feature_vector, vector_model, prediction_scoring_function, X_train, y_train, X_test, y_test):
    vector_model.fit(None, feature_vector, X_train, y_train)
    
    feature_scores = vector_model.get_feature_scores()
    
    predictions = vector_model.predict(None, feature_vector, X_test)
    prediction_quality = prediction_scoring_function(y_test, predictions)
            
    return ModelOutcome(feature_vector, feature_scores, prediction_quality, predictions)
