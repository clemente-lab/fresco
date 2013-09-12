from sklearn.cross_validation import KFold
from fresco.group_vector_model import build_sample_matrix
from fresco.model_outcome import ModelOutcome
from feature_vector import FeatureVector
import numpy as np

class GroupVectorScorer:
    def score_feature_vector(self, problem_data, feature_vector):
        pass
    
class CrossValidationGroupVectorScorer(GroupVectorScorer):
    def __init__(self, prediction_scoring_function, vector_model, n_cross_folds = 5):
        self.prediction_scoring_function = prediction_scoring_function
        self.vector_model = vector_model
        self.n_cross_folds = n_cross_folds
        
    def score_feature_vector(self, problem_data, feature_vector, masks = None):
        if masks == None:
            masks = [(mask[0], mask[1]) for mask in KFold(len(problem_data.get_response_variables()), n_folds=self.n_cross_folds)]
        feature_vectors = []
        if isinstance(feature_vector, FeatureVector):
            feature_vectors = [feature_vector for mask in masks]
        elif isinstance(feature_vector, list) and len(feature_vector) == len(masks) and all([isinstance(fv, FeatureVector) for fv in feature_vector]):
            feature_vectors = feature_vector
        else:
            raise TypeError("feature_vector is not either a FeatureVector or a list of FeatureVectors of length equal to the number of cross folds.")    
    
        fold_outcomes = []
        for i in range(len(masks)):
            train_mask, test_mask = masks[i]
            fold_outcomes.append(build_model_outcome(problem_data, feature_vectors[i], self.vector_model, self.prediction_scoring_function, train_mask, test_mask))
                    
        feature_vector = fold_outcomes[0].feature_vector
        avg_prediction_quality = sum([outcome.prediction_quality for outcome in fold_outcomes])
        avg_feature_scores = [sum([outcome.feature_scores[i] for outcome in fold_outcomes])/float(len(fold_outcomes)) for i in range(len(outcome.feature_vector.get_record_list()))]
        predictions = np.hstack([outcome.predictions for outcome in fold_outcomes])
        
        return CrossValidationModelOutcome(feature_vector, avg_feature_scores, avg_prediction_quality, predictions, fold_outcomes)

class CrossValidationModelOutcome(ModelOutcome):
    def __init__(self, feature_vector, feature_scores, prediction_quality, predictions, fold_outcome_list):
        ModelOutcome.__init__(self, feature_vector, feature_scores, prediction_quality, predictions)
        self.fold_outcome_list = fold_outcome_list
        
        
        
def build_model_outcome(problem_data, feature_vector, vector_model, prediction_scoring_function, train_mask, test_mask):
    problem_data.push_mask(train_mask)
    vector_model.fit(problem_data, feature_vector)
    problem_data.pop_mask()
    
    feature_scores = vector_model.get_feature_scores()
    
    problem_data.push_mask(test_mask)
    predictions = vector_model.predict(problem_data, feature_vector)
    prediction_quality = prediction_scoring_function(problem_data.get_response_variables(), predictions)
    problem_data.pop_mask()
            
    return ModelOutcome(feature_vector, feature_scores, prediction_quality, predictions)
        