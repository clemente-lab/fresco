import group_vector_model
from sklearn.cross_validation import KFold
import model_outcome

class GroupVectorScorer:
    def score_feature_vector(self, problem_data, feature_vector):
        pass
    
class CrossValidationGroupVectorScorer(GroupVectorScorer):
    def __init__(self, prediction_scoring_function, vector_model, n_cross_folds):
        self.prediction_scoring_function = prediction_scoring_function
        self.vector_model = vector_model
        self.n_cross_folds = n_cross_folds
        
    def score_feature_vector(self, problem_data, feature_vector):
        X = group_vector_model.build_sample_matrix(problem_data, feature_vector)
    
        masks = KFold(len(problem_data.get_response_variables()), n_folds=self.n_cross_folds, indices=False)
    
        predictions = []
        feature_score_lists = []
        for mask in masks:
            self.vector_model.fit(problem_data, feature_vector, X=X, mask=mask[0])
            predictions += self.vector_model.predict(problem_data, feature_vector, X=X, mask=mask[1])
            feature_score_lists.append(self.vector_model.get_feature_scores())
         
        prediction_quality = self.prediction_scoring_function(problem_data.get_response_variables(), predictions)
        feature_scores = sum(feature_score_lists)/len(feature_score_lists)
          
        return model_outcome.ModelOutcome(feature_vector, feature_scores, prediction_quality, predictions)