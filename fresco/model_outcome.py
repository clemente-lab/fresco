class ModelOutcome:
    def __init__(self, feature_vector, feature_scores, prediction_quality, predictions):
        self.feature_vector = feature_vector
        self.feature_scores = feature_scores
        self.prediction_quality = prediction_quality
        self.predictions = predictions