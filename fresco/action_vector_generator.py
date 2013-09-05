import numpy as np
from fresco.vector_generator import VectorGenerator
from fresco.feature_vector import FeatureVector

class ActionVectorGenerator(VectorGenerator):
    def __init__(self, group_actions):
        self.group_actions = group_actions

    def generate_vectors(self, outcomes, n_generate):
        vectors = []
        for outcome in outcomes:
            vectors += self.generate_action_vector(outcome, n_generate)
        return vectors

    def generate_action_vector(self, outcome, n_generate):
        def std_dev_dists(value_list):
            value_list = np.array(value_list)
            avg = np.mean(value_list)
            std = np.std(value_list)
        
            dists = [(s-avg)/std for s in value_list]
            return dists
        
        feature_vector = outcome.feature_vector

        feature_abundances = [record.get_abundance() for record in feature_vector.get_record_list()]
        feature_scores = outcome.feature_scores
        
        abundance_deviations = std_dev_dists(feature_abundances)
        score_deviations = std_dev_dists(feature_scores)
        action_scores = [[action.score(abundance_deviations[i], score_deviations[i], feature_vector.get_record_list()[i]) for i in range(len(feature_vector.get_record_list()))]
                          for action in self.group_actions]
        
        feature_vectors = []
        for i in range(n_generate):
            new_feature_vector = FeatureVector(feature_vector.get_record_list()[:])
            action_selections = self.stochastic_action_selection(action_scores)
            for feature_index in reversed(range(len(action_selections))):
                if action_selections[feature_index] != None:
                    action_selections[feature_index].apply(feature_index, new_feature_vector)
            feature_vectors.append(new_feature_vector)
 
        return feature_vectors
        
    def stochastic_action_selection(self, action_scores):
        def normalize_scores(scores, exclude_list):
            invalid_list = []
            for i in range(len(scores)):
                if scores[i] == None:
                    scores[i] = 0
                    invalid_list.append(i)
            
            mini = min(scores)
            if float(mini) < 0:
                scores -= mini

            for e in exclude_list:
                scores[e] = 0
            for i in invalid_list:
                scores[i] = 0
            
            sumi = sum(scores)
            if float(sumi) != 0:
                scores /= sumi
                return scores
            else: #since all elements are non-negative, all elements must be 0
                return []

        def sample_from_scores(scores, proportion, exclude_list):
            scores = normalize_scores(scores, exclude_list)
                
            np.random.seed() #necessary for mutlithreaded code
            sample = np.random.choice(len(scores), int(len(scores)*proportion), p=scores, replace=False) if len(scores) > 0 else []
            return sample
        
        actions = [None for i in range(len(action_scores[0]))]
        exclude_list = []
        
        #selection actions for priority in the order they were passed in
        for action_index in range(len(self.group_actions)):
            action_sample = sample_from_scores(action_scores[action_index], self.group_actions[action_index].get_proportion(), exclude_list)
            for i in action_sample:
                actions[i] = self.group_actions[action_index]
            exclude_list += list(action_sample)
       
        return actions
    
    
class GroupAction:
    def __init__(self, problem_data, proportion, abundance_coef, score_coef):
        self.problem_data = problem_data
        self.abundance_coef = abundance_coef
        self.score_coef = score_coef
        self.proportion = proportion

    def score(self, abundance_deviation, score_deviation, feature_record):
        return self.abundance_coef*abundance_deviation + self.score_coef*score_deviation
    
    def apply(self, feature_index, feature_vector):
        pass
    
    def get_proportion(self):
        return self.proportion

class SplitAction(GroupAction):
    def score(self, abundance_deviation, score_deviation, feature_record):
        if feature_record.get_scope() == self.problem_data.get_max_scope():
            return None
        return GroupAction.score(self, abundance_deviation, score_deviation, feature_record)

    def apply(self, feature_index, feature_vector):
        feature_record = feature_vector.pop_feature(feature_index)
        new_groups = self.problem_data.get_split_groups(feature_record.get_scope(), feature_record.get_id(), feature_record.get_scope()+1)
        for group in new_groups:
            feature_vector.get_record_list().append(self.problem_data.get_feature_record(*group))
        
class MergeAction(GroupAction):
    def score(self, abundance_deviation, score_deviation, feature_record):
        if feature_record.get_scope() == 0:
            return None
        return GroupAction.score(self, abundance_deviation, score_deviation, feature_record)

    def apply(self, feature_index, feature_vector):
        feature_record = feature_vector.pop_feature(feature_index)
        
        new_groups = self.problem_data.get_split_groups(feature_record.get_scope(), feature_record.get_id(), feature_record.get_scope()-1)
        for group in new_groups:
            feature_vector.get_record_list().append(self.problem_data.get_feature_record(*group))
        
        
class DeleteAction(GroupAction):
    def score(self, abundance_deviation, score_deviation, feature_record):
        return GroupAction.score(self, abundance_deviation, score_deviation, feature_record)

    def apply(self, feature_index, feature_vector):
        feature_vector.pop_feature(feature_index)
