import numpy as np
from fresco.vector_generator import VectorGenerator
from fresco.feature_vector import FeatureVector
from fresco.utils import normalized_scores

class ActionVectorGenerator(VectorGenerator):
    def __init__(self, group_actions, n_generate, problem_data):
        self.group_actions = group_actions
        self.n_generate = n_generate
        self.problem_data = problem_data

    def generate_vectors(self, outcomes):
        vectors = []
        for outcome in outcomes:
            vectors += self.generate_action_vector(outcome)
        return vectors

    def generate_action_vector(self, outcome):
        action_scores = self.generate_action_scores(outcome)
        
        feature_vector = outcome.feature_vector
        
        feature_vectors = []
        for i in range(self.n_generate):
            new_feature_vector = FeatureVector(feature_vector.get_record_list()[:])
            action_selections = self.stochastic_action_selection(action_scores)
            
            for n in range(len(action_scores[0])):
                action = action_selections[n]
                if isinstance(action, MergeAction):
                    assert feature_vector.get_record_list()[n].get_scope() != 0
                if isinstance(action, SplitAction):
                    assert feature_vector.get_record_list()[n].get_scope() != action.problem_data.get_max_scope()
            
            for feature_index in reversed(range(len(action_selections))):
                if action_selections[feature_index] != None:
                    action_selections[feature_index].apply(feature_index, new_feature_vector)
            feature_vectors.append(new_feature_vector)
 
        return feature_vectors
        
    def generate_action_scores(self, outcome):
        def std_dev_dists(value_list):
            value_list = np.array(value_list)
            avg = np.mean(value_list)
            std = np.std(value_list)
            assert std != None, "no deviation in feature scores"
            dists = [(s-avg)/std for s in value_list]
            return dists
        
        feature_vector = outcome.feature_vector

        feature_abundances = [self.problem_data.get_feature_abundance(record.get_scope(), record.get_id())
                              for record in feature_vector.get_record_list()]
        feature_scores = outcome.feature_scores
        
        abundance_deviations = std_dev_dists(feature_abundances)
        score_deviations = std_dev_dists(feature_scores)
        action_scores = [[action.score(abundance_deviations[i], score_deviations[i], feature_vector.get_record_list()[i]) for i in range(len(feature_vector.get_record_list()))]
                          for action in self.group_actions]
        
        return action_scores
        
    def stochastic_action_selection(self, action_scores):
        def sample_from_scores(scores, proportion, exclude_list):
            scores = normalized_scores(scores, exclude_list)

            np.random.seed() #necessary for mutlithreaded code
            n_nonzero = len([score for score in scores if score != 0])
            size = min(int(len(scores) * proportion), n_nonzero)
            sample = np.random.choice(len(scores), size, p=scores, replace=False) if len(scores) > 0 else []
            for s in sample:
                assert scores[s] != 0

            return sample
        
        actions = [None for i in range(len(action_scores[0]))]
        exclude_list = []
        
        #selection actions for priority in the order they were passed in
        for action_index in range(len(self.group_actions)):
            action_sample = sample_from_scores(action_scores[action_index], self.group_actions[action_index].get_proportion(), exclude_list)
            for i in action_sample:
                assert action_scores[action_index][i] != None
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
            new_feature_record = self.problem_data.get_feature_record(*group)
            found = False
            for record in feature_vector.get_record_list():
                if new_feature_record.get_id() == record.get_id() and new_feature_record.get_scope() == record.get_scope():     
                    found = True
                    break   
            if not found:
                feature_vector.get_record_list().append(new_feature_record)
        assert(len(set(feature_vector.get_record_list())) == len(feature_vector.get_record_list()))
        
class MergeAction(GroupAction):
    def score(self, abundance_deviation, score_deviation, feature_record):
        if feature_record.get_scope() == 0:
            return None
        return GroupAction.score(self, abundance_deviation, score_deviation, feature_record)

    def apply(self, feature_index, feature_vector):
        feature_record = feature_vector.pop_feature(feature_index)
        
        new_groups = self.problem_data.get_split_groups(feature_record.get_scope(), feature_record.get_id(), feature_record.get_scope()-1)
        for group in new_groups:
            new_feature_record = self.problem_data.get_feature_record(*group)
            found = False
            for record in feature_vector.get_record_list():
                if new_feature_record.get_id() == record.get_id() and new_feature_record.get_scope() == record.get_scope():     
                    found = True
                    break   
            if not found:
                feature_vector.get_record_list().append(new_feature_record)
        assert(len(set(feature_vector.get_record_list())) == len(feature_vector.get_record_list()))     

        
        
class DeleteAction(GroupAction):
    def score(self, abundance_deviation, score_deviation, feature_record):
        return GroupAction.score(self, abundance_deviation, score_deviation, feature_record)

    def apply(self, feature_index, feature_vector):
        feature_vector.pop_feature(feature_index)
