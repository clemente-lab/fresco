#!/usr/bin/env python
from __future__ import division
from unittest import TestCase, main
from fresco.feature_vector import FeatureVector
from fresco.action_vector_generator import ActionVectorGenerator, SplitAction, MergeAction, DeleteAction
from fresco.group_problem_data import build_problem_data
from fresco.model_outcome import ModelOutcome


class ActionVectorGeneratorTests(TestCase):
    """Tests for functions in the action_vector_generator module."""

    def setUp(self):
        """Initialize data used in the tests."""      
        prediction_field = "MEAL_TYPE"
        start_level = 1
        include_only = None
        negate = False
        n_processes = 1
        self.problem_data = build_problem_data(group_map_lines, mapping_lines, prediction_field, start_level, include_only, negate, n_processes, parse_object_string=parse_object_string)
        
        initial_feature_ids = self.problem_data.get_group_ids(start_level)
        self.initial_feature_vector = FeatureVector([self.problem_data.get_feature_record(start_level, id) for id in initial_feature_ids])
    
        self.split_proportion = 0.5
        self.split_abun_coef = 0
        self.split_score_coef = -1
        self.merge_proportion = 0.5
        self.merge_abun_coef = -1
        self.merge_score_coef = 0
        self.delete_proportion = 0.0
        self.delete_abun_coef = -1
        self.delete_score_coef = -1
        
        self.group_actions = [
                              SplitAction(self.problem_data, self.split_proportion, self.split_abun_coef, self.split_score_coef),
                              MergeAction(self.problem_data, self.merge_proportion, self.merge_abun_coef, self.merge_score_coef),
                              DeleteAction(self.problem_data, self.delete_proportion, self.delete_abun_coef, self.delete_score_coef)
                              ]
        self.n_generate = 2
        self.action_vector_generator = ActionVectorGenerator(self.group_actions, self.n_generate, self.problem_data)

        
    def test_generate_action_scores(self):
        """
        Test the score generation process.
        """
        
        n_iterations = 10
        feature_vector = self.initial_feature_vector
        max_length = 10000
        
        for i in range(n_iterations):
            #Generate action scores for each group in a feature vector
            if len(feature_vector.rec_list) > max_length:
                feature_vector.rec_list = feature_vector.rec_list[-max_length:]
            outcome = ModelOutcome(feature_vector, [r for r in range(len(feature_vector.get_record_list()))], prediction_quality=0, predictions=[])
            action_scores = self.action_vector_generator.generate_action_scores(outcome)
            
            #Test required invariants over the returned score values, like None for inapplicable actions
            self.assertEquals(len(action_scores), len(self.group_actions))
            for action_index in range(len(action_scores)):
                scores = action_scores[action_index]
                self.assertEquals(len(scores), len(feature_vector.get_record_list()))
                action = self.group_actions[action_index]
                if isinstance(action, MergeAction):
                    for feature_index in range(len(scores)):
                        if scores[feature_index] == None:
                            self.assertEquals(feature_vector.get_record_list()[feature_index].get_scope(), 0)
                        else:
                            self.assertNotEquals(feature_vector.get_record_list()[feature_index].get_scope(), 0)
                elif isinstance(action, SplitAction):
                    for feature_index in range(len(scores)):
                        if scores[feature_index] == None:
                            self.assertEquals(feature_vector.get_record_list()[feature_index].get_scope(), self.problem_data.get_max_scope())
                        else:
                            self.assertNotEquals(feature_vector.get_record_list()[feature_index].get_scope(), self.problem_data.get_max_scope())

            #Have the vector generator create a new vector to test
            feature_vector = self.action_vector_generator.generate_action_vector(outcome)[0]

def parse_object_string(object_string):
    try:
        underscore_index = object_string.rindex('_')
    except ValueError:
        raise ValueError("Could not find an underscore separating the sample "
                         "name from the object identifier.")
    else:
        return object_string[underscore_index+1:]


mapping_lines = (
                 "#SAMPLE_ID\tMEAL_TYPE\tTASTINESS\tPRICE",
                 "DINNER1\tDINNER\t10\t8.00",
                 "LUNCH2\tLUNCH\t8\t6.79",
                 "DINNER2\tDINNER\t3\t12.30",
                 "LUNCH1\tLUNCH\t5\t6.00"
                 )

group_map_lines = [(
                    "SIDE\tAPPLE1_DINNER1\tLEECHEE1_DINNER2\tAPPLE1_LUNCH1\tSALAD_DINNER2\tRICE_DINNER1",
                    "MAIN\tSTEAK1_DINNER2\tTURKEY1_LUNCH1\tCHICKEN1_LUNCH2\tSALMON1_DINNER1\tPOTATO_DINNER1",
                    "DRINK\tSODA1_LUNCH1\tSODA2_LUNCH2\tWINE_DINNER2\tBEER_DINNER1"
                    ),
                   (
                    "SIDE_FRUIT\tAPPLE1_DINNER1\tLEECHEE1_DINNER2\tAPPLE1_LUNCH1",
                    "SIDE_OTHER\tSALAD_DINNER2\tRICE_DINNER1",
                    "MAIN_MEAT\tSTEAK1_DINNER2\tTURKEY1_LUNCH1\tCHICKEN1_LUNCH2\tSALMON1_DINNER1",
                    "MAIN_OTHER\tPOTATO_DINNER1",
                    "DRINK_ALCOHOLIC\tWINE_DINNER2\tBEER_DINNER1",
                    "DRINK_NONALCOHOLIC\tSODA1_LUNCH1\tSODA2_LUNCH2"
                    )] 