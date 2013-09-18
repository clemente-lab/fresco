#!/usr/bin/env python
from __future__ import division
from unittest import TestCase, main

from fresco.action_vector_generator import ActionVectorGenerator, SplitAction, MergeAction, DeleteAction
from fresco.group_problem_data import build_problem_data
from fresco.model_outcome import ModelOutcome

from sys import stderr


class ActionVectorGeneratorTests(TestCase):
    """Tests for functions in the parallel_processing module."""

    def setUp(self):
        """Initialize data used in the tests."""      
        group_map_files = [
                           open("tests/test_files/study_451/ucrc_0.79/uclust_ref_picked_otus/study_451_split_library_seqs_otus.txt"),
                           open("tests/test_files/study_451/ucrc_0.82/uclust_ref_picked_otus/study_451_split_library_seqs_otus.txt"),
                           open("tests/test_files/study_451/ucrc_0.85/uclust_ref_picked_otus/study_451_split_library_seqs_otus.txt")
                           ]
        mapping_file = open("tests/test_files/study_451_mapping_file.txt")
        prediction_field = "TREATMENT"
        start_level = 1
        include_only = None
        negate = False
        n_processes = 1
        self.problem_data, self.initial_feature_vector = build_problem_data(group_map_files, mapping_file, prediction_field, start_level, include_only, negate, n_processes)
        
        self.split_proportion = 0.25
        self.split_abun_coef = 0.5
        self.split_score_coef = 0.5
        self.merge_proportion = 0.25
        self.merge_abun_coef = 0.5
        self.merge_score_coef = 0.5
        self.delete_proportion = 0.5
        self.delete_abun_coef = 0.5
        self.delete_score_coef = 0.5
        
        self.group_actions = [
                              SplitAction(self.problem_data, self.split_proportion, self.split_abun_coef, self.split_score_coef),
                              MergeAction(self.problem_data, self.merge_proportion, self.merge_abun_coef, self.merge_score_coef),
                              DeleteAction(self.problem_data, self.delete_proportion, self.delete_abun_coef, self.delete_score_coef)
                              ]
        self.n_generate = 2
        self.action_vector_generator = ActionVectorGenerator(self.group_actions, self.n_generate, self.problem_data)

        
    def test_generate_action_scores(self):
        """Test the score generation process."""
        
        n_iterations = 10
        feature_vector = self.initial_feature_vector
        max_length = 10000
        
        for i in range(n_iterations):
            if len(feature_vector.rec_list) > max_length:
                feature_vector.rec_list = feature_vector.rec_list[-max_length:]
            outcome = ModelOutcome(feature_vector, [r for r in range(len(feature_vector.get_record_list()))], prediction_quality=0, predictions=[])
            action_scores = self.action_vector_generator.generate_action_scores(outcome)
            
            assert len(action_scores) == len(self.group_actions)
            for action_index in range(len(action_scores)):
                scores = action_scores[action_index]
                assert len(scores) == len(feature_vector.get_record_list())
                action = self.group_actions[action_index]
                if isinstance(action, MergeAction):
                    for feature_index in range(len(scores)):
                        if scores[feature_index] == None:
                            assert feature_vector.get_record_list()[feature_index].get_scope() == 0
                        else:
                            assert feature_vector.get_record_list()[feature_index].get_scope() != 0
                elif isinstance(action, SplitAction):
                    for feature_index in range(len(scores)):
                        if scores[feature_index] == None:
                            assert feature_vector.get_record_list()[feature_index].get_scope() == self.problem_data.get_max_scope()
                        else:
                            assert feature_vector.get_record_list()[feature_index].get_scope() != self.problem_data.get_max_scope()

            feature_vector = self.action_vector_generator.generate_action_vector(outcome)[0]
