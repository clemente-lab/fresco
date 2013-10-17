#!/usr/bin/env python
from __future__ import division
from unittest import TestCase, main
from fresco.group_problem_data import build_problem_data, ProblemData


class GroupProblemDataTests(TestCase):
    """Tests for functions in the action_vector_generator module."""

    def setUp(self):
        """Initialize data used in the tests."""      
        self.prediction_field = "MEAL_TYPE"
        self.start_level = 1
        self.include_only = None
        self.negate = False
        self.n_processes = 1
        
        self.group_to_object = group_to_object
        self.object_to_group = [dict()] * len(self.group_to_object)
        self.sample_to_response = sample_to_response
        for scope in range(len(group_to_object)):
            for group in group_to_object[scope]:
                for obj in group_to_object[scope][group]:
                    self.object_to_group[scope][obj] = group
        
    def test_get_group_ids(self):
        """
        Test building single-scope feature vectors
        """
        problem_data = build_problem_data(group_map_lines, mapping_lines,
                                               self.prediction_field, self.start_level,
                                               self.include_only, self.negate, self.n_processes, 
                                               parse_object_string=parse_object_string)
       
        initial_feature_ids = problem_data.get_group_ids(self.start_level)
        
        self.assertEqual(set(initial_feature_ids), 
                         set(self.group_to_object[self.start_level].keys()))
        
        
    def test_build_problem_data(self):
        """
        Test the wrapper for the constructor.
        """
        problem_data = build_problem_data(group_map_lines, mapping_lines,
                                          self.prediction_field, self.start_level,
                                          self.include_only, self.negate, self.n_processes, 
                                          parse_object_string=parse_object_string)
        self._test_problem_data_object(problem_data)
        
    def _test_problem_data_object(self, problem_data):
        """
        Helper function to test validity of a problem_data object
        """
        group_records = problem_data.group_records
        
        #make sure we're not missing any scopes
        self.assertEqual(len(group_records), len(self.group_to_object))
        self.assertEqual(len(group_records), problem_data.n_scopes)
        self.assertEqual(len(group_records) - 1, problem_data.get_max_scope())
        
        for scope in range(len(self.group_to_object)):
            for group in self.group_to_object[scope]:
                #make sure the problem_data isn't missing any groups
                self.assertTrue(group in group_records[scope].keys())
                #make sure the problem_data isn't missing any objects
                self.assertEqual(len(self.group_to_object[scope][group]), problem_data.get_feature_abundance(scope, group))
                #make sure the feature record is right
                self.assertEqual(group_records[scope][group].feature_record.get_id(), group)
                self.assertEqual(group_records[scope][group].feature_record.get_scope(), scope)
            
                #TODO when it's not 1:00 AM: test scope_map for each group_record
        
        #make sure we haven't lost any samples
        self.assertEqual(problem_data.get_n_unmasked_samples(), len(self.sample_to_response))
        
        #make sure the response variables are correct
        y = problem_data.get_response_variables()
        for s in range(problem_data.get_n_unmasked_samples()):
            self.assertEqual(y[s], self.sample_to_response[self.sample_to_response.keys()[s]])
        
    def test_init(self):
        """
        Test the constructor.
        """
        problem_data = ProblemData(self.group_to_object, self.object_to_group, self.sample_to_response, self.n_processes, parse_object_string)
        self._test_problem_data_object(problem_data)
        
        
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

sample_to_response = {"DINNER1":"DINNER",
                      "LUNCH2":"LUNCH",
                      "DINNER2":"DINNER",
                      "LUNCH1":"LUNCH"}

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

group_to_object = [
                   {"SIDE":["APPLE1_DINNER1", "LEECHEE1_DINNER2", "APPLE1_LUNCH1", "SALAD_DINNER2", "RICE_DINNER1"],
                    "MAIN":["STEAK1_DINNER2", "TURKEY1_LUNCH1", "CHICKEN1_LUNCH2", "SALMON1_DINNER1", "POTATO_DINNER1"],
                    "DRINK":["SODA1_LUNCH1", "SODA2_LUNCH2", "WINE_DINNER2", "BEER_DINNER1"]},
                   {"SIDE_FRUIT": ["APPLE1_DINNER1", "LEECHEE1_DINNER2", "APPLE1_LUNCH1"],
                    "SIDE_OTHER": ["SALAD_DINNER2", "RICE_DINNER1"],
                    "MAIN_MEAT": ["STEAK1_DINNER2", "TURKEY1_LUNCH1", "CHICKEN1_LUNCH2", "SALMON1_DINNER1"],
                    "MAIN_OTHER": ["POTATO_DINNER1"],
                    "DRINK_ALCOHOLIC": ["WINE_DINNER2", "BEER_DINNER1"],
                    "DRINK_NONALCOHOLIC": ["SODA1_LUNCH1", "SODA2_LUNCH2"]}
                   ]