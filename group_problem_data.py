from feature_vector import FeatureRecord
import numpy as np
from utils import parse_object_string_sample

class ProblemData:
	def __init__(self, group_to_object, object_to_group, sample_to_response, parse_object_sample=parse_object_string_sample):
		self.response_variables = np.array([sample_to_response[sample] for sample in sample_to_response.keys()])
		self.max_scope = len(object_to_group)-1
		
		sample_indeces = dict([(sample_to_response.keys()[index], index) for index 
							in range(len(sample_to_response.keys()))])
		
		self.feature_columns = dict()
		self.feature_records = dict()
		for scope in range(len(group_to_object)):
			for group in group_to_object[scope]:
				objects = group_to_object[scope][group]
			
				feature_abundance = len(objects)
				feature_id = group
				self.feature_records[(scope, group)] = FeatureRecord(feature_id, scope, feature_abundance)
				
				feature_column = np.zeros((len(sample_indeces,)))
				for obj in objects:
					sample_index = sample_indeces[parse_object_sample(obj)]
					feature_column[sample_index] += 1
				self.feature_columns[(scope, group)] = feature_column
			
		self.scope_map = dict()
		for init_scope in range(len(group_to_object)):
			for group in group_to_object[init_scope]:
				objects = group_to_object[init_scope][group]
				for final_scope in range(len(group_to_object)):
					if final_scope == init_scope:
						self.scope_map[(init_scope, group, final_scope)] = [(init_scope, group)]
						continue
					
					final_groups = list(set(object_to_group[final_scope][obj] for obj in objects if obj in object_to_group[final_scope]))
					self.scope_map[(init_scope, group, final_scope)] = [(final_scope, final_group) for final_group in final_groups]
	
	def get_max_scope(self):
		return self.max_scope
	
	def get_feature_column(self, scope, group):
		return self.feature_columns[(scope, group)]
	
	def get_feature_record(self, scope, group):
		return self.feature_records[(scope, group)]
	
	def get_split_groups(self, initial_scope, group, final_scope):
		try:
			return self.scope_map[(initial_scope, group, final_scope)]
		except:
			return []
	
	def get_response_variables(self):
		return self.response_variables