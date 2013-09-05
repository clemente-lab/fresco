from feature_vector import FeatureRecord
import numpy as np
from utils import parse_object_string_sample
import parallel_processing

class ProblemData:
	def __init__(self, group_to_object, object_to_group, sample_to_response, n_processes, parse_object_sample=parse_object_string_sample, mask=None):
		self.response_variables = np.array([sample_to_response[sample] for sample in sample_to_response.keys()])
		self.max_scope = len(object_to_group)-1
		self.mask = mask
		self.n_samples = len(sample_to_response)
		
		sample_indeces = dict([(sample_to_response.keys()[index], index) for index 
							in range(len(sample_to_response.keys()))])
		
		self.feature_columns = dict()
		self.feature_records = dict()
		self.scope_map = dict()
		
		def build_scope_columns_records_splits(scope, group_to_object, sample_indeces):
			feature_records = dict()
			feature_columns = dict()
			scope_map = dict()
			for group in group_to_object[scope]:
				objects = group_to_object[scope][group]
			
				feature_column = np.zeros((len(sample_indeces,)))
				for obj in objects:
					sample_index = sample_indeces[parse_object_sample(obj)]
					feature_column[sample_index] += 1
				feature_columns[(scope, group)] = feature_column
			
				feature_abundance = len(objects)
				feature_id = group
				feature_records[(scope, group)] = FeatureRecord(feature_id, scope, feature_abundance)
			
				for final_scope in range(len(group_to_object)):
					if final_scope == scope:
						self.scope_map[(scope, group, final_scope)] = [(scope, group)]
						continue
					
					final_groups = list(set(object_to_group[final_scope][obj] for obj in objects if obj in object_to_group[final_scope]))
					scope_map[(scope, group, final_scope)] = [(final_scope, final_group) for final_group in final_groups]
			
			return (feature_columns, feature_records, scope_map)
		
		functions = []
		results = []
		for scope in range(len(group_to_object)):
			functions.append( (build_scope_columns_records_splits, (scope, group_to_object, sample_indeces)) )
		parallel_processing.multiprocess_functions(functions, results.append, n_processes)
		
		for feature_columns, feature_records, scope_map in results:
			self.feature_columns.update(feature_columns)	
			self.feature_records.update(feature_records)
			self.scope_map.update(scope_map)
	
	def set_mask(self, mask):
		self.mask = mask
		
		#since the column vectors have changed, we need to recalculate abundances
		for scope, group in self.feature_records.keys():
			record = self.get_feature_record(scope, group)
			feature_abundance = sum(self.get_feature_column(scope, group))
			record.abundance = feature_abundance
		
		
	def get_n_samples(self):
		return self.n_samples
	
	def get_max_scope(self):
		return self.max_scope
	
	def get_feature_column(self, scope, group):
		if self.mask != None:
			return self.feature_columns[(scope, group)][self.mask]
		else:
			return self.feature_columns[(scope, group)]
	
	def get_feature_record(self, scope, group):
		return self.feature_records[(scope, group)]
	
	def get_split_groups(self, initial_scope, group, final_scope):
		try:
			return self.scope_map[(initial_scope, group, final_scope)]
		except:
			return []
	
	def get_response_variables(self):
		if self.mask != None:
			return self.response_variables[self.mask]
		else:
			return self.response_variables