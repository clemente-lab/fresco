import numpy as np
import types
from fresco.utils import InputTypeError, check_input_type, parse_object_string_sample
from fresco.feature_vector import FeatureRecord
from fresco.parallel_processing import multiprocess_functions, ProcessDefinition
from fresco.parse_input_files import read_split_file, read_mapping_file
from feature_vector import FeatureVector
import inspect

class ProblemData:
    def __init__(self, group_to_object, object_to_group, sample_to_response, n_processes, parse_object_sample=parse_object_string_sample):
        if not isinstance(group_to_object, types.ListType):
            raise InputTypeError('group_to_object should be a list type')
        if not isinstance(object_to_group, types.ListType):
            raise InputTypeError('object_to_group should be a list type')
        if not len(object_to_group) == len(group_to_object):
            raise InputTypeError('object_to_group and group_to_object should be the same length')
        if not all([isinstance(o_to_g, types.DictType) for o_to_g in object_to_group]):
            raise InputTypeError('object_to_group should be a list of dict types')
        if not all([isinstance(g_to_o, types.DictType) for g_to_o in group_to_object]):
            raise InputTypeError('group_to_object should be a list of dict types')
        if not isinstance(sample_to_response, types.DictType):
            raise InputTypeError('sample_to_response should be a dict type')
        if not isinstance(n_processes, types.IntType) or n_processes < 1:
            raise InputTypeError('n_processes should be a positive int')
        if not isinstance(parse_object_sample, types.FunctionType):
            raise InputTypeError('parse_object_sample should be a function')
        if len(inspect.getargspec(parse_object_sample)[0]) < 1:
            raise InputTypeError('parse_object_sample should take at least one argument')
        
        self.response_variables = np.array([sample_to_response[sample] for sample in sample_to_response.keys()])
        self.max_scope = len(object_to_group)-1
        self.n_unmasked_samples = len(sample_to_response)
        self.mask_length_stack = []
        self.aggregate_index_mask = None

        sample_indices = dict([(sample_to_response.keys()[index], index) for index
                               in range(len(sample_to_response.keys()))])
        
        assert all([self.response_variables[sample_indices[sample]] == sample_to_response[sample] for sample in sample_to_response.keys()]),\
            "sample_indices are not able to map correctly back to the response variable"

        self.feature_columns = [None for i in range(len(group_to_object))]
        self.feature_records = [None for i in range(len(group_to_object))]
        self.scope_map = [None for i in range(len(group_to_object))]

        def build_scope_columns_records_splits(scope, group_to_object, object_to_group, sample_indices):
            feature_records = dict()
            feature_columns = dict()
            scope_map = dict()
            
            for group in group_to_object[scope]:
                objects = group_to_object[scope][group]

                feature_column = np.zeros(len(sample_indices))
                for obj in objects:
                    sample_id = parse_object_sample(obj)
                    assert sample_id, 'parsed sample_id is None or empty'
                    try:
                        feature_column[sample_indices[sample_id]] += 1
                    except KeyError:
                        continue
                feature_columns[group] = feature_column

                feature_abundance = feature_column.sum()
                feature_id = group
                feature_records[group] = FeatureRecord(feature_id, scope, feature_abundance)

                for final_scope in range(len(group_to_object)):
                    if final_scope == scope:
                        scope_map[(group, final_scope)] = [(scope, group)]
                        continue

                    final_group_set = set()
                    for obj in objects:
                        try:
                            final_group_set.add( (final_scope, object_to_group[final_scope][obj]) )
                        except KeyError:
                            continue
                    scope_map[(group, final_scope)] = list(final_group_set)
                    
            return (feature_columns, feature_records, scope_map)

        process_definitions = []
        results = []
        for scope in range(len(group_to_object)):
            process_definitions.append( ProcessDefinition(build_scope_columns_records_splits, positional_arguments=(scope, group_to_object, object_to_group, sample_indices), tag=scope) )
        multiprocess_functions(process_definitions, results.append, n_processes)

        for scope, result in results:
            feature_columns, feature_records, scope_map = result
            self.feature_columns[scope] = feature_columns
            self.feature_records[scope] = feature_records
            self.scope_map[scope] = scope_map
        for scope in range(len(self.scope_map)):
            assert isinstance(self.feature_records[scope], types.DictType),\
                "feature_records is not a list of dicts"
            assert isinstance(self.feature_columns[scope], types.DictType),\
                "feature_columns is not a list of dicts"
            assert isinstance(self.scope_map[scope], types.DictType),\
                "scope is not a list of dicts"
            for key in self.feature_records[scope]:
                assert self.feature_records[scope][key].get_id() == key,\
                    "feature_record had mismatched id to it's key"
                assert self.feature_records[scope][key].get_scope() == scope,\
                    "feature_record had mismatched scope to it's key"
            for key in self.feature_columns[scope]:
                assert self.feature_columns[scope][key].shape[0] == self.n_unmasked_samples
            for key in self.scope_map[scope]:
                assert isinstance(key, types.TupleType) and len(key) == 2,\
                    "scope_map key is not a scope/group tuple"
                final_groups = self.scope_map[scope][key]
                assert isinstance(final_groups, types.ListType),\
                    "scope_map includes a an entry that is not a list"
                #Note: this is a pretty good test of the scope_map that
                #was helpful in debugging, but it has nasty complexity
                #==============================================================
                # for t in final_groups:
                #    assert isinstance(t, types.TupleType) and len(t) == 2,\
                #        "scope_map includes an entry that is not a scope/group pair"
                #    final_scope, final_group = t
                #    final_objs = group_to_object[final_scope][final_group]
                #    objs = group_to_object[scope][key[0]]
                #    assert any([final_obj in objs for final_obj in final_objs]),\
                #        "scope_map has an entry that lists a group which shares no objects"
                #==============================================================

    def push_mask(self, mask):
        if not isinstance(mask, np.ndarray):
            raise InputTypeError('mask is not a Numpy array')
        self.mask_length_stack.append(mask.shape[0])
        
        if self.aggregate_index_mask == None:
            self.aggregate_index_mask = mask
        else:
            self.aggregate_index_mask = np.hstack( (self.aggregate_index_mask, mask) )
        
    def pop_mask(self):
        if len(self.mask_length_stack) == 0:
            return None
        pop_length = self.mask_length_stack.pop()
        
        assert self.aggregate_index_mask.shape[0] >= pop_length,\
            "the aggregate_index_mask is missing entries"
        
        self.aggregate_index_mask = self.aggregate_index_mask[:-pop_length]
        if self.aggregate_index_mask.shape[0] == 0:
            self.aggregate_index_mask = None
        
        return pop_length
        
    def get_feature_abundance(self, scope, group):
        return sum(self.get_feature_column(scope, group))
    
    def get_n_unmasked_samples(self):
        return self.n_unmasked_samples

    def get_max_scope(self):
        return self.max_scope

    def get_feature_column(self, scope, group):
        if not isinstance(scope, types.IntType) or scope < 0 or scope > self.get_max_scope():
            raise InputTypeError("scope (%s) is not a valid scope index" %scope)
        try:
            if self.aggregate_index_mask != None:
                return self.feature_columns[scope][group][self.aggregate_index_mask]
            else:
                return self.feature_columns[scope][group]
        except KeyError:
            raise KeyError("group (%s) not found at scope (%s)" %(group, scope))

    def get_feature_record(self, scope, group):
        if not isinstance(scope, types.IntType) or scope < 0 or scope > self.get_max_scope():
            raise InputTypeError("scope (%s) is not a valid scope index" %scope)
        try:
            return self.feature_records[scope][group]
        except KeyError:
            raise KeyError("group (%s) not found at scope (%s)" %(group, scope))

    def get_split_groups(self, scope, group, final_scope):
        if not isinstance(scope, types.IntType) or scope < 0 or scope > self.get_max_scope():
            raise InputTypeError("scope (%s) is not a valid scope index" %scope)
        if not isinstance(final_scope, types.IntType) or final_scope < 0 or final_scope > self.get_max_scope():
            raise InputTypeError("final_scope (%s) is not a valid scope index" %final_scope)
        try:
            return self.scope_map[scope][(group, final_scope)]
        except KeyError:
            raise KeyError("(group, final_scope) (%s, %s) not found at scope (%s)" %(group, final_scope, scope))

    def get_response_variables(self):
        if self.aggregate_index_mask != None:
            return self.response_variables[self.aggregate_index_mask]
        else:
            return self.response_variables

def build_problem_data(group_map_files, mapping_file, prediction_field,
                       start_level, include_only, negate, n_processes):
    simple_var_types = [
                 ("n_processes", types.IntType),
                 ("start_level", types.IntType),
                 ("mapping_file", types.FileType),
                 ("negate", types.BooleanType),
                 ("prediction_field", types.StringType),
                 ("include_only", (types.NoneType, types.ListType)),
                 ("group_map_files", types.ListType)
                ]
    for var_name, var_type in simple_var_types:
        check_input_type(var_name, locals()[var_name], var_type)
    if not all([isinstance(f, types.FileType) for f in group_map_files]):
        raise InputTypeError("group_map_files should be a list of open files")
    if include_only != None:
        if not isinstance(include_only[0], types.StringType):
            raise InputTypeError("include_only[0] should be of type string")
        if not isinstance(include_only[1], types.ListType) or \
                not all([isinstance(value, types.StringType) for value in include_only[1]]):
            raise InputTypeError("include_only[1] should be a list of strings")
    if start_level >= len(group_map_files) or start_level < 0:
        raise InputTypeError("start_level (%s) is not a valid scope index; group_map_files is of length %s" \
                             %(start_level, len(group_map_files)))
    
    #For each scope, build a map from group to object and vice versa
    group_to_object = []
    object_to_group = []
    for map_file in group_map_files:
        g_to_o, o_to_g = read_split_file(map_file)
        group_to_object.append(g_to_o)
        object_to_group.append(o_to_g)
        
        assert isinstance(g_to_o, types.DictType),\
            "read_split_file did not return a dict type"
        assert isinstance(o_to_g, types.DictType),\
            "read_split_file did not return a dict type"

    #Find a list of sample names from our group names
    #An alternative is 'samplenames = samplemap.keys()', but that may have records without features
    samplename_set = set()
    for grp in group_to_object[start_level]:
        objs = group_to_object[start_level][grp]
        for obj in objs:
            samplename = parse_object_string_sample(obj)
            samplename_set.add(samplename)
    samplenames = list(samplename_set)

    #get a map of sample name to it's properties
    samplemap = read_mapping_file(mapping_file)

    def include_samplename(samplename):
        if include_only == None:
            return True
        sample_dict = samplemap[samplename]
        
        try:
            if (sample_dict[include_only[0]] in include_only[1]) ^ negate:
                return True
            return False
        except KeyError:
            raise KeyError('include_only[0] is not a field in mapping_file')
        

    sample_to_response = {}
    for samplename in samplenames:
        if include_samplename(samplename):
            try:
                sample_to_response[samplename] = samplemap[samplename][prediction_field]
            except KeyError:
                raise KeyError('prediction_field is not a field in mapping_file.')

    problem_data = ProblemData(group_to_object, object_to_group, sample_to_response, n_processes)

    feature_vector = FeatureVector([FeatureRecord(group, start_level, problem_data.get_feature_abundance(start_level, group))
                                    for group in group_to_object[start_level].keys()])

    return problem_data, feature_vector

