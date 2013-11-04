import numpy as np
import types
from fresco.utils import InputTypeError, check_input_type, parse_object_string_sample
from fresco.feature_vector import FeatureRecord
from fresco.parallel_processing import multiprocess_functions, ProcessDefinition
from fresco.parse_input_files import read_split_file, read_mapping_file
from mask_stack import MaskStack
import inspect

class ProblemData:
    def __init__(self, group_to_object, object_to_group, sample_to_response, n_processes, parse_object_string=parse_object_string_sample):
        """
        Builds a ProblemData object which is responsible for providing an interface to all aspects of a dataset.
        
        Args: object_to_group, group_to_object, 
            
        """
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
        if not isinstance(n_processes, types.IntType) or n_processes < 0:
            raise InputTypeError('n_processes should be a non-negative int')
        if not isinstance(parse_object_string, types.FunctionType):
            raise InputTypeError('parse_object_sample should be a function')
        if len(inspect.getargspec(parse_object_string)[0]) < 1:
            raise InputTypeError('parse_object_sample should take at least one argument')
        
        self.response_variables = np.array([sample_to_response[sample] for sample in sample_to_response.keys()])
        self.n_scopes = len(object_to_group)
        self.n_unmasked_samples = len(sample_to_response)
        
        # Keep a stack of masks for various levels of data partitions
        self.mask_stack = MaskStack(self.n_unmasked_samples)

        sample_indices = dict([(sample_to_response.keys()[index], index) for index
                               in range(len(sample_to_response.keys()))])
        
        assert all([self.response_variables[sample_indices[sample]] == sample_to_response[sample] for sample in sample_to_response.keys()]),\
            "sample_indices are not able to map correctly back to the response variable"

        process_definitions = []
        results = []
        for scope in range(len(group_to_object)):
            process_definitions.append( ProcessDefinition(build_group_records, positional_arguments=(scope, group_to_object[scope], sample_indices, parse_object_string), tag=scope) )
        multiprocess_functions(process_definitions, results.append, n_processes)

        self.group_records = [None] * self.n_scopes
        for scope, result in results:
            self.group_records[scope] = result
        
        for scope in range(self.n_scopes):
            for group in self.group_records[scope]:
                self.build_scope_map(self.group_records[scope][group], group_to_object, object_to_group)
        

        for scope in range(self.n_scopes):
            for key in self.group_records[scope]:
                assert self.group_records[scope][key].feature_record.get_id() == key,\
                    "feature_record had mismatched id to it's key"
                assert self.group_records[scope][key].feature_record.get_scope() == scope,\
                    "feature_record had mismatched scope to it's key"
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

    def build_scope_map(self, group_record, group_to_object, object_to_group):
        scope_map = [None] * self.n_scopes
        scope = group_record.feature_record.get_scope()
        group = group_record.feature_record.get_id()
        objects = group_to_object[scope][group]
        
        for final_scope in range(self.n_scopes):
            if final_scope == scope:
                scope_map[final_scope] = [(scope, group)]
                continue
    
            final_group_set = set()
            for obj in objects:
                try:
                    #make sure not to reference the object_to_group string, or object_to_string won't be collected
                    final_group_id = object_to_group[final_scope][obj]
                    final_group = self.group_records[final_scope][final_group_id].feature_record.get_id()
                    final_group_set.add( (final_scope, final_group) )
                except KeyError:
                    continue
            scope_map[final_scope] = list(final_group_set)
            
        group_record.scope_map = scope_map

    
    def push_mask(self, mask):
        self.mask_stack.push_mask(mask)

    def pop_mask(self):
        self.mask_stack.pop_mask()
        
    def get_feature_abundance(self, scope, group):
        try:
            #is sum(self.get_feature_column(scope, group)) faster?
            mask = self.mask_stack.get_aggregate_mask()
            return sum([abundance for index, abundance in 
                        self.group_records[scope][group].sparce_sample_abundances
                        if mask[index]])
        except KeyError:
            raise KeyError("group (%s) not found at scope (%s)" %(group, scope))
    
    def get_n_unmasked_samples(self):
        return self.n_unmasked_samples

    def get_max_scope(self):
        return self.n_scopes - 1

    def get_group_ids(self, scope):
        return [group_id for group_id in self.group_records[scope]]
    
    def get_feature_column(self, scope, group):
        if not isinstance(scope, types.IntType) or scope < 0 or scope > self.get_max_scope():
            raise InputTypeError("scope (%s) is not a valid scope index" %scope)

        try:
            group_record = self.group_records[scope][group]

            #convert from sparce to dense format
            feature_column = np.zeros(self.n_unmasked_samples)
            for index, abundance in group_record.sparce_sample_abundances:
                feature_column[index] = abundance

            mask = self.mask_stack.get_aggregate_mask()
            return feature_column[mask] 
        except KeyError:
            raise KeyError("group (%s) not found at scope (%s)" %(group, scope))

    def get_feature_record(self, scope, group):
        if not isinstance(scope, types.IntType) or scope < 0 or scope > self.get_max_scope():
            raise InputTypeError("scope (%s) is not a valid scope index" %scope)
        try:
            return self.group_records[scope][group].feature_record
        except KeyError:
            raise KeyError("group (%s) not found at scope (%s)" %(group, scope))

    def get_split_groups(self, scope, group, final_scope):
        if not isinstance(scope, types.IntType) or scope < 0 or scope > self.get_max_scope():
            raise InputTypeError("scope (%s) is not a valid scope index" %scope)
        if not isinstance(final_scope, types.IntType) or final_scope < 0 or final_scope > self.get_max_scope():
            raise InputTypeError("final_scope (%s) is not a valid scope index" %final_scope)
        
        try:
            return self.group_records[scope][group].scope_map[final_scope]
        except KeyError:
            raise KeyError("(group, final_scope) (%s, %s) not found at scope (%s)" %(group, final_scope, scope))

    def get_response_variables(self):
        mask = self.mask_stack.get_aggregate_mask()
        return self.response_variables[mask]

class GroupRecord:
    def __init__(self, feature_record, scope_map, sparce_sample_abundances):
        self.feature_record = feature_record
        self.scope_map = scope_map
        self.sparce_sample_abundances = sparce_sample_abundances #should be numpy array

#should split off the scope maps so that this can be run once per scope
#should not be a method, because we don't want a reference to self
def build_group_records(scope, group_to_object, sample_indices, parse_object_string):
    group_records = dict()
    n_samples = len(sample_indices)
    
    for group in group_to_object.keys():
        objects = group_to_object[group]
        
        sparce_sample_abundances = []
        feature_column = np.zeros((n_samples,))
        for obj in objects:
            sample_id = parse_object_string(obj)
            assert sample_id, 'parsed sample_id is None or empty'
            try:
                feature_column[sample_indices[sample_id]] += 1
            except KeyError:
                continue
            
        for index in range(n_samples):
            abundance = feature_column[index]
            sparce_sample_abundances.append((index, abundance))

        #copy group string to remove reference to group_to_object
        feature_record = FeatureRecord(group[:], scope)
            
        group_record = GroupRecord(feature_record, None, sparce_sample_abundances)
        group_records[group] = group_record
    
    return group_records
        
def build_problem_data(group_map_files, mapping_file, prediction_field,
                       start_level, include_only, negate, n_processes, parse_object_string=parse_object_string_sample):
    simple_var_types = [
                 ("n_processes", types.IntType),
                 ("start_level", types.IntType),
                 ("negate", types.BooleanType),
                 ("prediction_field", types.StringType),
                 ("include_only", (types.NoneType, types.ListType,
                                   types.TupleType)),
                 ("group_map_files", types.ListType)
                ]
    for var_name, var_type in simple_var_types:
        check_input_type(var_name, locals()[var_name], var_type)
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
            samplename = parse_object_string(obj)
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
            sample_fields = None
            try:
                sample_fields = samplemap[samplename]
            except KeyError:
                raise KeyError('A sample name (%s) found in the group files is not a sample in mapping_file.' %samplename)    
            try:
                sample_to_response[samplename] = sample_fields[prediction_field]
            except KeyError:
                raise KeyError('prediction_field is not a field in mapping_file.')

    problem_data = ProblemData(group_to_object, object_to_group, sample_to_response, n_processes, parse_object_string)

    return problem_data
