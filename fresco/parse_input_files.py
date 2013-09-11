import types
from fresco.utils import InputTypeError

#Given an open file that maps group names to object names, get maps of group -> object, object -> group  
#Each line of the file should be a tab-separated list of names, where object names follow group name     
#For example:                                                                                            
#[GROUP NAME 1]\t[OBJECT NAME 1]\t[OBJECT NAME 2]                                                        
#[GROUP NAME 2]\t[OBJECT NAME 3]                                                                         
def read_split_file(split_file):
    if not isinstance(split_file, types.FileType):
        raise InputTypeError("split_file should of an open file")
    
    group_to_object = {}
    object_to_group = {}

    while True:
        line = split_file.readline()
        if line == '':
            break
        entries = line.split('\t')
        group_to_object[entries[0]] = entries[1:]
        for obj in entries[1:]:
            object_to_group[obj] = entries[0]

    return group_to_object, object_to_group

def read_mapping_file(mapping_file):
    if not isinstance(mapping_file, types.FileType):
        raise InputTypeError("mapping_file should of an open file")
    
    samplemap = {}
    
    header_row = True
    headerfields = None
    
    n_fields = None
    
    for row in mapping_file:
        if header_row:
            headerfields = row.split("\t")
            n_fields = len(headerfields)
            header_row = False

        fields = row.split("\t")
        if len(fields) != n_fields:
            raise MappingFileFormatError("row does not have the same number of columns (=%s) as the first row" %n_fields)
        
        fieldmap = {}
        for i in range(1, len(fields)):
            fieldmap[headerfields[i]] = fields[i]
        samplemap[fields[0]] = fieldmap
    
    return samplemap

class MappingFileFormatError(Exception):
    pass