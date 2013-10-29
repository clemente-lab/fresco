#Given an open file that maps group names to object names, get maps of group -> object, object -> group  
#Each line of the file should be a tab-separated list of names, where object names follow group name     
#For example:                                                                                            
#[GROUP NAME 1]\t[OBJECT NAME 1]\t[OBJECT NAME 2]                                                        
#[GROUP NAME 2]\t[OBJECT NAME 3]                                                                         
def read_split_file(split_file):
    group_to_object = {}
    object_to_group = {}

    for line in split_file:
        line = line.strip()

        if not line:
            continue

        entries = map(lambda e: e.strip(), line.split('\t'))
        group_id = entries[0]
        obj_ids = entries[1:]

        if group_id in group_to_object:
            raise FeatureMapFileFormatError("The feature with ID '%s' was "
                    "already found in the feature map file. Feature IDs must "
                    "be unique." % group_id)
        else:
            group_to_object[group_id] = obj_ids

        for obj_id in obj_ids:
            if obj_id in object_to_group:
                raise FeatureMapFileFormatError("The object with ID '%s' was "
                        "already found in the feature map file (mapped to "
                        "feature '%s'). Object IDs must be unique." % (obj_id,
                        object_to_group[obj_id]))
            else:
                object_to_group[obj_id] = group_id

    return group_to_object, object_to_group

def read_mapping_file(mapping_file):   
    samplemap = {}
    
    header_row = True
    headerfields = None
    
    n_fields = None
    
    for row in mapping_file:
        row = row.strip()

        if not row:
            continue

        fields = map(lambda e: e.strip(), row.split('\t'))

        if header_row:
            headerfields = fields
            n_fields = len(headerfields)
            header_row = False
            continue

        if len(fields) != n_fields:
            raise MappingFileFormatError("row does not have the same number of columns (=%s) as the first row" % n_fields)
        
        sample_id = fields[0]
        sample_md = fields[1:]
        

        if sample_id in samplemap:
            raise MappingFileFormatError("The sample with ID '%s' was already "
                    "found in the mapping file. Sample IDs must be unique."
                    % sample_id)
        else:
            fieldmap = {}
            for i in range(len(sample_md)):
                fieldmap[headerfields[i + 1]] = sample_md[i]
            samplemap[sample_id] = fieldmap

    return samplemap


class MappingFileFormatError(Exception):
    pass


class FeatureMapFileFormatError(Exception):
    pass
