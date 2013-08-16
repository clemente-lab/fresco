import numpy as np
import parse

#Build a dictionary from class strings (as given the in the samplemap) to numerical values
#samplemap should be a dictionary from sample name to a dictionary of field names to field values
#predictfield should be the field name of iterest to the problem
def build_classmap(samplemap, predictfield):
    class_set = set([samplemap[samplename][predictfield]
                     for samplename in samplemap.keys()])
    classes = list(class_set)
    classmap = {}
    for i in range(len(classes)):
        classmap[classes[i]] = i

    return classmap


#Build a feature matrix, where columns are features and rows are samples
#samplename_map maps from a samplename to the row index it should represent
#rec_list is a list of FeatureRecords
#group_to_object maps from group names to object names
#The scale option can be used to normalize the values in each row to a specified value
def build_sample_matrix(samplename_map, rec_list, group_to_object, scale=None):
    #helper function for normalizing every row in a table
    def scale_table(table):
        if scale != None:
            for r_index in range(len(table)):
                row = table[r_index]
                s = scale / np.sum(row)
                table[r_index] /= s
        return table

    #initialize a matrix
    mat = np.zeros( (len(samplename_map.keys()), len(rec_list)) )

    #iterate over each record and add the objects for each sample to the sample
    for rec_index in range(len(rec_list)):
        record = rec_list[rec_index]
        group_id = record.get_ID()
        objects = group_to_object[record.get_threshold()][group_id]
        for obj in objects:
            sample_id = parse.parse_object_name(obj)
            try:
                sample_index = samplename_map[sample_id]
                mat[sample_index][rec_index] += 1
            except KeyError:
                continue
    return mat
