import copy
import numpy as np
from parse import parse_object_name

class FeatureVector:
    def __init__(self, rec_list, sample_matrix, name_dict, to_copy=None):
        if to_copy != None:
            self.rec_list = copy.deepcopy(to_copy.rec_list)
            self.sampletable_arr = copy.deepcopy(to_copy.sampletable_arr)
            self.sampletable_name_map = to_copy.sampletable_name_map
        else:
            self.rec_list = rec_list
            self.sampletable_arr = sample_matrix
            self.sampletable_name_map = name_dict

    def pop_feature(self, feature_n):
        #NOTE: this process is 2x faster than np.delete                      
        new_arr = self.sampletable_arr[:, :-1]
        new_arr[:, :feature_n] = self.sampletable_arr[:, :feature_n]
        new_arr[:, feature_n:] = self.sampletable_arr[:, feature_n+1:]
        self.sampletable_arr = new_arr
        return self.rec_list.pop(feature_n)

    def add_feature(self, feature_rec, seq_list, partial = False):
        #if the record has already been added, there's no more work to do          
        old_record_index = None
        for i in range(len(self.rec_list)):
            record = self.rec_list[i]
            if record.get_ID() == feature_rec.get_ID() and \
                    record.get_threshold() == feature_rec.get_threshold():
                if partial:
                    old_record_index = i
                    break
                else:
                    return

        #otherwise, build a new feature column                                  
        new_feature = np.zeros( (len(self.sampletable_name_map.keys()), 1) )

        for seq_id in seq_list:
            sample_id = parse_object_name(seq_id)

            #use map, catch a KeyError for missing sample in rarefaction table              
            try:
                index = self.sampletable_name_map[sample_id]
                new_feature[index] += 1
            except KeyError:
                continue

        if partial and old_record_index != None:
            for i in range(new_feature.shape[0]):
                self.sampletable_arr[i, old_record_index] += new_feature[i]
            parent_list = self.rec_list[old_record_index].get_parents()
            parent_list += feature_rec.get_parents()
        else:
            #add the new column                                                                       
            new_arr = np.zeros( (self.sampletable_arr.shape[0], self.sampletable_arr.shape[1] + 1) )
            new_arr[:, :self.sampletable_arr.shape[1]] = self.sampletable_arr
            new_arr[:, -1:] = new_feature
            self.sampletable_arr = new_arr
            #self.sampletable_arr = np.hstack((self.sampletable_arr, new_feature)) #this takes ~60%   
            self.rec_list.append(feature_rec)

    def add_feature_partial(self, feature_rec, seq_list):
        new_feature = np.zeros( (len(self.sampletable_name_map.keys()), 1) )
        for seq_id in seq_list:
            sample_id = parse_object_name(seq_id)
            #Check first if the sample still exists, it might just be in the split file              
            #print "looking for id", sample_id, "in", len(self.sampletable_names)                    
            try:
                index = self.sampletable_name_map[sample_id]
                new_feature[index] += 1
            except KeyError:
                continue

        #print "after adding", len(seq_list), "new sequences, the average population is", np.mean(new\_feature)                                                                                             
        #we need to make sure this feature id and theshold aren't already in the vector before adding it
        old_record_i = None
        for i in range(len(self.rec_list)):
            record = self.rec_list[i]
            if record.get_ID() == feature_rec.get_ID() and \
                    record.get_threshold() == feature_rec.get_threshold():
                old_record_i = i
                break
        if old_record_i != None:
            #the feature is already in use, so we can just add our population vector to the old one       

            for i in range(new_feature.shape[0]):
                self.sampletable_arr[i, old_record_i] += new_feature[i]
            parent_list = self.rec_list[old_record_i].get_parents()
            parent_list += feature_rec.get_parents()
        else:
            self.rec_list.append(feature_rec)
            self.sampletable_arr = np.hstack((self.sampletable_arr, new_feature))

    def get_sampletable(self, rarify=None):
        samplenames = sorted(self.sampletable_name_map.keys(), 
                             key=lambda name: self.sampletable_name_map[name])
        sampletable = []
        for i in range(len(samplenames)):
            if rarify != None:
                s = float(rarify) / np.sum(nrow)
                nrow = self.sampletable_arr[i] * s
            else:
                nrow = self.sampletable_arr[i]
            sampletable.append( (samplenames[i], nrow) )
        print "IN FUNC"
        return sampletable

    def get_record_list(self):
        return self.rec_list

    def get_copy(self):
        return FeatureVector(None, None, None, self)



class FeatureRecord:
    def __init__(self, feature_id, threshold, pop, parents = [], children = []):
        self.feature_id = feature_id
        self.threshold = threshold
        self.parents = parents
        self.children = children
        self.pop = pop

    def get_pop(self):
        return self.pop
    def get_threshold(self):
        return self.threshold
    def get_ID(self):
        return self.feature_id
    def get_parents(self):
        return self.parents
    def get_children(self):
        return self.children
