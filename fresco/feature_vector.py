class FeatureVector:
    def __init__(self, rec_list):
        self.rec_list = rec_list

    def get_record_list(self):
        return self.rec_list
    
    def pop_feature(self, index):
        return self.rec_list.pop(index)

class FeatureRecord:
    def __init__(self, feature_id, scope):
        self.feature_id = feature_id
        self.scope = scope
        
    def get_scope(self):
        return self.scope
    def get_id(self):
        return self.feature_id
