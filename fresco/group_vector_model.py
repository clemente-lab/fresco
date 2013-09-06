import numpy as np

class GroupVectorModel:
    def __init__(self, sklmodel):
        self.sklmodel = sklmodel
        
    def fit(self, problem_data, feature_vector, X=None, mask=None):
        self.sklmodel.fit( *get_Xy(problem_data, feature_vector, X, mask) )
    
    def predict(self, problem_data, feature_vector, X=None, mask=None):
        return self.sklmodel.predict( get_Xy(problem_data, feature_vector, X, mask)[0] )
    
    def get_feature_scores(self):
        #Some models report the coeficients of each feature instead of the importances
        #For these, multiclass problems will generate a score lists for each class
        #To get a roughly meaningful single score list, we average these score lists
        if hasattr(self.sklmodel, 'coef_'):
            coefs = np.absolute(self.sklmodel.coef_)
            if len(coefs.shape) == 2:
                avgs = coefs[0]
                for c in coefs[1:]:
                    avgs += c
                avgs /= len(avgs)
                return avgs
            else:
                return coefs
        elif hasattr(self.sklmodel, 'feature_importances_'):
            return self.sklmodel.feature_importances_
        return None
    
    
def build_sample_matrix(problem_data, feature_vector):
    columns = []
    rec_list = feature_vector.get_record_list()
    for record in rec_list:
        group_id = record.get_id()
        group_scope = record.get_scope()
        columns.append(problem_data.get_feature_column(group_scope, group_id))
            
    sample_matrix = np.array(columns).T
        
    return sample_matrix
    
def get_Xy(problem_data, feature_vector, X_, mask):
    X = build_sample_matrix(problem_data, feature_vector) if X_ == None else X_
    y = problem_data.get_response_variables()
    
    if mask != None:
        X = X[mask]
        y = y[mask]
        
    return X, y
        