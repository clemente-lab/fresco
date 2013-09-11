from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def parse_object_string_sample(object_string):
    underscore_index = object_string.rfind('_')
    return object_string[:underscore_index]

#Parse a string describing a classifier (lr, sv, rf) into a Model object
def parse_model_string(model_str):
    if model_str == 'rf':
        return RandomForestClassifier(n_estimators=10)
    if model_str == 'sv':
        return LinearSVC()
    if model_str == 'lr':
        return LogisticRegression(penalty='l2')
    raise InputTypeError("model_str should be one of the following strings: rf, sv, lr")

def get_list_accuracy(y1, y2):
    same_vector = [1 if y1[i] == y2[i] else 0 for i in range(len(y1))]
    correct = sum( same_vector )
    return float(correct) / len(y1)

class InputTypeError(Exception):
    pass

def check_input_type(var_name, var, var_type):
    if isinstance(var_type, tuple):
        if any([isinstance(var, t) for t in var_type]):
            return
    elif isinstance(var, var_type):
        return   
    raise_input_type(var_name, var_type.__name__, type(var).__name__)
        
def raise_input_type(var_name, var_required_type, var_actual_type):
    raise InputTypeError("%s should be of type %s, but is of type %s"
                         %(var_name, var_required_type, var_actual_type))