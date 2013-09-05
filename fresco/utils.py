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
    else:
        return LogisticRegression(penalty='l2')

def get_list_accuracy(y1, y2):
    same_vector = [1 if y1[i] == y2[i] else 0 for i in range(len(y1))]
    correct = sum( same_vector )
    return float(correct) / len(y1)
