#!/usr/bin/env python
from __future__ import division

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def parse_object_string_sample(object_string):
    try:
        underscore_index = object_string.rindex('_')
    except ValueError:
        raise ValueError("Could not find an underscore separating the sample "
                         "name from the object identifier.")
    else:
        return object_string[:underscore_index]

def parse_model_string(model_str):
    """Parse a string describing a classifier (lr, sv, rf) into a Model object."""
    if model_str == 'rf':
        return RandomForestClassifier(n_estimators=10)
    elif model_str == 'sv':
        return LinearSVC()
    elif model_str == 'lr':
        return LogisticRegression(penalty='l2')
    else:
        raise InputTypeError("model_str should be one of the following "
                             "strings: rf, sv, lr")

def get_list_accuracy(y1, y2):
    if len(y1) != len(y2):
        raise ValueError("The two lists must be of equal length when being "
                         "compared for accuracy. One list was of length %d "
                         "and the other was of length %d."
                         % (len(y1), len(y2)))
    elif len(y1) < 1:
        raise ValueError("Cannot compute the accuracy of two empty lists.")
    else:
        match_count = 0
        for e1, e2 in zip(y1, y2):
            if e1 == e2:
                match_count += 1
             
        score = float(match_count) / len(y1)
        return score

def normalized_scores(scores, exclude_list):
    scores = scores[:]
    
    invalid_list = []
    for i in range(len(scores)):
        if scores[i] == None:
            scores[i] = 0
            invalid_list.append(i)
            
    mini = min(scores)
    if float(mini) < 0:
        scores -= mini

    for e in exclude_list:
        scores[e] = 0
    for i in invalid_list:
        scores[i] = 0
    
    sumi = sum(scores)
    if float(sumi) != 0:
        scores /= sumi
        return scores
    else: #since all elements are non-negative, all elements must be 0
        return []

class InputTypeError(Exception):
    pass

def check_input_type(var_name, var, var_type):
    valid = False

    if isinstance(var_type, tuple):
        if any([isinstance(var, t) for t in var_type]):
            valid = True
    elif isinstance(var, var_type):
        valid = True

    if not valid:
        raise_input_type(var_name, var_type, type(var).__name__)

def raise_input_type(var_name, var_required_type, var_actual_type):
    if isinstance(var_required_type, tuple):
        error_msg = ("%s is of type %s, but should be one of the following "
                     "types: %s" % (var_name, var_actual_type,
                     ', '.join([t.__name__ for t in var_required_type])))
    else:
        error_msg = "%s should be of type %s, but is of type %s" % (var_name,
                var_required_type.__name__, var_actual_type)

    raise InputTypeError(error_msg)
