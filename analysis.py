import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB

import os
import os.path
import time
import multiprocessing


def paralel_mp_limited(functions, n_procs, outcome_handler, verb = False):
    processes = []
    result_queue = multiprocessing.Queue()

    def function_wrapper(function, function_args, result_queue):
        outcome = function(*function_args)
        result_queue.put_nowait(outcome)

    def wait_for_finished(n_survivers):
        while len(processes) > n_survivers:
            while not result_queue.empty():
                res = result_queue.get()
                if verb: print "Result finished."
                outcome_handler.handle_outcome(res)

            for pn in range(len(processes)):
                process = processes[pn]
                if not process.is_alive():
                    processes.pop(pn)
                break
        

    for i in range(len(functions)):
        wait_for_finished(n_procs)

        process = multiprocessing.Process(target = function_wrapper, args = 
                                          (functions[i][0], functions[i][1], result_queue) )
        processes.append(process)
        process.start()

    wait_for_finished(0)

def crossValidate(n, xtrain, ytrain, model, nclasses):
    l = len(ytrain)
    ypred = []
    conf = np.zeros( (nclasses, nclasses) )
    for i in range(n):
        start = int( l*(float(i)/n) )
        end = int( l*(float(i+1)/n) )

        xptest = xtrain[start:end]
        yptest = ytrain[start:end]

        xtrainlist = xtrain.tolist()
        ytrainlist = ytrain.tolist()

        xptrain = []
        yptrain = []
        for x in xtrain[0:start]:
            xptrain.append(x)
        for y in ytrain[0:start]:
            yptrain.append(y)
        for x in xtrain[end:]:
            xptrain.append(x)
        for y in ytrain[end:]:
            yptrain.append(y)

        xptrain = np.array(xptrain)
        yptrain = np.array(yptrain)

        model.fit(xptrain, yptrain)
        ypred += model.predict(xptest).tolist()
    return ypred

def confusionMatrix(yreal, ypred, nclasses):
    matrix = np.zeros( (nclasses, nclasses) )
    for i in range(len(yreal)):
        row = int(yreal[i])
        col = int(ypred[i])
        matrix[row, col] += 1
    return matrix

def process(model, problem, n_cross_folds, outcome_maker, outcome_maker_params):
    nclasses = problem.getNClasses()

    #for stochastic classifiers, average over niter                                                  
    niter = model.getIterations()

    outcome = {}
#    outcome['problem_spec'] = problem_spec
    outcome['problem'] = problem
    outcome['model'] = model
    outcome['n_cross_folds'] = n_cross_folds

    outcome['predictions'] = []
    for i in range(niter):
        ypred = crossValidate(n_cross_folds, problem.getX(), problem.getY(), 
                              model.getModel(), nclasses)
        outcome['predictions'].append(ypred)
   
    return outcome_maker(*outcome_maker_params, **outcome)

def get_average_list_accuracy(yreal, predictions):
    acc = 0
    for i in range(len(predictions)):
        acc += get_list_accuracy(yreal, predictions[i])
    return float(acc)/len(predictions)

def get_list_accuracy(y1, y2):
    correct = sum( [1 if y1[i] == y2[i] else 0 for i in range(len(y1))] )
    return float(correct) / len(y1)

class OutcomeHandler:
    #upon producing an outcome, dump_function will be called with                                            #dump_function(outcome, *dump_parameters)
    def __init__(self, dump_function, dump_parameters):
        self.dump_parameters = dump_parameters
        self.dump_function = dump_function

    def handle_outcome(self, outcome):
        self.dump_function(outcome, *self.dump_parameters)
