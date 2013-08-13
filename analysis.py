
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB

from setup_problems import build_problem

import os
import os.path
import time
import multiprocessing


def paralel_mp_limited(functions, n, outcome_handler, verb = False):
    def fwrapper(function, results):
        outcome = function[0](*function[1])
        results.put_nowait(outcome)

    ret = []

    processes = []
    results = multiprocessing.Queue()

    for i in range(len(functions)):
        while len(processes) >= n:
            while not results.empty():
                res = results.get()
                if verb: print "Result finished."
                outcome_handler.handle_outcome(res)

            for pn in range(len(processes)):
                process = processes[pn]
                if not process.is_alive():
                    processes.pop(pn)
                    break

        process = multiprocessing.Process(target = fwrapper, args = (functions[i], results) )
        processes.append(process)
        process.start()

    
    while len(processes) != 0:
        while not results.empty():
            res = results.get()
            if verb: print "Result finished."
            outcome_handler.handle_outcome(res)
            
        for pn in range(len(processes)):
            process = processes[pn]
            if not process.is_alive():
                processes.pop(pn)
                break
        
    for process in processes:
        process.join()

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

def build_problem_process(model, problem_builder, problem_spec, settings, outcome_maker):
    problem = problem_builder(**problem_spec)
    return process(model, problem, settings, outcome_maker)

def process(model, problem, settings, outcome_maker):
    nclasses = problem.getNClasses()
    cross_folds = settings['cross_folds']

    collect_duration = 'duration' in settings.keys() and settings['duration']
    
    #for stochastic classifiers, average over niter                                                  
    niter = model.getIterations()
    #for models with feature ranks, like decision trees                                              
    rank = model.getFeatureRank()

    if collect_duration:
        before = time.time()

    outcome = {}
#    outcome['problem_spec'] = problem_spec
    outcome['problem'] = problem
    outcome['model'] = model
    outcome['settings'] = settings

    outcome['predictions'] = []
    for i in range(niter):
        ypred = crossValidate(cross_folds, problem.getX(), problem.getY(), 
                              model.getModel(), nclasses)
        outcome['predictions'] .append( ypred )

    if collect_duration:
        duration = time.time() - before
        outcome['duration'] = duration
   
    if rank and model.getModel().compute_importances:
        outcome['importances'] = model.getModel().feature_importances_
    if rank and isinstance(model.getModel(), LogisticRegression):
        outcome['importances'] = model.getModel().coef_
    if rank and isinstance(model['model'], SVC) and model.getModel().kernel == 'linear':
        outcome['importances'] = model.getModel().coef_

    return outcome_maker(**outcome)

def get_average_list_accuracy(yreal, predictions):
    acc = 0
    for i in range(len(predictions)):
        acc += get_list_accuracy(yreal, predictions[i])
    return float(acc)/len(predictions)

def get_list_accuracy(y1, y2):
    correct = sum( [1 if y1[i] == y2[i] else 0 for i in range(len(y1))] )
    return float(correct) / len(y1)

def decompose(X, dim):
    pca = PCA(n_components=dim)
    pca.fit(X)
    dX = pca.transform(X)
    return dX

def rarefactor(biomfile, outputfile, n):
    if not os.path.isfile(outputfile + ".txt"):
        os.system("python $QIIME/scripts/single_rarefaction.py -i "+biomfile+
                  " -o "+outputfile+".biom -d "+str(n))
        os.system("convert_biom.py -i "+outputfile+".biom -o "
                  +outputfile+".txt -b --header_key=\"taxonomy\"")
        os.system("rm "+outputfile+".biom")


class OutcomeHandler:
    #upon producing an outcome, dump_function will be called with                                                                                                                                            
    #dump_function(outcome, *dump_parameters)                                                                                                                                                                
    def __init__(self, dump_function, dump_parameters):
        self.dump_parameters = dump_parameters
        self.dump_function = dump_function

    def handle_outcome(self, outcome):
        self.dump_function(outcome, *self.dump_parameters)
