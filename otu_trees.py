import setup_problems
import numpy as np
import argparse
import sys
from utils import *
import analysis
import copy
import time
from OtuVector import *
import cPickle as pickle
from parse import parse_sequence_name

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser(description=
                                      'Run predictive models on otu tables/sample tables')
    parser.add_argument('-d', help='Set the root directory of the data files',
                        type=str, default='', dest='rootdir')
    parser.add_argument('-o', help='Output pickle file (default stdout)', type=str,
                        default=sys.stdout, dest='output')
    parser.add_argument('-p', help='pop coef', type=float,
                        default=0, dest='p')
    parser.add_argument('-s', help='score coef', type=float,
                        default=-20, dest='s')
    parser.add_argument('-t', help='threshold', type=float,
                        default=None, dest='t')
    parser.add_argument('-k', help='proportion to prune', type=float,
                        default=.01, dest='k')
    parser.add_argument('-proc', help='number of processes', type=int,
                        default=.1, dest='proc')
    parser.add_argument('-c', help='cutoff', type=int,
                        default=None, dest='cutoff')
    parser.add_argument('-start', help='starting threshold', type=int,
                        default=10, dest='start')
    parser.add_argument('-iter', help='iterations', type=int,
                        default=15, dest='iter')
    parser.add_argument('-nrand', help='number of runs to average over', type=int,
                        default=10, dest='n_rand')
    parser.add_argument('-model', help='Model, {rf, lr, sv}', type=str,
                        default="lr", dest='model_str')
    parser.add_argument('-ds', help='Dataset, {449, 626}', type=str,
                        default="449", dest='ds_str')
    parser.add_argument('-prob', help='Predict field', type=str,
                        default="BODY_HABITAT", dest='prob_str')
    parser.add_argument('-sp', help='proportion to prune', type=float,
                        default=.05, dest='split_prop')
    parser.add_argument('-mp', help='proportion to prune', type=float,
                        default=0, dest='merge_prop')
    parser.add_argument('-kp', help='proportion to prune', type=float,
                        default=0, dest='kill_prop')


    args = parser.parse_args()

    procs = args.proc
    output = args.output
    rootdir = args.rootdir

    rarefaction = None
    thresholds = []
    for t in range(0, 38, 3):
        s = '%.2f' % (.61 + .01*t)
        thresholds.append(s)
    thresholds.append('0.99')

    predictfield = args.prob_str
    #predictfield = "DIET"
    #predictfield = "HOST_INDIVIDUAL"
    #predictfield = "BODY_HABITAT"
    #predictfield = "ORIGINAL_SAMPLE_SITE"
    #prules = [('BODY_HABITAT', 'UBERON:skin')]                                        
    #predictfields = ["BODY_HABITAT", "ORIGINAL_SAMPLE_SITE", "HOST_INDIVIDUAL"]                     
    #prules = [None, [('BODY_HABITAT', 'UBERON:skin')], None]                                         
    settings = {'cross_folds':5}

    if args.model_str == 'rf':
        model = Model(model=RandomForestClassifier(n_estimators=10, compute_importances=True), name="RF 500")
    elif args.model_str == 'sv':
        model = Model(model=LinearSVC(), name="Linear SVC")
    else:
        model = Model(model=LogisticRegression(penalty='l2'), name="Logistic Regression L2")
    
    if args.ds_str == '626':
        dataset = setup_problems.get_626_dataset(rootdir)
    elif args.ds_str == '550':
        dataset = setup_problems.get_550_dataset(rootdir)
    else:
        dataset = setup_problems.get_449_dataset(rootdir)

    split_coefs = (20, -20) #(args.p, args.s)
    merge_coefs = (-10, -20) #(1, 1)
    kill_coefs = (0, 0)
    start = args.start

    niterations = args.iter
    n_keep = 1

    #split, merge, kill
    #props = [(.05, 0, 0),
    #         (.05, 0, 0),
    #         (.1, .01, 0),
    #         (.1, .01, 0),
    #         (.15, .05, 0),
    #         (.15, .05, 0)]

    props = [(args.split_prop, args.merge_prop, args.kill_prop) for i in range(10)]

    #Build maps for otu/sequence relationships for each threhold
    otu_to_seq = {}
    seq_to_otu = {}
    for threshold in thresholds:
        o_to_s, s_to_o = setup_problems.read_split_file(dataset.get_split_file(threshold))
        otu_to_seq[threshold] = o_to_s
        seq_to_otu[threshold] = s_to_o

#    samplenames = set()
#    for key in otu_to_seq:
#        l = otu_to_seq[key]
#        for sequence in l:
#            samplenames.add(parse_sequence_name(sequence))
#    samplenames = list(samplenames)


    #Get information about the OTU table of the first threshold
    sampletable, otunames = setup_problems.readOTUTableTXT(dataset.get_otu_table(thresholds[start],
                                                                                 rarefaction, 10),
                                                           holdout=0)
    samplenames = [sample[0] for sample in sampletable]
    
    samplemap = setup_problems.readMappingFile(dataset.get_mappingfile())
    classmap = setup_problems.build_classmap(samplemap, predictfield)

    y = [classmap[samplemap[sample][predictfield]] for sample in samplenames if sample in samplemap.keys()]
    
    if False:
        classes = list(set(y))
    #X = [sample[1] for sample in sampletable]
        rec_list = [OtuRecord(otuname[0], thresholds[start], 
                              misc={'pop':len(otu_to_seq[thresholds[start]][otuname[0]])})
                    for otuname in otunames]
        samplename_map = dict( [(samplenames[index], index) for index in range(len(samplenames))] )
        X = setup_problems.build_sample_matrix(samplename_map, rec_list, 
                                               otu_to_seq)
        cmap = dict( [(yclass, []) for yclass in classes] )
        for i in range(len(y)):
            row = X[i][:]
            s = sum(row)
            cmap[y[i]].append(s)
        for key in classmap.keys():
            cpmap_key = classmap[key]
            pops = np.array(cmap[cpmap_key])
            print "Class", key, "avg", np.mean(pops), "std", np.std(pops)
                
        exit()
    all_accuracies = []
#    n_accuracies = []
#    r_accuracies = []
#    s_accuracies = []
    n_rand = args.n_rand
    niterations = args.iter
    for i in range(n_rand):
        test_indecies = sorted(get_test_indecies(y, .15))
        train_samplenames = []
        test_samplenames = []
        train_y = []
        test_y = []
        train_sampletable = []
        test_sampletable = []
        for i in range(len(y)):
            if len(test_indecies) == 0:
                break
            if test_indecies[0] == i:
                test_indecies.pop(0)
            #test record
                test_sampletable.append(sampletable[i][1])
                test_y.append(y[i])
                test_samplenames.append(samplenames[i])
            else:
            #train record
                train_sampletable.append(sampletable[i][1])
                train_y.append(y[i])
                train_samplenames.append(samplenames[i])
        train_samplename_map = dict( [(train_samplenames[index], index) 
                                      for index in range(len(train_samplenames))] )
        test_samplename_map = dict( [(test_samplenames[index], index) 
                                     for index in range(len(test_samplenames))] )
        train_y = np.array(train_y)
        test_y = np.array(test_y)
        train_sampletable = np.array(train_sampletable)
        test_sampletable = np.array(test_sampletable)

        def score_otu_vector(test_rec_list, scale=None):
            #train_X = train_sampletable
            #test_X = test_sampletable

            def scale_table(table):
                if scale == None:
                    return table
                for r_index in range(table.shape[0]):
                    row = table[r_index]
                    s = scale / np.sum(row)
                    table[r_index] *= s
                return table

            train_X = scale_table(setup_problems.build_sample_matrix(train_samplename_map, test_rec_list, 
                                                         otu_to_seq, scale))
            test_X = scale_table(setup_problems.build_sample_matrix(test_samplename_map, test_rec_list, 
                                                        otu_to_seq, scale))
            model.getModel().fit(train_X, train_y)
            def get_row_pops(table):
                return [np.sum(table[i]) for i in range(table.shape[0])]
            row_pops = get_row_pops(train_X) + get_row_pops(test_X)
            return model.getModel().score(test_X, test_y)

        misc = {"rarefaction":rarefaction, "n":10}
        rec_list = [OtuRecord(otuname[0], thresholds[start], 
                              misc={'pop':len(otu_to_seq[thresholds[start]][otuname[0]])})
                    for otuname in otunames]
        
#        s_accuracies.append( score_otu_vector(rec_list, 500.0) )
#        n_accuracies.append( score_otu_vector(rec_list) )
#        model.getModel().fit(train_sampletable, train_y)
#        r_accuracies.append(model.getModel().score(test_sampletable, test_y))
#        continue

        otu_vectors, outcomes = otu_optimization(train_samplename_map, otu_to_seq, seq_to_otu, train_y,
                                                 rec_list, thresholds, settings, model, misc, procs, 
                                                 niterations, props, n_keep, split_coefs, merge_coefs,
                                                 kill_coefs, initial_sample_matrix=None, saveall=True)

        accuracies = []
        for otu_vector in otu_vectors:
#            rec_list = [OtuRecord(otuname[0], thresholds[start])
#                        for otuname in otunames]
            accuracies.append(score_otu_vector(otu_vector.get_record_list()))
        all_accuracies.append(accuracies)

    #s_acc = np.array(s_accuracies)
    #n_acc = np.array(n_accuracies)
    #r_acc = np.array(r_accuracies)
    #print "Scaling accuracies:", "avg", np.mean(s_acc), "std", np.std(s_acc)
    #print "No rarefaction accuracies:", "avg", np.mean(n_acc), "std", np.std(n_acc)
    #print "Normal rarefaction accuracies (", rarefaction, "):", "avg", np.mean(r_acc), "std", np.std(r_acc)
    #exit()

    avg_accuracies = [sum([all_accuracies[n][iteration] for n in range(len(all_accuracies))])/float(len(all_accuracies)) for iteration in range(len(all_accuracies[0]))]

    print avg_accuracies


def get_test_indecies(y, prop):
    n_records = len(y)
    sample = np.random.choice(n_records, int(n_records*prop), replace=False)
    return sample

def otu_optimization(samplename_map, otu_to_seq, seq_to_otu, y, rec_list, thresholds, settings, 
                     model, misc, procs, niterations, props, n_keep, split_coefs, merge_coefs,
                     kill_coefs, initial_sample_matrix=None, saveall=False):

    if initial_sample_matrix == None:
        initial_sample_matrix = setup_problems.build_sample_matrix(samplename_map, rec_list, 
                                                                   otu_to_seq, None)
    current_otu_vectors = [OtuFeatureVector(rec_list, initial_sample_matrix, samplename_map)]
    
    nclasses = len(set(y))
    if saveall:
        save_vectors = []
        save_outcomes = []

    #get the initial model outcome for the initial splitting scores
    nmisc = {}
    nmisc.update(misc)
    nmisc['otu_vector'] = current_otu_vectors[0]
    functions = [function_from_vector(current_otu_vectors[0], y, "initial run", nclasses, model,
                                      settings, nmisc)]
    current_outcomes = []
    outcome_handler = analysis.OutcomeHandler(current_outcomes.append, ())
    analysis.paralel_mp_limited(functions, procs, outcome_handler)
    
    if saveall:
        save_vectors.append(current_otu_vectors[0])
        save_outcomes.append(current_outcomes[0])

    for iteration in range(niterations):
        print "ITERATION", iteration
        print_progress_report(current_outcomes[0])

        #get a list of functions to try out in paralel
        functions = []
        for vector_index in range(len(current_otu_vectors)):
            otu_vector = current_outcomes[vector_index]['otu_vector']
            #score the current otus for splitting, the larger the better for splitting
            split_scores, merge_scores, kill_scores = \
                get_scores(current_outcomes[vector_index], otu_vector, split_coefs, merge_coefs,
                           kill_coefs)
        
            for split_prop, merge_prop, kill_prop in props:
                functions.append(function_from_scores( (otu_vector, y, nclasses, model, settings, 
                                                        nmisc, split_scores, merge_scores, 
                                                        kill_scores, split_prop, merge_prop, 
                                                        kill_prop, thresholds, otu_to_seq, 
                                                        seq_to_otu)) )

        
        
        #run the functions in paralel, write the outcomes to a list
        outcomes = []
        outcome_handler = analysis.OutcomeHandler(outcomes.append, ())
        analysis.paralel_mp_limited(functions, procs, outcome_handler)

        #when picking the leads to follow next, we should consider those we already have
        outcomes += current_outcomes
        
        best_outcomes = sorted(outcomes, key = lambda outcome: outcome['accuracy'], reverse=True)
        current_outcomes = best_outcomes[:n_keep]
        current_otu_vectors = [outcome['otu_vector'] for outcome in current_outcomes]
            
        if saveall:
            save_outcomes.append(current_outcomes[0])
            save_vectors.append(current_otu_vectors[0])

    if saveall:
        return save_vectors, save_outcomes
    else:
        return current_otu_vectors[0], current_outcomes[0]

def function_from_scores(params):
    return (process_score_model, params)

def process_score_model(otu_vector, y, nclasses, model, settings, misc,
                        split_scores, merge_scores, kill_scores, split_prop,
                        merge_prop, kill_prop, thresholds, otu_to_seqs, seqs_to_otu):
    
    rec_list = otu_vector.get_record_list()
    split_indecies, merge_indecies, kill_indecies = choose_indecies(rec_list, split_scores, 
                                                                    merge_scores, 
                                                                    kill_scores, split_prop, 
                                                                    merge_prop, kill_prop,
                                                                    thresholds)
    
    new_vector = apply_split_merge_kill(otu_vector, split_indecies, merge_indecies, 
                                        kill_indecies, thresholds, otu_to_seqs, seqs_to_otu)
    
    #using the new vector, make a new job to run in paralel
    nmisc = {}
    nmisc.update(misc)
    nmisc['otu_vector'] = new_vector
    nmisc['props'] = (split_prop, merge_prop, kill_prop)
    function = function_from_vector(new_vector, y, "score model", nclasses, model, settings, nmisc)

    return function[0](*function[1])

def apply_split_merge_kill(otu_vector, split_indecies, merge_indecies, kill_indecies, 
                           thresholds, otu_to_seqs, seqs_to_otu):
    pop_list = []
    pop_list += [(i, 1) for i in split_indecies]
    pop_list += [(i, -1) for i in merge_indecies]
    pop_list += [(i, None) for i in kill_indecies]
    pop_list = sorted(pop_list, reverse=True)
    
    #make a new otu vector to alter, split the otus
    new_vector = otu_vector.get_copy()
    
    #since splitting involves poping the otus off, we need to do so in reverse order to
    #preserve indecies
    pop_otus = [(new_vector.pop_otu(t[0]), t[1]) for t in pop_list]
    for otu_rec, t_change in pop_otus:
        if t_change != None:
            split_otu_rec(otu_rec, new_vector, thresholds, t_change, otu_to_seqs,
                          seqs_to_otu, partial = False)
            
    return new_vector

def choose_indecies(rec_list, split_scores, merge_scores, kill_scores, split_prop, merge_prop, 
                    kill_prop, thresholds):
    #get a random sampling of otus to split using the scores
    split_exclude_list = [i for i in range(len(rec_list)) if rec_list[i].get_threshold()
                          == thresholds[-1]]
    split_indecies = sample_from_scores(split_scores, split_prop, split_exclude_list)
    
    merge_exclude_list = [i for i in range(len(rec_list)) if rec_list[i].get_threshold() == 
                          thresholds[0]] + list(split_indecies)
    merge_indecies = sample_from_scores(merge_scores, merge_prop, merge_exclude_list)
    
    kill_exclude_list = list(split_indecies) + list(merge_indecies)
    kill_indecies = sample_from_scores(kill_scores, kill_prop, kill_exclude_list)
    
    return split_indecies, merge_indecies, kill_indecies

def get_scores(outcome, otu_vector, split_coefs, merge_coefs, kill_coefs):
    rec_list = otu_vector.get_record_list()

    populations = np.array([record.get_misc()['pop'] for record in rec_list])
    
    coefs = []
    if len(outcome['coef']) != len(rec_list):
        coefs = outcome['coef'][0]
        for c in outcome['coef'][1:]:
            coefs += c
        coefs /= len(outcome['coef'])
    else:
        coefs = outcome['coef']
#    remove_proportion(kill_prop, otu_vector, coefs, populations)

    p_scores = std_dev_dists(populations)
    c_scores = std_dev_dists(coefs)
    
    scores = [(p_scores[i], c_scores[i]) for i in range(len(rec_list))]
    
    #return a list of the dot products of scores and coefs, unless the record is excluded in which
    #case fill in None
    def dot_prod(coefs, scores, rec_list):
        ret = []
        for x in range(len(rec_list)):
            s = sum([coefs[i] * scores[x][i] for i in range(len(coefs))])
            ret.append(s)
        return ret

    split_scores = dot_prod(split_coefs, scores, rec_list)
    merge_scores = dot_prod(merge_coefs, scores, rec_list)
    kill_scores = dot_prod(kill_coefs, scores, rec_list)

    return split_scores, merge_scores, kill_scores

def std_dev_dists(scores):
    avg = np.mean(scores)
    std = np.std(scores)
    
    dists = [(s-avg)/std for s in scores]
    return dists

def sample_from_scores(scores, prop, exclude_list):
    #make positive and normalize
    mini = min(scores)
    if float(mini) < 0:
        scores -= mini

    for e in exclude_list:
        scores[e] = 0

    sumi = sum(scores)
    if float(sumi) != 0:
        scores /= sumi
    else: #since positive, all elements must be 0
        return []
    
    np.random.seed() #necessary for mutlithreaded code
    sample = np.random.choice(len(scores), int(len(scores)*prop), p=scores, replace=False)
    return sample

def nonrandom_from_scores(scores, otu_vector, cutoff = None, prop = None, thresh = 1):
    rec_list = otu_vector.get_record_list()

    populations = np.array([record.get_misc()['pop'] for record in rec_list])
    
    coefs = []
    if len(outcomes[0]['coef']) != len(rec_list):
        coefs = outcomes[0]['coef'][0]
        for c in outcomes[0]['coef'][1:]:
            coefs += c
            coefs /= len(outcomes[0])
    else:
        coefs = outcomes[0]['coef']

    avg = np.mean(scores)
    std = np.std(scores)
    split = []
    if cutoff != None or prop != None:
        best = sorted([(scores[i], i) for i in range(len(scores)) if 
                       rec_list[i].get_threshold() != '0.99'], key=lambda s:-s[0])
        if cutoff != None:
            split = [elem[1] for elem in best[:cutoff]]
        else:
            
            split = [elem[1] for elem in best[:int(len(best) * prop)]]
    else:
        split = [i for i in range(len(rec_list)) if rec_list[i].get_threshold() != '0.99' and
                 scores[i] >= avg+thresh*std]
                            
    #print "splitting", len(split), "/", len(rec_list)

    return split

def function_from_vector(otu_vector, y, name, nclasses, model, settings, misc):
    nmisc = {}
    nmisc.update(misc)
    nmisc['otu_vector'] = otu_vector

    rarify = 500.0
    cp = np.copy(otu_vector.sampletable_arr)
    for r_index in range(len(cp)):
        row = cp[r_index]
        s = rarify / np.sum(row)
        cp[r_index] /= s

    problem = Problem(otu_vector.sampletable_arr, y, nclasses, name, nmisc)
    function = (analysis.process, (model, problem, settings, outcome_maker))
    return function

def split_otu_rec(otu_rec, otu_vector, thresholds, threshold_change, otu_to_seqs, seqs_to_otu,
                  partial = False):
    otu_id = otu_rec.get_ID()
    old_threshold = otu_rec.get_threshold()
    
    #assume that we haven't been told to split a max-threshold otu?
    new_threshold = thresholds[thresholds.index(old_threshold)+threshold_change] #1 to split, -1 to merge
    sequences = otu_to_seqs[old_threshold][otu_id]

    new_otus = otus_from_sequences(sequences, seqs_to_otu[new_threshold], otu_to_seqs[new_threshold],
                                   partial = partial)

    for new_otu in new_otus.keys():
        seq_list = new_otus[new_otu]
        new_otu_rec = OtuRecord(new_otu, new_threshold, parents = [otu_rec],
                                misc={'pop':len(seq_list)})
        otu_rec.get_children().append(new_otu_rec)
        #99.5% of time being spent in add_otu
        otu_vector.add_otu(new_otu_rec, seq_list, partial = partial)

        
#gets a map of each otu found to a list of that otu's sequences.
#if partial=True, only sequences included in the original sequence list are included.
def otus_from_sequences(sequences, seqs_to_otus, otu_to_seqs, partial = False):
    otus = {}
    for sequence in sequences:
        #there might have been a clustering failure at this threshold level, in which case the
        #sequence might not be in our lookup table
        try:
            otu = seqs_to_otus[sequence]
            if partial:
                #only add the sequence that found the otu
                if not otu in otus.keys():
                    otus[otu] = []
                otus[otu].append(sequence)
            else:        
                #add all of the otu's sequences
                if not otu in otus.keys():
                    otus[otu] = otu_to_seqs[otu]
        except KeyError:
            continue
    return otus

def outcome_maker(problem, model, settings, predictions, duration = None,
                  importances = None):
    result = {}

    result['rarefaction'] = problem.getMisc()['rarefaction']
    result['n'] = problem.getMisc()['n']
    result['model_name'] = model.getName()
    result['problem_name'] = problem.getName()
    if duration != None:
        result['duration'] = duration
    if importances != None:
        result['importances'] = importances
    if hasattr(model.getModel(), 'coef_'):
        result['coef'] = model.getModel().coef_
    elif hasattr(model.getModel(), 'feature_importances_'):
        result['coef'] = model.getModel().feature_importances_
    result['accuracy'] = analysis.get_average_list_accuracy(problem.getY(), predictions)
    result['predictions'] = predictions
    if 'otu_vector' in problem.getMisc():
        result['otu_vector'] = problem.getMisc()['otu_vector']
    result.update(problem.getMisc())

    return result

def print_progress_report(current_outcome):
    otu_vector = current_outcome['otu_vector']
    rec_list = otu_vector.get_record_list()
    t_list = [record.get_threshold() for record in rec_list]
    t_set = set(t_list)
    dist = [(thresh, t_list.count(thresh)) for thresh in sorted(list(t_set))]
    print '\t Accuracy:', current_outcome['accuracy']
    print '\t Distribution:', dist
    if 'props' in current_outcome.keys():
        print '\t Action proportions:', current_outcome['props']

if __name__=="__main__":
    main()
