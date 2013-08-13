from parse import *

def build_classmap(samplemap, predictfield):
    class_set = set([samplemap[samplename][predictfield]
                     for samplename in samplemap.keys()])
    classes = list(class_set)
    classmap = {}
    for i in range(len(classes)):
        classmap[classes[i]] = i

    return classmap

#prules = [(key, value), (key, value)..] that must be true of a record to include it                  
#nrules = [(key, value), (key, value)..] that must be false of a record to include it                 
def build_txt_problems(otutables, mappingfile, predictfields, miscs, prules = None, nrules = None, rem_otu = None):
    problems = []

    samplemap = readMappingFile(mappingfile)
    if prules == None:
        prules = [None for p in predictfields]
    if nrules == None:
        nrules = [None for p in predictfields]

    for i in range(len(otutables)):
        otutable = otutables[i]

        res = readOTUTableTXT(otutable)
        if res == None:
            continue
        sampletable, otunames = res

        for p in range(len(predictfields)):
            predictfield = predictfields[p]

            #predsampletable = sampletable
            #if (prules != None and prules[p] != None) or (nrules != None and nrules[p] != None):
            #    predsampletable = prune_samples(samplemap, sampletable, prules, nrules)

            classmap = build_classmap(samplemap, predictfield)

            name="OTU Table: \""+otutable+"\". Predict field: \""+predictfield+"\"."

            #X, y = buildXy(samplemap, predsampletable, predictfield, classmap)                       
            #problems.append( Problem(X=X, y=y, name="OTU Table: \""+otutable+"\". Predict field: \""\+predictfield+"\".", nclasses = len(classmap), misc = miscs[i]) )                                     
            if rem_otu == None:
                rem_otu = [None]
            for ro in rem_otu:

                #we will build the specification for a problem to build later and save memory         
                problem_spec = {
                    'samplemap':samplemap,
                    'sampletable':sampletable,
                    'rem_otu':ro,
                    'prules':prules[p],
                    'nrules':nrules[p],
                    'predictfield':predictfield,
                    'classmap':classmap,
                    'name':name,
                    'nclasses':len(classmap),
                    'misc':miscs[i]
                    }

                problems.append(problem_spec)

    return problems


def build_problem(samplemap, sampletable, rem_otu, prules, nrules, predictfield, classmap, name, nclasses, misc):
    pred_sampletable = prune_samples(samplemap, sampletable, prules, nrules)
    X, y = buildXy(samplemap, pred_sampletable, predictfield, classmap)
    X, misc = remove_otu(X, rem_otu, misc)
    problem = Problem(X, y, nclasses, name, misc)
    return problem

def build_problem_small(x, y, name, nclasses, misc):
    problem = Problem(X, y, nclasses, name, misc)

def prune_samples(samplemap, sampletable, prules, nrules):
    if prules == None and nrules == None:
        return sampletable

    nsampletable = []
    for row in sampletable:
        sample = samplemap[row[0]]
        add = True
        #if a prule is false in a record, remove it
        if prules != None:
            for rule in prules:
                if sample[rule[0]] != rule[1]:
                    add = False
                    break
        #if a nrule is true in a record, remove it
        if nrules != None and add:
            for rule in nrules:
                if sample[rule[0]] != rule[1]:
                    add = False
                    break
        if add:
            nsampletable.append( row )

    return nsampletable

def remove_otu(X, otu, misc):
    if otu == None:
        return X, misc

    new_X = np.delete(X, otu, axis = 1)

    new_misc = {}
    new_misc.update(misc)
    new_misc["OTU Removed"] = otu

    return new_X, new_misc

def read_split_file(split_filename):
    otu_to_seq = {}
    seq_to_otu = {}

    f = open(split_filename)
    while True:
        line = f.readline()
        if line == '':
            break
        entries = line.split('\t')
        otu_to_seq[entries[0]] = entries[1:]
        for seq in entries[1:]:
            seq_to_otu[seq] = entries[0]
 
    return otu_to_seq, seq_to_otu

def get_449_dataset(rootdir, mappingfile = "study_449/study_449_mapping_file.txt",
                    foldername = "study_449", study_number = "449"):
    return Dataset(rootdir, mappingfile, foldername, study_number)

def get_626_dataset(rootdir, mappingfile = "study_626/study_626_mapping_file.txt",
                    foldername = "study_626", study_number = "626"):
    return Dataset(rootdir, mappingfile, foldername, study_number)

def get_550_dataset(rootdir, mappingfile = "study_550/study_550_mapping_file.txt",
                    foldername = "study_550", study_number = "500"):
    return Dataset(rootdir, mappingfile, foldername, study_number)

class Dataset:
    def __init__(self, rootdir, mappingfile, foldername, study_number):
        self.study_number = study_number
        self.rootdir = rootdir
        self.mappingfile = mappingfile
        self.foldername = foldername

    def get_mappingfile(self):
        return self.rootdir + self.mappingfile

    def get_otu_table(self, threshold, rarefaction, n):
        if rarefaction != None:
            return self.rootdir + self.foldername + "/ucrc_"+str(threshold)+"/rarefied_tables/"+str(rarefaction)+"/otu_table."+str(rarefaction)+"."+str(n)+".biom.txt"
        else:
            return self.rootdir + self.foldername + "/ucrc_"+str(threshold)+"/uclust_ref_picked_otus/otu_table.biom.txt"
    def get_otu_table_grid(self, rarefactions, thresholds, ns):
        otutables = []
        miscs = []
        for thresh in thresholds:
            for rarefaction in rarefactions:
                for n in ns:
                    miscs.append( {"theshold":thresh, "rarefaction":rarefaction, "n":n} )
                    otutable = self.get_otu_table(thresh, rarefaction, n)
                    otutables.append(otutable)
                    
        return otutables, miscs

    def get_split_file(self, theshold):
        return self.rootdir + self.foldername + "/ucrc_" + theshold + \
            "/uclust_ref_picked_otus/study_" + self.study_number + "_split_library_seqs_otus.txt"


def build_sample_matrix(samplename_map, rec_list, otu_to_seq, scale):
    def scale_table(table):
        if scale == None:
            return table
        cp = np.copy(table)
        for r_index in range(len(cp)):
            row = cp[r_index]
            s = scale / np.sum(row)
            cp[r_index] /= s
        return cp

    mat = np.zeros( (len(samplename_map.keys()), len(rec_list)) )
    for rec_index in range(len(rec_list)):
        record = rec_list[rec_index]
        otu_id = record.get_ID()
        seqs = otu_to_seq[record.get_threshold()][otu_id]
        for seq in seqs:
            sample_id = parse_sequence_name(seq)
            try:
                sample_index = samplename_map[sample_id]
                mat[sample_index][rec_index] += 1
            except KeyError:
                continue
    return mat
                
