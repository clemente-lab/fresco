from utils import *
import numpy as np
import os.path

def getFileRows(filepath):
    if not os.path.exists(filepath):
        return None
    rows = open(filepath)
    rowList = [row for row in rows]
    rows.close()
    return rowList

def parse_sequence_name(seq):
    underscore_index = seq.rfind('_')
    return seq[:underscore_index]

def readMappingFile(filepath):
    rows = getFileRows(filepath)
    
    if rows == None:
        return None

    #first line describes the columns
    #ignore first letter (#)
    header = rows[0][1:]
    headerfields = header.split("\t")

    samplemap = {}
    
    for row in rows[1:]:
        fieldmap = {}
        fields = row.split("\t")
        for i in range(1, len(fields)):
            fieldmap[headerfields[i]] = fields[i]
        samplemap[fields[0]] = fieldmap
    
    return samplemap

#hold out a proportion of the samples
def readOTUTableTXT(filepath, holdout=0):
    rows = getFileRows(filepath)
    if rows == None:
        return None

    header = rows[1]
    #get the sample names, ignore the OTU ID and taxonomy string headers
    samples = tuple(header[1:].split("\t")[1:-1]) #[2:-1]?

    #ordered list of otu names
    otunames = []
    #ordered list of list of otu frequencies
    otus = []
    for row in rows[2:]:
        fields = row.split("\t")
        frequencies = []
        for field in fields[1:-1]:
            frequencies.append(float(field))

        #list of tuples of otu names and sample frequencies
        otus.append( tuple(frequencies) )
        otunames.append( (fields[0], fields[-1]) )

    #list of (sample name, otu frequncy list)
    sampletable = []
    for i in range(len(samples)):
        row = []
        for otu in otus:
            row.append(otu[i])
        sampletable.append( (samples[i], row) )

    if holdout <  0:
        #hold out the end
        sampletable = sampletable[:int(len(sampletable) * (1 + holdout))]
    elif holdout > 0:
        sampletable = sampletable[int(len(sampletable) * (holdout)):]

    return sampletable, otunames

def readOTUTableBIOM(filepath):
    f = open(filepath, "rb")
    content = f.read()
    content = content.replace("null", "None")
    biommap = eval(content)
    f.close()

    samples = []
    for sample in biommap["columns"]:
        samples.append(sample["id"])

    otunames = []
    for otu in biommap["rows"]:
        otunames.append( (otu["id"], otu["metadata"]) )

    sampletable = []
    for sample in samples:
        sampletable.append( (sample, [[0] * len(otunames)]) )

    for otu_n, sample_n, value in biommap["data"]:
        if otu_n >= len(otunames):
            print "OTU INDEX", otu_n, "OUT OF RANGE!", len(otunames)
        if sample_n >= len(samples):
            print "OTU INDEX", sample_n, "OUT OF RANGE!", len(samples)
        print "sampletable[", sample_n, "/", len(samples), "][1][", otu_n, "/", len(otunames), "]"
        print "sampletable[sample_n]=", sampletable[sample_n]
        sampletable[sample_n][1][otu_n] = value

    return sampletable, otunames

def readOTUTableCSV(filepath):
    rows = getFileRows(filepath)
    
    header = rows[0]
    #get the sample names, ignore the OTU ID and taxonomy string headers
    samples = tuple(header.split(",")[1:])

    #ordered list of otu names
    otunames = []
    #ordered list of tuples of otu frequencies
    otus = []
    for row in rows[1:]:
        fields = row.split(",")
        frequencies = []
        for field in fields[1:]:
            frequencies.append(float(field))

        #list of tuples of otu names and sample frequencies
        otus.append( tuple(frequencies) )
        otunames.append( fields[0] )

    #list of (sample name, otu frequncy list)
    sampletable = []
    for i in range(len(samples)):
        row = []
        for otu in otus:
            row.append(otu[i])
        sampletable.append( row )
    sampletable = np.array(sampletable)

    #don't need otunames?
    return sampletable, samples, otunames
    
    

def buildXy(samplemap, sampletable, predictfield, classmap):
    y = []
    X = []

    for samplename, frequencies in sampletable:
        if not samplename in samplemap.keys():
            continue
        sample = samplemap[samplename]
        pclass = sample[predictfield]
        if not pclass in classmap.keys():
            continue
        nclass = classmap[pclass]
        y.append(nclass)

        X.append(frequencies)

    return np.array(X), np.array(y)

def get1MProblem(otutable = "data/study_850_closed_reference_otu_table.1M.txt",
                     mappingfile = "data/study_850_mapping_file.txt"):
    problem = Problem()

    problem.setName("(OTU table: "+otutable+")")

    samplemap = readMappingFile(mappingfile)
    sampletable, otunames = readOTUTableTXT(otutable)

    predictfield = 'COUNTRY'
    classmap = {'GAZ:Venezuela':0, 'GAZ:United States of America':1, 'GAZ:Malawi':2}
    #classmap = {'GAZ:Venezuela':0, 'GAZ:Malawi':1}

    X, y = buildXy(samplemap, sampletable, predictfield, classmap)

    problem.setX(X)
    problem.setY(y)

    problem.setNClasses(len(classmap.keys()))

    return problem

def getBPProblem():
    problem = {}
    otutable = "2049-2618-1-11-s6\BP.csv"

    problem['name'] = "(OTU Table: " + otutable + ")"
    
    problem['X'], classes, problem['otunames'] = readOTUTableCSV(otutable)
    y = []
    for c in classes:
        c = c.replace("\n", "")
        if c == "Psoriasis Lesion":
            y.append(1)
        else:
            y.append(0)
        #elif c == "Psoriasis Normal":
        #    y.append(1)
        #elif c == "Control":
        #    y.append(2)
        #else:
        #    print c
    problem['y'] = np.array(y)

    problem['nclasses'] = 2
    
    return problem
