from utils import *
import numpy as np
import os.path

def parse_object_name(obj):
    underscore_index = obj.rfind('_')
    return obj[:underscore_index]

def readMappingFile(mapping_file):
    rows = [row for row in mapping_file]
    
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
