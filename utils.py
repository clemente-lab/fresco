import copy

class Problem:
    def __init__(self, X, y, nclasses = 0, name = "", misc = {}):
        self.y = y
        self.X = X
        self.nclasses = nclasses
        self.name = name
        self.misc = misc

    def getMisc(self):
        return self.misc

    def getX(self):
        return self.X

    def getY(self):
        return self.y

    def getNClasses(self):
        return self.nclasses

    def getName(self):
        return self.name

    def setNClasses(self, nclasses):
        self.nclasses = nclasses

    def setX(self, X):
        self.X = X

    def setY(self, y):
        self.y = y

    def setName(self, name):
        self.name = name

class Model:
    def __init__(self, model = None, name = "", iterations = 1, featureRank = False, misc = {}):
        self.model = model
        self.name = name
        self.iterations = iterations
        self.featureRank = featureRank
        self.misc = misc

    def __deepcopy__(self, memo):
        new = Model(name = self.name, iterations = self.iterations, featureRank = self.featureRank, misc = self.misc)
        new.model = copy.deepcopy(self.model, memo)
        return new

    def getMisc(self):
        return self.misc

    def getModel(self):
        return self.model

    def getIterations(self):
        return self.iterations

    def getFeatureRank(self):
        return self.featureRank

    def getName(self):
        return self.name

    def setX(self, model):
        self.model = model

    def setY(self, iterations):
        self.iterations = iterations

    def setFeatureRanks(self, featureRanks):
        self.featureRanks = featureRanks

    def setName(self, name):
        self.name = name
