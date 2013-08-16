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
    def __init__(self, model = None, name = "", iterations = 1, misc = {}):
        self.model = model
        self.name = name
        self.iterations = iterations
        self.misc = misc

    def getMisc(self):
        return self.misc

    def getModel(self):
        return self.model

    def getIterations(self):
        return self.iterations

    def getName(self):
        return self.name

    def setX(self, model):
        self.model = model

    def setY(self, iterations):
        self.iterations = iterations

    def setName(self, name):
        self.name = name
