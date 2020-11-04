from abc import ABC, abstractmethod
from simufit.Types import DistributionType
 

class IDistribution(ABC):
    """The abstract base class for all distribution types in Simufit"""

    def __init__(self):        
        super().__init__()    
    
    # Distribution Parameters
    @abstractmethod
    def setSeed(self, seed):
        pass

    @abstractmethod
    def getSeed(self):
        pass

    @abstractmethod
    def setRange(self, min, max):
        pass

    @abstractmethod
    def getRange(self):
        pass

    @abstractmethod
    def setSamples(self, samples):
        pass

    @abstractmethod
    def readCsv(self, filename):
        pass

    @abstractmethod
    def getSamples(self):
        pass

    @abstractmethod
    def setDistribution(self, distribution_type):
        pass

    @abstractmethod
    def setRandomDistribution(self):
        pass

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def printReport(self):
        pass

    # Statistical Methods
    @abstractmethod
    def getMedian(self):
        pass

    @abstractmethod
    def getExpectedValue(self):
        pass

    @abstractmethod
    def getVariance(self):
        pass

    @abstractmethod
    def getStandardDeviation(self):
        pass

    @abstractmethod
    def getCoefficientOfVariation(self):
        pass

    @abstractmethod
    def getLexisRatio(self):
        pass


    # Descriptive Properties
    @abstractmethod
    def isDiscrete(self):
        pass

    @abstractmethod
    def isContinuous(self):
        pass    

    # Graphical Methods
    @abstractmethod
    def drawHistogram(self):
        pass

    @abstractmethod
    def drawHistogram(self, dataset):
        pass

    @abstractmethod
    def drawScatterPlot(self):
        pass

    @abstractmethod
    def drawScatterPlot(self, dataset):
        pass

    @abstractmethod
    def drawLinePlot(self):
        pass

    @abstractmethod
    def drawLinePlot(self, dataset):
        pass

    @abstractmethod
    def drawBoxPlot(self):
        pass

    @abstractmethod
    def drawBoxPlot(self, dataset):
        pass

    @abstractmethod
    def drawDistributionFunctionDifferences(self, distribution_type):
        pass

    @abstractmethod
    def drawQQPlot(self, dataset):
        pass

    @abstractmethod
    def drawPPPlot(self, dataset):
        pass    

    # Fitting Methods
    @abstractmethod
    def fit(self, samples):
        pass

    @abstractmethod
    def identifyDistribution(self):
        pass
