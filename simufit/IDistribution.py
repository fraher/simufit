from abc import ABC, abstractmethod
from simufit.Types import DistributionType
 

class IDistribution(ABC):
    """The abstract base class for all distribution types in Simufit"""

    def __init__(self):        
        super().__init__()    

    # General Functions
    @abstractmethod
    def clearSamples(self):
        pass

    @abstractmethod
    def readCsv(self, filepath, skip_header=True, delimiter=','):
        pass

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def printReport(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    # Distribution Parameters
    @abstractmethod
    def setSeed(self, seed):
        pass

    @abstractmethod
    def getSeed(self):
        pass

    @abstractmethod
    def setSize(self, size):
        pass

    @abstractmethod
    def getSize(self):
        pass

    @abstractmethod
    def getRange(self):
        pass    

    @abstractmethod
    def generateSamples(self, **kwargs):
        pass
        
    @abstractmethod
    def setSamples(self, samples):
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
    def getMeasureType(self):
        pass   

    # Graphical Methods    
    @abstractmethod
    def drawHistogram(self, comparison_distribution):
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
    def MLE(self, **kwargs):
        pass

    @abstractmethod
    def identifyDistribution(self):
        pass
