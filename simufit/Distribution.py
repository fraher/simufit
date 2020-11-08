from simufit.IDistribution import IDistribution
from simufit.Types import DistributionType as dt
from simufit.Report import DistributionReport as dr
import simufit.dist_generator as dg
import random as rand

class Distribution(IDistribution):
    """The Distribution class contains a stateful collection of parameters which
    define a distribution for analysis. This includes information about the seed for 
    sample generation, range, size, the sample collection generated. Reports and additional
    parameters. It interfaces with functionality within the dist_generator."""
    
    def __init__(self):                
        self._seed = None
        self._range = [None, None]
        self._size = None
        self._samples = list()        
        self._distribution_report = dr()     
        self._continuous = None
        self._discrete = None   
        self._type = None
        self.Distribution = dt.UNKNOWN
        pass

    # Distribution Parameters    
    def setSeed(self, seed):
        """This method stores a seed value for randomization"""
        self._seed = seed
    
    def getSeed(self):
        """This method returns the seed value set for randomization"""
        return self._seed
    
    def setRange(self, min, max):
        """This method sets the range of values for randomization"""
        self._range = [min, max]
    
    def getRange(self):
        """This method retrieves the range of values for randomization"""
        return self._range
    
    def setSamples(self, samples):        
        """This method loads a collection of samples"""        
        self._samples = samples

    def getSamples(self):
        """This method returns the current collection of generated samples"""
        return self._samples
    
    def readCsv(self, filename):
        """This method loads a collection of samples from a CSV file"""
        # TODO: Write Method
        raise NotImplementedError

    def setDistribution(self, distribution_type):
        """This method takes in a DistributionType and creates a new object 
        based on classes defined in dist_generator"""

        if distribution_type == dt.GEOMETRIC:            
            self._type = distribution_type
            self.Distribution = dg.Geometric()

        if distribution_type == dt.UNIFORM:
            self._type = distribution_type
            self.Distribution = dg.Uniform()
        
        if distribution_type == dt.NORMAL:
            self._type = distribution_type
            self.Distribution = dg.Normal()

        if distribution_type == dt.EXPONENTIAL:
            self._type = distribution_type
            self.Distribution = dg.Exponential()
        
        if distribution_type == dt.GAMMA:
            self._type = distribution_type
            self.Distribution = dg.Gamma()
        
        if distribution_type == dt.BERNOULLI:
            self._type = distribution_type
            self.Distribution = dg.Bernoulli()
            
        if distribution_type == dt.BINOMIAL:
            self._type = distribution_type
            self.Distribution = None # TODO: Add dg.Binomial()
        
        if distribution_type == dt.BERNOULLI:
            self._type = distribution_type
            self.Distribution = dg.Weibull()

        if distribution_type == None:            
            self._type = dt.UNKNOWN
                        
    def setRandomDistribution(self):
        """This method selects a random distribution of DistributionType 
        and creates a new object based on classes defined in dist_generator"""

        self.setDistribution(rand.choice(list(dt)))

    def display(self):
        """This method prints the parameters specific to the Distribution object"""

        print("Seed: ", self._seed)
        print("Range: ", self._range[0], "-", self._range[1])
        print("Size: ", self._size)
        print("Samples: ", self._samples)
        print("Continuous: ", self._continuous)
        print("Discrete:", self._discrete)
        print("Distribution: ", self._type)

    def printReport(self):
        """This method displays the distribution fitting report"""
        dr.printReport()

    # Statistical Methods    
    def getMedian(self):
        """This method returns the median value of the sample set"""
        # TODO: Write Method
        raise NotImplementedError

    def getExpectedValue(self):
        """This method returns the expected value of the sample set"""
        # TODO: Write Method
        raise NotImplementedError
    
    def getVariance(self):
        """This method returns the variance of the sample set"""
        # TODO: Write Method
        raise NotImplementedError
    
    def getStandardDeviation(self):
        """This method returns the standard deviation of the sample set"""
        # TODO: Write Method
        raise NotImplementedError
    
    def getCoefficientOfVariation(self):
        """This method returns the coefficient of variation of the sample set"""
        # TODO: Write Method
        raise NotImplementedError
    
    def getLexisRatio(self):
        """This method returns the Lexis Ratio of the sample set"""
        # TODO: Write Method
        raise NotImplementedError

    # Descriptive Properties    
    def isDiscrete(self):
        """This method returns the True/False value for the loaded distribution 
        identifying whether it is Discrete or not"""
        # TODO: Write Method
        raise NotImplementedError
    
    def isContinuous(self):
        """This method returns the True/False value for the loaded distribution 
        identifying whether it is Continuous or not"""
        # TODO: Write Method
        raise NotImplementedError

    # Graphical Methods    
    def drawHistogram(self):
        """This method opens a graph and displays a histogram of the sample set"""
        # TODO: Write Method
        raise NotImplementedError
    
    def drawHistogram(self, dataset):
        """This method opens a graph and displays two histograms, one of the sample set
        alongside a different dataset for comparison"""
        # TODO: Write Method
        raise NotImplementedError
    
    def drawSatterPlot(self):
        """This method opens a graph and displays a scatter plot of the sample set"""
        # TODO: Write Method
        raise NotImplementedError
    
    def drawScatterPlot(self, dataset):
        """This method opens a graph and displays two scatter plots, one of the sample set
        alongside a different dataset for comparison"""
        # TODO: Write Method
        raise NotImplementedError
    
    def drawLinePlot(self):
        """This method opens a graph and displays a line plot of the sample set"""
        # TODO: Write Method
        raise NotImplementedError
    
    def drawLinePlot(self, dataset):
        """This method opens a graph and displays two line plots, one of the sample set
        alongside a different dataset for comparison"""
        # TODO: Write Method
        raise NotImplementedError
    
    def drawBoxPlot(self):
        """This method opens a graph and displays a box plot of the sample set"""
        # TODO: Write Method
        raise NotImplementedError
    
    def drawBoxPlot(self, dataset):
        """This method opens a graph and displays two box plots, one of the sample set
        alongside a different dataset for comparison"""
        # TODO: Write Method
        raise NotImplementedError
    
    def drawDistributionFunctionDifferences(self, distribution_type):
        """This method opens a graph and displays a distribution-function-difference 
        plot of the sample set"""
        # TODO: Write Method
        raise NotImplementedError
    
    def drawQQPlot(self, dataset):
        """This method opens a graph and displays a QQ plot of the sample set"""
        # TODO: Write Method
        raise NotImplementedError
    
    def drawPPPlot(self, dataset):
        """This method opens a graph and displays a PP plot of the sample set"""
        # TODO: Write Method
        raise NotImplementedError

    # Fitting Methods
    
    def fit(self):
        """Run the PyQt/MPL visualization for the loaded distribution"""
        self.Distribution.fit(self._samples)
    
    def identifyDistribution(self):
        """Executes logic to identify the most likely distribution"""
        # TODO: Write Method
        raise NotImplementedError

