from matplotlib import use
from scipy.stats import distributions
from simufit.IDistribution import IDistribution
from simufit.Types import DistributionType as dt
from simufit.Types import MeasureType as mt
from simufit.Report import DistributionReport as dr
import simufit.dist_generator as dg
import random as rand
import numpy as np
import inspect
import copy

from simufit.dist_generator import Bernoulli

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
        self._distribution_report = list()
        self._measureType = mt.UNKNOWN
        self._type = None
        self.Distribution = dt.UNKNOWN
        pass

    # General Functions
    def clearSamples(self):
        self._samples = None
        self._range = [None, None]

    def reset(self):
        self._seed = None
        self._range = [None, None]
        self._size = None
        self._samples = list()
        self._distribution_report = dr()
        self._measureType = mt.UNKNOWN
        self._type = None
        self.Distribution = dt.UNKNOWN


    def readCsv(self, filepath, skip_header=True, delimiter=','):
        """This method loads a collection of samples from a CSV file"""
        self._samples = np.genfromtxt(fname=filepath, delimiter=delimiter, skip_header=skip_header)
        self._range = [min(self._samples), max(self._samples)] # Update the range
        self._size = len(self._samples)
        self._type = dt.UNKNOWN
        self.Distribution = dg.Unknown()


    def display(self):
        """This method prints the parameters specific to the Distribution object"""
        print("Seed: ", self._seed)
        print("Range: ", self._range[0], "-", self._range[1])
        print("Size: ", self._size)
        print("Samples: ", self._samples)
        print("Measure Type: ", str(self._measureType).replace("MeasureType.",""))
        print("Distribution: ", str(self._type).replace("DistributionType.",""))

    def printReport(self):
        """This method displays the distribution fitting report"""                
        top_reports = []
        nan_reports = []
        bernoulli = None

        for report in self._distribution_report:            
            if report.isBernoulli():
                bernoulli = report
            elif np.isnan(report.getScore()):                
                nan_reports.append(report)
            else:
                top_reports.append(report)        

        top_reports.sort(key=lambda x: x.getScore(), reverse=True)        
        if bernoulli is not None:
            top_reports.insert(0, bernoulli)

        print('\n')
        [x.printReport() for x in top_reports]
        [x.printReport() for x in nan_reports]


    # Distribution Parameters
    def setSeed(self, seed):
        """This method stores a seed value for randomization"""
        if len(self._samples) > 0:
            print("Cannot change seed while there are samples in the object, must clear samples first.")
        self._seed = seed

    def getSeed(self):
        """This method returns the seed value set for randomization"""
        return self._seed

    def setSize(self, size):
        """This method stores a size value for randomization"""
        if len(self._samples) > 0:
            print("Cannot change size while there are samples in the object, must clear samples first.")
        self._size = size

    def getSize(self):
        """This method returns the size value set for randomization"""
        return self._size

    def getRange(self):
        """This method retrieves the range of values from the sample set"""
        return self._range

    def generateSamples(self, **kwargs):
        """This method loads a collection of samples and updates the seed, size and range as applicable"""

        if self._type is None:
            print('Set a distribution type first using the setDistribution method.')
            return

        # Use or Update Seed
        if 'seed' in kwargs:
            if self._seed is None:
                self._seed = kwargs.get('seed')

        if 'seed' not in kwargs and self._seed is not None:
            kwargs['seed'] = self._seed

        # Use or Update Size
        if 'size' in kwargs:
            if self._size is None:
                self._size = kwargs.get('size')

        if 'size' not in kwargs and self._size is not None:
            kwargs['size'] = self._size

        if self.Distribution.name == 'Unknown':
            if self._size is None:
                self._size = np.random.randint(1, 1000) # Generates up to 1000 samples
                kwargs['size'] = self._size

            random_distribution = rand.choice([x for x in list(dt) if x != dt.UNKNOWN])
            distribution = getattr(dg, str(random_distribution).replace('DistributionType.','').title())()
            print(random_distribution.name)

            # Generate values for required parameters
            min_range = None

            for parameter in distribution._parameters:
                label = parameter['label']
                value = None

                if 'probability' in parameter:
                    value = np.random.uniform(parameter['probability'][0], parameter['probability'][1])
                    kwargs[label] = value

                if 'mean' in parameter:
                    value = np.random.randint(parameter['mean'][0], parameter['mean'][1])
                    kwargs[label] = value

                if 'variance' in parameter:
                    value = np.random.randint(parameter['var'][0], parameter['var'][1])
                    kwargs[label] = value

                if 'span' in parameter:
                    value = np.random.randint(parameter['span'][0], parameter['span'][1])
                    kwargs[label] = value

                if 'range' in parameter:
                    if self.getRange() == [None, None]:
                        if min_range is None:
                            min_range = np.random.randint(parameter['range'][0], parameter['range'][1])
                            value = min_range
                            kwargs[label] = value
                        elif min_range is not None:
                            min_range = np.random.randint(min_range, parameter['range'][1])
                            value = min_range
                            kwargs[label] = value
                    else:
                        if min_range is None:
                            min_range = self.getRange()[0]
                            value = min_range
                            kwargs[label] = value
                        elif min_range is not None:
                            value = self.getRange()[1]
                            kwargs[label] = value

            self._samples = distribution.sample(**kwargs) # Update the samples
        else:

            self._samples = self.Distribution.sample(**kwargs) # Update the samples


        self._range = [min(self._samples), max(self._samples)] # Update the range

    def setSamples(self, samples):
        """This method loads a collection of samples"""
        self._samples = samples
        self._range = [min(self._samples), max(self._samples)] # Update the range

    def getSamples(self):
        """This method returns the current collection of generated samples"""
        return self._samples

    def setDistribution(self, distribution_type):
        """This method takes in a DistributionType and creates a new object
        based on classes defined in dist_generator"""
        if distribution_type == dt.GEOMETRIC:
            self.Distribution = dg.Geometric()

        if distribution_type == dt.UNIFORM:
            self.Distribution = dg.Uniform()

        if distribution_type == dt.NORMAL:
            self.Distribution = dg.Normal()

        if distribution_type == dt.EXPONENTIAL:
            self.Distribution = dg.Exponential()

        if distribution_type == dt.GAMMA:
            self.Distribution = dg.Gamma()

        if distribution_type == dt.BERNOULLI:
            self.Distribution = dg.Bernoulli()

        if distribution_type == dt.BINOMIAL:
            self.Distribution = dg.Binomial()

        if distribution_type == dt.WEIBULL:
            self.Distribution = dg.Weibull()

        if distribution_type is None or distribution_type == dt.UNKNOWN:
            self._type = dt.UNKNOWN
            self._measureType = mt.UNKNOWN
            self.Distribution = dg.Unknown()
        elif distribution_type is not None:
            self._type = distribution_type
            self._measureType = self.Distribution.measure_type

    def setRandomDistribution(self):
        """This method selects a random distribution of DistributionType
        and creates a new object based on classes defined in dist_generator"""
        self.setDistribution(rand.choice(list(dt)))

    # Statistical Methods
    def getMedian(self):
        """This method returns the median value of the sample set"""
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
    def getMeasureType(self):
        """This method returns whther the value for the loaded distribution
        identifying is Continuous or not"""
        # TODO: Write Method
        raise NotImplementedError

    # Graphical Methods
    def drawHistogram(self, bins=None, comparison_distribution=None):
        """This method opens a graph and displays one or two histograms, one of the sample set
        alongside a different dataset for comparison if dataset is selected"""
        self.Distribution.histogram(self._samples, bins, comparison_distribution)

    def drawScatterPlot(self):
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

    def MLE(self, **kwargs):
        """Run the MLE method for the loaded distribution"""

        if self._type is None:
            print('Set a distribution first using the setDistribution method.')
            return

        if self._type == dt.UNKNOWN:
            print("Must identify distribution type before performing MLE.")
            return

        if len(self._samples) == 0:
            print('Generate some samples using the generateSamples method first.')
            return

        result = []

        if len(kwargs) > 0:
            result = self.Distribution.MLE(self._samples, **kwargs)
        else:
            result = self.Distribution.MLE(self._samples)

        print(result) # Placeholder until set to report

    def GOF(self, **kwargs):
        """Run chi-square goodness of fit (GOF) method for the loaded distribution"""

        if self._type is None:
            print('Set a distribution first using the setDistribution method.')
            return

        if self._type == dt.UNKNOWN:
            print("Must identify distribution type before performing GOF.")
            return

        elif self._type == dt.BERNOULLI:
            print('No GOF test for Bernoulli-distributed samples.')
            return np.nan
            
        else:
            if len(kwargs) > 0:
                mle_params = self.Distribution.MLE(self._samples, **kwargs)
            else:
                mle_params = self.Distribution.MLE(self._samples)

            if self._type == dt.BINOMIAL:
                result = self.Distribution.GOF(self._samples, kwargs['n'], mle_params)
            else:
                result = self.Distribution.GOF(self._samples, *mle_params)

            print(result)

    def identifyDistribution(self, use_minimizer=None, a0=None, b0=None, mean0=None, var0=None, p0=None, lambd0=None, n=None):
        """Executes logic to identify the most likely distribution"""
        self._distribution_report = list()
        temp = copy.deepcopy(self)
        kwargs = {}

        # Perform MLE for all distribution types based on available and provided parameters
        for distribution_type in dt:
            kwargs.clear()

            if distribution_type not in [dt.UNKNOWN]:
                temp.setDistribution(distribution_type)
                method_args = inspect.getargspec(temp.Distribution.MLE).args
                valid = True

                if temp.Distribution.name.title() == dt.BINOMIAL.name.title():
                    if n is None:
                        valid = False
                    else:
                        kwargs['n'] = n

                report = dr()
                report.setDistributionType(temp.Distribution.name.title())
                print('Evaluating {}'.format(temp.Distribution.name.title()))
                print('-----------------------------------------------')
                if valid:
                    print('Starting...')
                    mle_result = temp.Distribution.MLE(samples=temp._samples, **kwargs)     

                    if temp.Distribution.name.title() == dt.BERNOULLI.name.title():
                        gof_result = 'No GOF for Bernoulli'
                    elif temp.Distribution.name.title() == dt.BINOMIAL.name.title():
                        gof_result = temp.Distribution.GOF(temp._samples, kwargs['n'], mle_result)
                    else:                        
                        gof_result = temp.Distribution.GOF(temp._samples, *mle_result)
                    try:
                        report.setMLE(mle_result)
                    except Exception as e:
                        report.setMLE('Error: {}'.format(e))
                        print('Error: {}'.format(e))

                    try:
                        report.setGOF(gof_result)
                    except Exception as e:
                        report.setGOF('Error: {}'.format(e))

                    print('Completed')
                else:
                    print('Distribution Skipped')
                    report.setMLE('Not Performed')
                    report.setGOF('Not Performed')
                print('-----------------------------------------------')
                print('\n\n')

                self._distribution_report.append(report)

        best_distribution = dt.UNKNOWN.name.title()
        best_score = np.nan  

        for report in self._distribution_report:
            report.evaluateScore(self._samples)
            if report.isPass():
                if report.isBernoulli() or best_distribution == dt.BERNOULLI.name.title():
                    best_distribution = dt.BERNOULLI.name                    
                
                if best_distribution != dt.BERNOULLI.name.title() and (best_score is np.nan or report.getScore() > best_score):
                    best_distribution = report.getDistributionType()                
                    best_score = report.getScore()
        
        if best_distribution is not dt.UNKNOWN.name:
            print(best_distribution)            
            print(best_score)            
            self.setDistribution(best_distribution)
        else:            
            print ('Could not identify distribution.')

        