import numpy as np
from numpy.testing._private.utils import measure
from simufit.Types import DistributionType as dt
import simufit.dist_generator as dg
from simufit.Types import MeasureType as mt
class DistributionReport():
    """The Report class provides contains a full analysis of the
    dataset evaluated."""

    def __init__(self):
        self._distribution_type = None
        self._mle = None
        self._gof = None
        self._pass = False
        self._score = np.nan
        self._unique_elements = 0
        self._bernoulli = False
        self._measure_type = None
        self._measure_type_match = False

    def setDistributionType(self, distribution_type):
        """Sets the Distribution Type of the Report"""
        self._distribution_type = distribution_type
    
    def setMLE(self, mle):
        """Sets the Maximum Likelihood Expression for the Reported Distribution"""
        self._mle = mle

    def setGOF(self, gof):
        """Sets the Goodness of Fit value for the Reported Distribution"""
        self._gof = gof

    def getDistributionType(self):
        """Returns the Distribution Type of the Report"""
        return self._distribution_type

    def getScore(self):
        """Returns the Comparison Score of the Goodness of Fit Evaluation"""
        return self._score    

    def isPass(self):
        """Returns a TRUE/FALSE value indicating if the Goodness of Fit Evaluation passed."""
        return self._pass

    def isBernoulli(self):
        """Returns a TRUE/FALSE value indicating if the distribution is Bernoulli."""
        return self._bernoulli

    def isDiscrete(self):
        """Returns a TRUE/FALSE value indicating if this distributino is discrete."""
        return self._discrete

    def isMeasureTypeMatch(self):
        """Returns a TRUE/FALSE value indicating if the distribution matches the sample set measure type of Discrete or Continuous"""
        return self._measure_type_match

    def getMeasureType(self):
        return self._measure_type

    def evaluateDistribution(self, samples):
        """Calculates the proportional distance of a given Goodness of Fit value for a given evaluated
        distribution and then is mapped to an inverse exponential to generate a score."""
        self._unique_elements = len(np.unique(samples))
        
        if np.all(samples - samples.astype(np.int32) == 0):
            self._measure_type = mt.DISCRETE
        else:
            self._measure_type = mt.CONTINUOUS                

        measure_type = eval('dg.{}'.format(self._distribution_type))().measure_type
        self._measure_type_match = measure_type == self._measure_type
        
        if self._distribution_type == dt.BERNOULLI.name.title():
            
            if self._unique_elements == 2:
                self._pass = True
                self._bernoulli = True
        np.seterr('ignore')
        if self._gof is not None and self._score != 1:
            if type(self._gof) is not str:
                if self._gof[0] != np.nan and self._gof[1] != np.nan and self._gof[0] is not None and self._gof[1] is not None:                                           
                    self._score = 1/np.exp((self._gof[0] - self._gof[1])/self._gof[1])
                    if self._gof[0] < self._gof[1]:
                        self._pass = True            

    def printReportHeader(self):
        print('EVALUATION REPORT:\n')
        print('Type Detected: ', str(self._measure_type).replace('MeasureType.',''))
        print('Unique Elements: ', str(self._unique_elements))
        print('=============')

    def printReport(self):
        """Displays all Distribution Report Items collected 
        during the Simufit analysis."""        
        
        print('Distribution: ', self._distribution_type)
        print('Distribution Type: ', str(self._measure_type).replace('MeasureType.',''))        
        print('Type Detection Match: ', str(self._measure_type_match))
        print('MLE: ', str(self._mle))
        print('Goodness of Fit: ', str(self._gof))             
        print('Goodness of Fit Pass: ', str(self._pass))        
        print('Overall Score: ', str(self._score))                
        print('-------------')
        

    