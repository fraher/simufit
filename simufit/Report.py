import numpy as np
from simufit.Types import DistributionType as dt

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

    def setDistributionType(self, distribution_type):
        self._distribution_type = distribution_type
    
    def setMLE(self, mle):
        self._mle = mle

    def setGOF(self, gof):
        self._gof = gof

    def getDistributionType(self):
        return self._distribution_type

    def getScore(self):
        return self._score    

    def isPass(self):
        return self._pass

    def evaluateScore(self, samples):
        self._unique_elements = len(np.unique(samples))
        if self._distribution_type == dt.BERNOULLI:
            if self._unique_elements <= 2:
                self._pass = True
                self._score = np.inf
        
        if self._gof is not None and self._score != 1:
            if type(self._gof) is not str:
                if self._gof[0] != np.nan and self._gof[1] != np.nan:                                           
                    self._score = 1/np.exp((self._gof[0] - self._gof[1])/self._gof[1])
                    if self._gof[0] < self._gof[1]:
                        self._pass = True
                

    def printReport(self):
        """Displays all Distribution Report Items collected 
        during the Simufit analysis."""
        print('Distribution Type: ', self._distribution_type)
        print('MLE: ', str(self._mle))
        print('Goodness of Fit: ', str(self._gof))     
        print('Unique Elements: ', str(self._unique_elements)) 
        print('Pass: ', str(self._pass))  
        print('Overall Score: ', str(self._score))        
        

    