class DistributionReport():
    """The Report class provides contains a full analysis of the
    dataset evaluated."""

    def __init__(self):
        self._distribution_type = None
        self._mle = None
        self._gof = None        

    def setDistributionType(self, distribution_type):
        self._distribution_type = distribution_type
    
    def setMLE(self, mle):
        self._mle = mle

    def setGOF(self, gof):
        self._gof = gof

    def printReport(self):
        """Displays all Distribution Report Items collected 
        during the Simufit analysis."""
        print('Distribution Type: ', self._distribution_type)
        print('MLE: ', str(self._mle))
        print('Goodness of Fit: ', str(self._gof))        
        

    