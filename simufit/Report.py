class DistributionReport():
    """The Report class provides contains a full analysis of the
    dataset evaluated."""

    def __init__(self):
        self.topDistributions = list()
        pass

    def printReport(self):
        """Displays the top 3 Distribution Report Items collected 
        during the Simufit analysis."""

        # TODO: Restrict to 3 report items based on highest MLE
        for distribution in self.topDistributions: 
            print(distribution)

    