from enum import Enum

class DistributionType(Enum):
    """The distribution type enumeration specifies each supported
    distribution type in the Simufit package."""
    GEOMETRIC = 1
    UNIFORM = 2
    NORMAL = 3
    EXPONENTIAL = 4
    GAMMA = 5
    BERNOULLI = 6
    BINOMIAL = 7
    WEIBULL = 8
    UNKNOWN = 9  

class MeasureType(Enum):
    """This measure type enumeration specifies whether a distribution
    is discrete or continuous."""
    DISCRETE = 1
    CONTINUOUS = 2
    UNKNOWN = 3