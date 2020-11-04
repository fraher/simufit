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
    