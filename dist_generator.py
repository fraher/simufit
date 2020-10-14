import numpy as np
from scipy.optimize import minimize


class Uniform:

    def __init__(self):
        pass

    def sample(self, a=0., b=1., size=None, seed=None):

        if seed:
            np.random.seed(seed)

        return np.random.uniform(low=a, high=b, size=size)

    def MLE(self, samples):
        a = np.min(samples)
        b = np.max(samples)

        return a, b


class Normal:

    def __init__(self):
        pass

    def sample(self, mean=0., var=1., size=None, seed=None):

        if var < 0:
            raise ValueError('var must be non-negative.')

        if seed:
            np.random.seed(seed)

        return np.random.normal(loc=mean, scale=np.sqrt(var), size=size)

    def negLogL(self, mean, var, samples):

        n = len(samples)

        return (-(n / 2) * np.log(2 * np.pi * var) - (1 / (2 * var)) * np.sum((samples - mean) ** 2)) * -1

    def MLE(self, samples, use_minimizer=False, x0=None):

        if use_minimizer:
            if x0 is None:
                raise ValueError('Supply an initial guess x0=(mean, var) to the optimizer.')
            def nll(x, samples):
                return self.logL(*x, samples)
            return minimize(nll, x0, args=samples, method='Nelder-Mead')

        else:
            mu = np.mean(samples)
            var = np.std(samples) ** 2

            return mu, var



