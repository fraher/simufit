import numpy as np
from scipy.optimize import minimize


class Geometric:

    def __init__(self):
        pass

    def sample(self, p, size=None):
        """Get samples from Geom(p). The size argument is the number of samples (default 1)."""

        if p <= 0:
            raise ValueError('p must be greater than 0.')

        return np.random.geometric(p=p, size=size)

    def negLogL(self, p, samples):

        n = len(samples)

        return (n * np.log(p) + np.sum(samples - 1) * np.log(1 - p)) * -1

    def MLE(self, samples, use_minimizer=False, x0=None):

        if use_minimizer:
            if x0 is None:
                raise ValueError('Supply an initial guess x0=p to the optimizer.')
            return minimize(self.negLogL, x0, args=samples, method='Nelder-Mead')

        else:
            return 1 / np.mean(samples)


class Uniform:

    def __init__(self):
        pass

    def sample(self, a=0., b=1., size=None):
        """Get samples from Unif(a, b). The size argument is the number of samples (default 1)."""

        return np.random.uniform(low=a, high=b, size=size)

    def MLE(self, samples):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random uniformly distributed samples. Returns the MLE parameters a and b."""

        a = np.min(samples)
        b = np.max(samples)

        return a, b


class Normal:

    def __init__(self):
        pass

    def seed(self, seed):
        """Set seed."""

        np.random.seed(seed)

    def sample(self, mean=0., var=1., size=None):
        """Get samples from Norm(μ, σ^2). The size argument is the number of samples (default 1)."""

        if var < 0:
            raise ValueError('var must be non-negative.')

        return np.random.normal(loc=mean, scale=np.sqrt(var), size=size)

    def negLogL(self, mean, var, samples):
        """Calculate the negative log likelihood for a collection of random
        normally distributed samples, and a specified mean and variance."""

        if var < 0:
            raise ValueError('var must be non-negative.')

        n = len(samples)

        return (-(n / 2) * np.log(2 * np.pi * var) - (1 / (2 * var)) * np.sum((samples - mean) ** 2)) * -1

    def MLE(self, samples, use_minimizer=False, x0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random normally distributed samples. Returns the MLE mean and variance.
        If use_minimizer=True, provide a initial guess for the optimizer in
        form x0=(mean_guess, var_guess)."""

        if use_minimizer:
            if x0 is None:
                raise ValueError('Supply an initial guess x0=(mean, var) to the optimizer.')
            def nll(x, samples):
                return self.negLogL(*x, samples)
            return minimize(nll, x0, args=samples, method='Nelder-Mead')

        else:
            mu = np.mean(samples)
            var = np.std(samples) ** 2

            return mu, var

class Exponential:

    def init(self):
        pass

    def sample(self, lambd=1., size=None):
        """Get samples from Exp(λ). The size argument is the number of samples (default 1)."""

        if lambd < 0:
            raise ValueError('lambd must be non-negative.')

        return np.random.exponential(scale=1/lambd, size=size)

    def negLogL(self, lambd, samples):
        """Calculate the negative log likelihood for a collection of random
        exponentially distributed samples, and a specified scale parameter lambd."""

        if lambd < 0:
            raise ValueError('lambd must be non-negative.')

        n = len(samples)

        return (n * np.log(lambd) - lambd * np.sum(samples)) * -1

    def MLE(self, samples, use_minimizer=False, x0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random normally distributed samples. Returns the MLE λ (lambd).
        If use_minimizer=True, provide a initial guess for the optimizer in
        form x0=lambd."""

        if use_minimizer:
            if x0 is None:
                raise ValueError('Supply an initial guess x0=lambd to the optimizer.')
            return minimize(self.negLogL, x0, args=samples)

        else:
            n = len(samples)
            lambd = n / np.sum(samples)
            return lambd

