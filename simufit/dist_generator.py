import numpy as np
from simufit.Helpers import mergeBins, gammaMLE, weibullMLE
from scipy.optimize import minimize
import scipy.stats
import scipy.special

from simufit.Types import MeasureType as mt
from simufit.Display import Display

class Bernoulli(Display):

    def __init__(self):
        self.name = 'Bernoulli'
        self.measure_type = mt.DISCRETE
        self._parameters = [{'label': 'p', 'probability':[0,1]}]

    def sample(self, p, size=None, seed=None):
        """Get samples from Bern(p). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if p <= 0 or p >= 1:
            raise ValueError('p must be in the range (0, 1).')

        samples = np.random.uniform(size=size)
        mask1 = samples >= p
        mask2 = samples < p
        samples[mask1] = 0
        samples[mask2] = 1

        return samples.astype(np.int64)

    def negLogL(self, p, samples):
        """Calculate the negative log likelihood for a collection of random
        Bernoulli-distributed samples, and a specified p."""

        n = len(samples)
        m = np.sum(samples)

        return (m * np.log(p) + (n - m) * np.log(1 - p)) * -1

    def MLE(self, samples, use_minimizer=False, p0=None):
        """Returns the maximum likelihood estimate of parameter p, given a collection of samples.
        If use_minimizer=True, an initial guess p0 for p must be provided. Otherwise, the closed
        form expression for the MLE of p is used."""

        if use_minimizer:
            if p0 is None:
                raise ValueError('Supply an initial guess p0=p to the optimizer.')
            if p0 <= 0 or p0 >= 1:
                raise ValueError('p must be in the range (0, 1). Supply an initial guess in this range.')
            res = minimize(self.negLogL, p0, args=samples, method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {p0}. Returned None for MLE values. Try another initial guess.')
                return None
            else:
                return res.x

        else:
            return np.array([np.mean(samples)])

class Binomial(Display):

    def __init__(self):
        self.name = 'Binomial'
        self.measure_type = mt.DISCRETE
        self._parameters = [{'label': 'n', 'range':[0,100]}, {'label': 'p', 'probability':[0,1]}]

    def sample(self, n, p, size=None, seed=None):
        """Get samples from Bin(n, p). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if type(n) != int or n < 0 or p <= 0 or p >= 1:
            raise ValueError('n must be an integer >= 0. p must be in the range (0, 1).')

        return np.random.binomial(n, p, size=size)

    def negLogL(self, p, n, samples):
        """Calculate the negative log likelihood for a collection of random
        Bernoulli-distributed samples, and a specified p."""

        return (np.sum(scipy.special.comb(n, samples)) + np.sum(samples) * np.log(p) + (n * len(samples) - np.sum(samples)) * np.log(1 - p)) * -1

    def MLE(self, samples, n, use_minimizer=False, p0=None):
        """Returns the maximum likelihood estimate of parameter p, given a collection of samples.
        The Binomial parameter n must be known or estimated to use this function."""

        if use_minimizer:
            if p0 is None:
                raise ValueError('Supply an initial guess p0=p to the optimizer.')
            if p0 <= 0 or p0 >= 1:
                raise ValueError('p must be in the range (0, 1).')

            res = minimize(self.negLogL, p0, args=(n, samples), method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {p0}. Returned None for MLE values. Try another initial guess.')
                return None
            else:
                return res.x
        else:
            return np.array([np.mean(samples) / n])

    def GOF(self, samples, n, mle_p):
        """Returns the chi-squared goodness of fit statistic for a set of MLE paramters."""

        edges, f_exp = mergeBins(samples, scipy.stats.binom, n, mle_p)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges+1)
            chisq, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=len(f_obs)-2)

            return np.array([chisq, scipy.stats.chi2.isf(0.05, len(f_obs)-2)])
        else:
            return None, None

class Geometric(Display):

    def __init__(self):
        self.name = 'Geometric'
        self.measure_type = mt.DISCRETE
        self._parameters = [{'label': 'p', 'probability':[0,1]}]

    def sample(self, p, size=None, seed=None):
        """Get samples from Geom(p). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if p <= 0 or p >= 1:
            raise ValueError('p must be in the range (0, 1).')

        return np.random.geometric(p=p, size=size)

    def negLogL(self, p, samples):
        """Returns the negative log likelihood given a collection of samples and parameter p."""

        n = len(samples)

        return (n * np.log(p) + np.sum(samples - 1) * np.log(1 - p)) * -1

    def MLE(self, samples, use_minimizer=False, p0=None):
        """Returns the maximum likelihood estimate of parameter p, given a collection of samples.
        If use_minimizer=True, an initial guess p0 for p must be provided. Otherwise, the closed
        form expression for the MLE of p is used."""

        if use_minimizer:
            if p0 is None:
                raise ValueError('Supply an initial guess p0=p to the optimizer.')
            if p0 <= 0 or p0 >= 1:
                raise ValueError('p must be in the range (0, 1). Supply an initial guess in this range.')
            res = minimize(self.negLogL, p0, args=samples, method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {p0}. Returned None for MLE values. Try another initial guess.')
                return None
            else:
                return res.x

        else:
            return np.array([1 / np.mean(samples)])

    def GOF(self, samples, mle_p):
        """Returns the chi-squared goodness of fit statistic for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.geom, mle_p)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges+1)
            ddof = len(f_obs) - 2
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, ddof)

        return np.array([chisq0, chisq])

class Uniform(Display):

    def __init__(self):
        self.name = 'Uniform'
        self.measure_type = mt.CONTINUOUS
        self._parameters = [{'label':'a', 'range':[0,100], 'position':'min'}, {'label':'b', 'range':[0,100], 'position':'max'}]

    def sample(self, a=0., b=1., size=None, seed=None):
        """Get samples from Unif(a, b). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        return np.random.uniform(low=a, high=b, size=size)

    def MLE(self, samples):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random uniformly distributed samples. Returns the MLE parameters a and b."""
        a = np.min(samples)
        b = np.max(samples)

        return np.array([a, b])

    def GOF(self, samples, mle_a, mle_b):
        """Returns the chi-squared goodness of fit statistic for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.uniform, mle_a, mle_b)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges)
            ddof = len(f_obs) - 3
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, ddof)

        return np.array([chisq0, chisq])

class Normal(Display):

    def __init__(self):
        self.name = 'Normal'
        self.measure_type = mt.CONTINUOUS
        self._parameters = [{'label':'mean', 'span':[0,10]}, {'label':'var', 'span':[0,100]}]

    def sample(self, mean=0., var=1., size=None, seed=None):
        """Get samples from Norm(μ, σ^2). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

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

    def MLE(self, samples, use_minimizer=False, mean0=None, var0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random normally distributed samples. Returns the MLE mean and variance.
        If use_minimizer=True, provide a initial guess for the optimizer in
        form mean0, var0."""

        if use_minimizer:
            if mean0 is None:
                raise ValueError('Supply an initial guess mean0 to the optimizer.')
            if var0 is None:
                raise ValueError('Supply an initial guess var0 to the optimizer.')
            if var0 < 0:
                raise ValueError('var0 must be non-negative. Supply an initial guess var0 with a positive var.')
            def nll(x, samples):
                return self.negLogL(*x, samples)
            res = minimize(nll, (mean0, var0), args=samples, method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {mean0}, {var0}. Returned None for MLE values. Try another initial guess.')
                return None, None
            else:
                return res.x

        else:
            mu = np.mean(samples)
            var = np.var(samples)

            return np.array([mu, var])

    def GOF(self, samples, mle_mu, mle_var):
        """Return the chi-squared goodness of fit statistic and p-value for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.norm, mle_mu, np.sqrt(mle_var))
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges)
            ddof = len(f_obs) - 3
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, ddof)

        return np.array([chisq0, chisq])

class Exponential(Display):

    def __init__(self):
        self.name = 'Exponential'
        self.measure_type = mt.CONTINUOUS
        self._parameters = [{'label':'lambd','span':[1,10]}]

    def sample(self, lambd=1., size=None, seed=None):
        """Get samples from Exp(λ). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if not lambd > 0:
            raise ValueError('lambd must be greater than 0.')

        return np.random.exponential(scale=1/lambd, size=size)

    def negLogL(self, lambd, samples):
        """Calculate the negative log likelihood for a collection of random
        exponentially distributed samples, and a specified scale parameter lambd."""

        if not lambd > 0:
            raise ValueError('lambd must be greater than 0.')

        n = len(samples)

        return (n * np.log(lambd) - lambd * np.sum(samples)) * -1

    def MLE(self, samples, use_minimizer=False, lambd0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random normally distributed samples. Returns the MLE λ (lambd).
        If use_minimizer=True, provide a initial guess for the optimizer in
        form lambd0=lambd."""

        if use_minimizer:
            if lambd0 is None:
                raise ValueError('Supply an initial guess lambd0=lambd to the optimizer.')
            if lambd0 < 0:
                raise ValueError('lambd must be non-negative. Supply an positive initial guess lambd0=lambd.')
            res = minimize(self.negLogL, lambd0, args=samples)
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {lambd0}. Returned None for MLE values. Try another initial guess.')
                return None
            else:
                return res.x

        else:
            lambd = 1 / np.mean(samples)
            return np.array([lambd])

    def GOF(self, samples, mle_lambda):
        """Return the chi-squared goodness of fit statistic and p-value for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.expon, 1/mle_lambda)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges)
            ddof = len(f_obs) - 2
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, len(f_obs)-2)

        return np.array([chisq0, chisq])

class Gamma(Display):

    def __init__(self):
        self.name = 'Gamma'
        self.measure_type = mt.CONTINUOUS
        self._parameters = [{'label':'a', 'span':[0,100]}]

    def sample(self, a, b=1., size=None, seed=None):
        """Get samples from Gamma(a, b). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if not a > 0 or not b > 0:
            raise ValueError('a and b must be greater than 0.')

        return np.random.gamma(shape=a, scale=b, size=size)

    def negLogL(self, a, b, samples):
        """Calculate the negative log likelihood for a collection of random
        gamma-distributed samples, and specified shape and scale parameters a and b."""

        n = len(samples)

        return ((a - 1) * np.sum(np.log(samples)) - n * np.log(scipy.special.gamma(a)) - n * a * np.log(b) - (np.sum(samples) / b)) * -1

    def MLE(self, samples, use_minimizer=False, a0=None, b0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random gamma-distributed samples. Returns the MLE a and b. Provide an
        initial guess for the optimizer in form a0, b0."""

        if use_minimizer:
            if a0 is None:
                raise ValueError('Supply an initial guess for a0 to the optimizer.')
            if b0 is None:
                raise ValueError('Supply an initial guess for b0 to the optimizer.')
            if not a0 > 0 or not b0 > 0:
                raise ValueError('a0 and b0 must be greater than 0. Supply an initial guess with a0 > 0 and b0 > 0.')

            def nll(x, samples):
                return self.negLogL(*x, samples)

            res = minimize(nll, (a0, b0), args=samples, method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {a0}, {b0}. Returned None for MLE values. Try another initial guess.')
                return None, None
            else:
                return res.x
        else:
            return gammaMLE(samples)

    def GOF(self, samples, mle_a, mle_b):
        """Return the chi-squared goodness of fit statistic and p-value for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.gamma, mle_a, mle_b)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges)
            ddof = len(f_obs) - 3
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, ddof)

        return np.array([chisq0, chisq])

class Weibull(Display):

    def __init__(self):
        self.name = 'Weibull'
        self.measure_type = mt.CONTINUOUS
        self._parameters = [{'label':'a', 'span':[1,100]}]

    def sample(self, a, b=1, size=None, seed=None):
        """Get samples from Weibull(a, b). The shape parameter is a, the scale parameter is b (default 1).
        The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if not a > 0 or not b > 0:
            raise ValueError('a and b must be greater than 0.')

        return b * np.random.weibull(a=a, size=size)

    def negLogL(self, a, b, samples):
        """Calculate the negative log likelihood for a collection of random
        weibull-distributed samples, and specified shape and scale parameters a and b."""

        n = len(samples)

        return (n * np.log(a) - n * np.log(b) + (a - 1) * np.sum(np.log(samples / b)) - np.sum(np.power((samples / b), a))) * -1

    def MLE(self, samples, use_minimizer=False, a0=None, b0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random weibull-distributed samples. Returns the MLE a and b. Provide an
        initial guess for the optimizer in form a0, b0."""

        if use_minimizer:
            if a0 is None:
                raise ValueError('Supply an initial guess for a0 to the optimizer.')
            if b0 is None:
                raise ValueError('Supply an initial guess for b0 to the optimizer.')
            if not a0 > 0 or not b0 > 0:
                raise ValueError('a and b must be greater than 0. Supply an initial guess with a0 > 0 and b0 > 0.')

            def nll(x, samples):
                return self.negLogL(*x, samples)

            res = minimize(nll, (a0, b0), args=samples, method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {a0}, {b0}. Returned None for MLE values. Try another initial guess.')
                return None, None
            else:
                return res.x
        else:
            return weibullMLE(samples)

    def GOF(self, samples, mle_a, mle_b):
        """Return the chi-squared goodness of fit statistic and p-value for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.weibull_min, mle_a, mle_b)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges)
            ddof = len(f_obs) - 3
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, ddof)

        return np.array([chisq0, chisq])

class Unknown(Display):

    def __init__(self):
        self.name = 'Unknown'
        self.measure_type = mt.UNKNOWN
