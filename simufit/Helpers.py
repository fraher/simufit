import numpy as np
import scipy.stats
from scipy.special import digamma, polygamma

def gammaMLE(samples):
    """Returns MLE parameters a_hat, b_hat for Gamma-distributed samples.
    See https://en.wikipedia.org/wiki/Gamma_distribution#Maximum_likelihood_estimation."""

    n = len(samples)
    s = np.log((1 / n) * np.sum(samples)) - (1 / n) * np.sum(np.log(samples))
    a_hat = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)

    diff = np.inf
    i = 0
    while diff > 1e-8:
        a_new = a_hat - (np.log(a_hat) - digamma(a_hat) - s) / ((1 / a_hat) - polygamma(1, a_hat))
        diff = np.abs(a_new - a_hat)
        a_hat = a_new
        i += 1
        if i == 1000:
            print('Failed to converge in 1000 iterations.')
            break
    b_hat = (1 / (a_hat * n)) * np.sum(samples)

    if i < 1000:
        print(f'Converged in {i} iterations.')

    return np.array([a_hat, b_hat])

def weibullMLE(samples):
    """Returns MLE parameters a_hat, b_hat for Weibull-distributed samples.
    See Simulation & Modeling Chapter pp. 290-292 (Law 5e)."""

    n = len(samples)

    A = np.sum(np.log(samples)) / n
    B = lambda x: np.sum(samples ** x)
    C = lambda x: np.sum((samples ** x) * np.log(samples))
    H = lambda x: np.sum((samples ** x) * (np.log(samples) ** 2))

    a_hat = (((6 / (np.pi ** 2)) * (np.sum(np.log(samples) ** 2) - ((np.sum(np.log(samples))) ** 2) / n)) / (n - 1)) ** -0.5

    diff = np.inf
    i = 0
    while diff > 1e-8:
        a_new = a_hat + (A + (1 / a_hat) - (C(a_hat) / B(a_hat))) / ((1 / (a_hat ** 2)) + (B(a_hat) * H(a_hat) - C(a_hat) ** 2) / (B(a_hat) ** 2))
        diff = np.abs(a_new - a_hat)
        a_hat = a_new
        i += 1
        if i == 1000:
            print('Failed to converge in 1000 iterations.')
            break
    b_hat = (np.sum(samples ** a_hat) / n) ** (1 / a_hat)

    if i < 1000:
        print(f'Converged in {i} iterations.')

    return np.array([a_hat, b_hat])

def getBadBins(num_samples, edges, bad_bin, merge_bin, func, *args):
    """Helper function for mergeBins. Returns the remaining bad bins
    after merging."""

    # Determine bad bins
    edges = np.delete(edges, np.arange(bad_bin+1, merge_bin+1))

    if func == scipy.stats.norm:
        probs = func.cdf(x=edges, loc=args[0], scale=args[1])
    elif func == scipy.stats.expon:
        probs = func.cdf(x=edges, scale=args[0])
    elif func == scipy.stats.gamma:
        probs = func.cdf(x=edges, a=args[0], scale=args[1])
    elif func == scipy.stats.weibull_min:
        probs = func.cdf(x=edges, c=args[0], scale=args[1])
    elif func == scipy.stats.geom:
        probs = func.cdf(k=edges, p=args[0])
    elif func == scipy.stats.binom:
        probs = func.cdf(k=edges, n=args[0], p=args[1])

    expected = np.diff(probs, n=1) * num_samples

    bad_bins = list(np.where(expected < 5)[0])

    return edges, expected, bad_bins

def mergeBins(samples, func, *args):
    """Merge bins of histogram so that no bin has less than 5 samples.
    This is to prepare histogram for chi-squared goodness of fit test."""

    # Get initial histogram and expected # samples in each bin
    n = len(samples)

    if func == scipy.stats.geom:
        hist, edges = np.histogram(samples, bins=np.arange(np.max(samples) + 1))
        probs = func.cdf(k=edges, p=args[0])
    elif func == scipy.stats.binom:
        hist, edges = np.histogram(samples, bins=np.arange(np.max(samples) + 1))
        probs = func.cdf(k=edges, n=args[0], p=args[1])
    else:
        hist, edges = np.histogram(samples, bins='fd')
        if func == scipy.stats.norm:
            probs = func.cdf(x=edges, loc=args[0], scale=args[1])
        elif func == scipy.stats.expon:
            probs = func.cdf(x=edges, scale=args[0])
        elif func == scipy.stats.gamma:
            probs = func.cdf(x=edges, a=args[0], scale=args[1])
        elif func == scipy.stats.weibull_min:
            probs = func.cdf(x=edges, c=args[0], scale=args[1])

    expected = np.diff(probs, n=1) * n

    # Get bins where # samples is less than 5
    bad_bins = list(np.where(expected < 5)[0])

    # Main loop, looks from the left and merges bin with less than 5 counts
    # with bins to the right until counts exceed 5. If cannot reach 5 by merging
    # all right-side bins, merge all remaining to the right. Then merge to the
    # left until counts exceed 5.
    while len(bad_bins) > 0:
        bad_bin = bad_bins.pop(0)
        merge_bin = np.argmax(np.cumsum(expected[bad_bin:]) > 5) + bad_bin

        if bad_bin != merge_bin:
            edges, expected, bad_bins = getBadBins(n, edges, bad_bin, merge_bin, func, *args)

        if bad_bin == merge_bin:  # This happens when you are at the right-most bin or when merging all remaining cannot get to 5.
            if not np.all(np.cumsum(expected[bad_bin:]) > 5) and merge_bin != len(expected) - 1:
                merge_bin = len(expected) - 1
                edges, expected, bad_bins = getBadBins(n, edges, bad_bin, merge_bin, func, *args)
            else:
                merge_bin = bad_bin - np.argmax(expected[::-1].cumsum() > 5)
                edges, expected, bad_bins = getBadBins(n, edges, merge_bin, bad_bin, func, *args)  # Reverse the order here b/c we are merging left

    return edges, expected
