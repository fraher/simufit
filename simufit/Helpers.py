import numpy as np
import scipy.stats


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
