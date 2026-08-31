"""
Monte Carlo distribution samplers for the GUV scale-up model.

Provides bounded normal, lognormal, uniform, and Generalized Pareto sampling with
configurable confidence intervals (Guesstimate-style where applicable).
"""

import numpy as np
from scipy.stats import norm, genpareto


def sample_normal(low, high, n, confidence=90, absolute=False):
    """
    Generate random samples from a normal distribution. Based on Guesstimate's
    implementation, translated from Javascript to Python.

    Arguments:
        low (float): The lower bound of the distribution.
        high (float): The upper bound of the distribution.
        n (int): The number of samples to generate.
        confidence (int): The confidence level. Must be 90, 95, or 99. Default 90.
        absolute (bool): If True, return absolute values of samples. Default False.

    Returns:
        numpy.ndarray: Random samples from the normal distribution, shape (n,).
    """
    if confidence == 90:
        z = 1.645
    elif confidence == 95:
        z = 1.96
    elif confidence == 99:
        z = 2.575
    else:
        raise ValueError("Confidence level must be 90, 95, or 99")

    mean = np.mean([high, low])
    stdev = (high - mean) / z
    samples = norm.rvs(loc=mean, scale=stdev, size=n)
    if absolute:
        samples = np.abs(samples)

    return samples


def sample_lognormal(low, high, n, confidence=90, absolute=False):
    """
    Generate random samples from a lognormal distribution.

    Arguments:
        low (float): The lower bound of the distribution. Must be > 0.
        high (float): The upper bound of the distribution.
        n (int): The number of samples to generate.
        confidence (int): The confidence level. Must be 90, 95, or 99. Default 90.
        absolute (bool): If True, return absolute values of samples. Default False.

    Returns:
        numpy.ndarray: Random samples from the lognormal distribution, shape (n,).
    """
    assert low > 0, "Low must be greater than 0 for lognormal distributions."

    if confidence == 90:
        z = 1.645
    elif confidence == 95:
        z = 1.96
    elif confidence == 99:
        z = 2.575
    else:
        raise ValueError("Confidence level must be 90, 95, or 99")
    logHigh = np.log(high)
    logLow = np.log(low)

    mean = np.mean([logHigh, logLow])
    stdev = (logHigh - logLow) / (2 * z)
    samples = np.random.lognormal(mean=mean, sigma=stdev, size=n)
    if absolute:
        samples = np.abs(samples)

    return samples


def sample_uniform(low, high, n):
    """
    Generate random samples from a uniform distribution.

    Arguments:
        low (float): The lower bound of the distribution.
        high (float): The upper bound of the distribution.
        n (int): The number of samples to generate.

    Returns:
        numpy.ndarray: Random samples from the uniform distribution, shape (n,).
    """
    return np.random.uniform(low=low, high=high, size=n)


def sample_gpd(shape_param, scale_param, loc_param=1.0, n=1000):
    """
    Generate random samples from a Generalized Pareto Distribution (GPD).

    Arguments:
        shape_param (float): The shape parameter of the GPD.
        scale_param (float): The scale parameter of the GPD.
        loc_param (float): The location parameter of the GPD. Default 1.0.
        n (int): The number of samples to generate. Default 1000.

    Returns:
        numpy.ndarray: Random samples from the GPD, shape (n,).
    """
    samples = genpareto.rvs(c=shape_param, loc=loc_param, scale=scale_param, size=n)
    return samples
