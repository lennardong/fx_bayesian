import numpy as np
from scipy.stats import norm


class BayesModeller:
    """
    A class for Bayesian modeling and forecasting of time series data.
    """

    def __init__(self, data):
        self.historical_data = data
        self.model = None


def estimate_parameters(fx_high, fx_low, fx_expected, weight):
    """
    estimate parameters of a prior distribution.

    uses beta distributions where:
    Args:
        fx_high: expert opinon on upper bound of fx
        fx_low: expert opinon on lower bound of fx rate
        fx_expected: expert opinon on expected fx rate
        weight: weight of expert opinon, should be between 0 and 1 where 1 is most confident

    Returns:
        alpha and beta parameters of beta distribution
    """
    mu_norm = (fx_expected - fx_low) / (fx_high - fx_low)
    var = (1 - weight) * 0.25
    alpha = ((1 - mu_norm) / var - 1 / mu_norm) * mu_norm**2

    beta = alpha * (1 / mu_norm - 1)
    return alpha, beta
