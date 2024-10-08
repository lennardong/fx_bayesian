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


import scipy.stats as stats
import scipy.optimize as opt
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity

"""
The goal here is to model the subjective opinions of experts regarding future values of an FX rate (or any other financial metric) in a probabilistic way, allowing for asymmetry in their expectations (skew) and varying levels of confidence (concentration or bandwidth). The ultimate aim is to translate these opinions into a smooth, well-behaved distribution, which can then be used for forecasting, simulations, or pricing models.

Triangular distribution serves as an intuitive starting point for modeling the expert's central tendency (mode) and bounds, but lacks smoothness.
KDE introduces the smoothness needed for real-world financial models where we prefer continuous, differentiable distributions over jagged ones.
Beta fitting attempts to create a parametric model that represents the expert's opinion while adhering to the desired statistical properties, which may be further used in pricing models or stochastic simulations.
"""


class ExpertOpinion:
    def __init__(
        self,
        label: str,
        expected_mode: float,
        expected_lower_bound: float,
        expected_upper_bound: float,
        z_score: float,
    ) -> None:
        self.label: str = label
        self.expected_mode: float = expected_mode
        self.expected_lower_bound: float = expected_lower_bound
        self.expected_upper_bound: float = expected_upper_bound
        self.z_score: float = z_score

    def synthetic_data(self, bandwidth=0.001):
        """
        Generate synthetic data from expert opinion using KDE for smoothness and peakiness
        """
        min_val = self.expected_lower_bound
        max_val = self.expected_upper_bound
        mode = self.expected_mode

        # Generate initial data
        sample_size = 5000
        initial_data = np.random.triangular(
            min_val, mode, max_val, sample_size
        ).reshape(-1, 1)

        # Apply KDE with controllable bandwidth
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(initial_data)

        # Generate smooth data using KDE
        x = np.linspace(min_val, max_val, sample_size).reshape(-1, 1)
        log_dens = kde.score_samples(x)
        y = np.exp(log_dens)

        # Normalize probabilities
        probabilities = y / np.sum(y)

        # Resample data based on KDE probabilities
        smooth_data = np.random.choice(x[:, 0], size=sample_size, p=probabilities)

        return smooth_data

    def fit_beta(self, data=None):
        if data is None:
            data = self.synthetic_data()

        scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))

        mean = np.mean(scaled_data)
        var = np.var(scaled_data)

        # Initial estimates using method of moments
        alpha = mean * ((mean * (1 - mean) / var) - 1)
        beta = (1 - mean) * ((mean * (1 - mean) / var) - 1)

        # Iterative adjustment for tighter fit
        max_iterations = 100
        learning_rate = 0.1

        for _ in range(max_iterations):
            current_mean = alpha / (alpha + beta)
            current_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

            # Adjust alpha and beta based on the difference between current and target statistics
            alpha += learning_rate * (mean - current_mean) * alpha
            beta += learning_rate * (mean - current_mean) * beta

            # Adjust both parameters to match variance
            var_factor = var / current_var
            alpha *= var_factor
            beta *= var_factor

            # Ensure parameters remain positive
            alpha = max(alpha, 0.01)
            beta = max(beta, 0.01)

        return alpha, beta

    def fit_beta_v1(self, data=None):
        if data is None:
            data = self.synthetic_data()

        scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))

        mean = np.mean(scaled_data)
        var = np.var(scaled_data)
        alpha_guess = max(mean * (mean * (1 - mean) / var - 1), 1)
        beta_guess = max((1 - mean) * (mean * (1 - mean) / var - 1), 1)

        def beta_mle(params: Tuple[float, float]):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return np.inf
            return -np.sum(stats.beta.logpdf(scaled_data, alpha, beta))

        result = opt.minimize(
            fun=beta_mle,
            x0=[alpha_guess, beta_guess],
            method="L-BFGS-B",
            bounds=[(0.01, 500), (0.01, 500)],
            options={"ftol": 1e-10, "maxiter": 2000_000},
        )

        return result.x[0], result.x[1]

    def beta_samples(self, data=None):
        if data is None:
            alpha, beta = self.fit_beta()
        else:
            alpha, beta = data

        if alpha is None or beta is None or np.isnan(alpha) or np.isnan(beta):
            raise ValueError(
                "Failed to fit beta distribution. Alpha or beta is None or NaN."
            )

        # Generate x values in the original currency range
        x = np.linspace(self.expected_lower_bound, self.expected_upper_bound, 1000)

        # Scale x to [0, 1] for beta.pdf
        x_scaled = (x - self.expected_lower_bound) / (
            self.expected_upper_bound - self.expected_lower_bound
        )

        # Generate y values (probabilities)
        y = stats.beta.pdf(x_scaled, alpha, beta)

        # Normalize y using standard NumPy functions
        # y_normalized = y / np.sum(y)
        y_normalized = y

        return x, y_normalized

    def plot_beta(self, data=None):
        x, y_pdf = data or self.beta_samples()
        y_cdf = stats.beta.cdf(x, *self.fit_beta())

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        # Plot PDF
        sns.lineplot(x=x, y=y_pdf, ax=ax1)
        ax1.set_title(f"Beta Distribution PDF for {self.label}")
        ax1.set_ylabel("Probability Density")

        # Plot CDF
        sns.lineplot(x=x, y=y_cdf, ax=ax2)
        ax2.set_title(f"Beta Distribution CDF for {self.label}")
        ax2.set_xlabel("Currency Value")
        ax2.set_ylabel("Cumulative Probability")

        plt.tight_layout()
        plt.show()


##############
if __name__ == "__main__":
    expert1 = ExpertOpinion(
        label="expert1",
        expected_mode=3.25,
        expected_lower_bound=3.2,
        expected_upper_bound=3.45,
        z_score=1.96,
    )

    # Assuming you have data to plot
    # sns.histplot(data=your_data)
    # plt.show()
    for i in range(1, 11):
        bandwidth = i / 1000
        data = expert1.synthetic_data(bandwidth)
        sns.histplot(data)
        plt.show()
        betas = expert1.fit_beta(data)
        print(betas)
        samples = expert1.beta_samples(betas)
        expert1.plot_beta(samples)

    # print(expert1.fit_beta())
    # expert1.plot_beta()
