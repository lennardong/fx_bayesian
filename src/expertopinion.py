# import os
# os.environ['PYTENSOR_FLAGS'] = 'device=cuda,floatX=float32'

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pytensor


"""

# P1 - build skew normal estimation model
"""


class ExpertOpinion:
    """
    Expert Opinion of FX exchange rates.
    """

    def __init__(self, median, q25, q75, q05, q95):
        self.median = median
        self.q25 = q25
        self.q75 = q75
        self.q05 = q05
        self.q95 = q95


def estimate_parameters(
    expert: ExpertOpinion, *, samples: int = 2000, chains: int = 2
) -> az.InferenceData:
    with pm.Model() as model:  # why use context manager for pmmodel?

        # Build hyper params for stocahstic vars
        mu = pm.Normal("mu", mu=expert.median, sigma=0.2)
        sigma = pm.HalfNormal("sigma", sigma=0.2)
        alpha = pm.Normal("alpha", mu=0, sigma=3)

        # Build model
        fx_distribution = pm.SkewNormal(
            "fx_distribution", mu=mu, sigma=sigma, alpha=alpha
        )

        # Generate samples to compute the quantiles
        fx_samples = pm.draw(fx_distribution, draws=samples)
        q25_val = np.percentile(fx_samples, 25)
        q75_val = np.percentile(fx_samples, 75)
        medial_val = np.percentile(fx_samples, 50)

        # Fit the model
        likelihood = pm.Normal(
            "likelihood",
            mu=0,
            sigma=0.5,  # acceptable error
            observed=[
                expert.q25 - q25_val,
                expert.q75 - q75_val,
                expert.median - medial_val,
            ],
        )

    # Sample from the posterior
    with model:  # ? why with nested in withth?
        trace = pm.sample(
            # model=model,
            draws=samples,
            tune=samples,
            chains=chains,
            target_accept=1.00,
            return_inferencedata=True,
        )

    return trace


def plot_az(trace: az.InferenceData):
    """
    Plots posterior distributions, credible intervals, and skew normal traces using Arviz.

    Parameters:
    - trace (az.InferenceData): The posterior samples
    """
    # Plot posterior distributions for mu, sigma, and alpha
    az.plot_posterior(trace, var_names=["mu", "sigma", "alpha"])

    # Generate skew normal samples using the posterior
    mu_samples = trace.posterior["mu"].values.flatten()
    sigma_samples = trace.posterior["sigma"].values.flatten()
    alpha_samples = trace.posterior["alpha"].values.flatten()

    x = np.linspace(
        mu_samples.mean() - 3 * sigma_samples.mean(),
        mu_samples.mean() + 3 * sigma_samples.mean(),
        1000,
    )

    # Plot skew normal traces
    plt.figure(figsize=(10, 6))
    for i in range(1000):  # Plot 100 random traces
        idx = np.random.randint(len(mu_samples))
        y = stats.skewnorm.pdf(
            x, alpha_samples[idx], mu_samples[idx], sigma_samples[idx]
        )
        plt.plot(x, y, alpha=0.1)

    plt.title("Skew Normal Traces from Posterior Samples")
    plt.xlabel("Value")
    plt.ylabel("Density")

    plt.show()


def test_function(median, q25, q75, q05, q95) -> az.InferenceData:
    """
    Test function to run the model estimation, plot the trace, and visualize distributions.
    """
    expert = ExpertOpinion(median, q25, q75, q05, q95)
    trace = estimate_parameters(expert)
    # plot_trace(trace)
    # plot_skewnormal(expert, trace)
    plot_az(trace)
    return trace


if __name__ == "__main__":
    test_trace = test_function(3.28, 3.25, 3.29, 3.20, 3.35)
