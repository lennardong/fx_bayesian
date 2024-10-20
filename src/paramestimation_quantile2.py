import numpy as np
from scipy import stats
from scipy.optimize import minimize, curve_fit, brentq
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns


@dataclass
class ExpertInput:
    """Container for expert inputs with validation"""

    median: float
    q25: float
    q75: float
    q05: float = None  # Optional tail information
    q95: float = None
    confidence: float = 0.8  # Expert's confidence level

    def __post_init__(self):
        """Validate expert inputs"""
        if not (self.q25 < self.median < self.q75):
            raise ValueError("Median must be between Q25 and Q75")
        if self.q05 is not None and self.q05 >= self.q25:
            raise ValueError("Q05 must be less than Q25")
        if self.q95 is not None and self.q95 <= self.q75:
            raise ValueError("Q95 must be greater than Q75")
        if not (0 < self.confidence < 1):
            raise ValueError("Confidence must be between 0 and 1")


def logistic_function(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


class ExpertOpinion:
    """
    Enhanced expert opinion model using quantile-based Beta distribution fitting
    with improved initialization and shape control.
    """

    def __init__(self, expert_input: ExpertInput):
        self.expert = expert_input
        self.beta_params = None
        self._normalize_inputs()

    def _normalize_inputs(self):
        """Normalize all inputs to [0,1] interval"""
        # Find global bounds
        all_quantiles = [
            self.expert.q05 if self.expert.q05 is not None else self.expert.q25,
            self.expert.q25,
            self.expert.median,
            self.expert.q75,
            self.expert.q95 if self.expert.q95 is not None else self.expert.q75,
        ]
        self.lower = min(all_quantiles) - 0.1
        self.upper = max(all_quantiles) + 0.1
        self.range = self.upper - self.lower
        # self._calculate_bounds()

        # Normalize all quantiles
        self.norm_q25 = (self.expert.q25 - self.lower) / self.range
        self.norm_q75 = (self.expert.q75 - self.lower) / self.range
        self.norm_median = (self.expert.median - self.lower) / self.range

        if self.expert.q05 is not None:
            self.norm_q05 = (self.expert.q05 - self.lower) / self.range
        if self.expert.q95 is not None:
            self.norm_q95 = (self.expert.q95 - self.lower) / self.range

    def _calculate_bounds(self):
        """
        This doesn't work well
        """
        # Left side (median to Q05)
        x_left = np.array([0.5, 0.05])
        y_left = np.array([self.expert.median, self.expert.q05])
        slope_left, intercept_left = np.polyfit(x_left, y_left, 1)

        # Right side (median to Q95)
        x_right = np.array([0.5, 0.95])
        y_right = np.array([self.expert.median, self.expert.q95])
        slope_right, intercept_right = np.polyfit(x_right, y_right, 1)

        # Calculate bounds
        self.lower = slope_left * 0 + intercept_left
        self.upper = slope_right * 1 + intercept_right
        self.range = self.upper - self.lower

    def _estimate_initial_params(self):
        """
        Improved initial parameter estimation using method of moments
        and quantile relationships
        """
        # Use IQR to estimate spread
        iqr = self.norm_q75 - self.norm_q25

        # Estimate mean and variance using median and IQR
        # For beta distribution, we assume median ≈ mean for initial guess
        mean = self.norm_median

        # Estimate variance using IQR (assuming approximately normal for initial guess)
        # IQR ≈ 1.35 * σ for normal distribution
        variance = (iqr / 1.35) ** 2

        # Apply confidence adjustment
        # Higher confidence = lower variance
        variance = variance * (1 - 0.5 * self.expert.confidence)

        # Method of moments for beta distribution
        mean_term = mean * (1 - mean) / variance - 1

        # Initial alpha/beta estimates
        alpha = mean * mean_term
        beta = (1 - mean) * mean_term

        # Adjust for skewness
        skew = self.norm_q75 + self.norm_q25 - 2 * self.norm_median
        if abs(skew) > 0.01:  # If distribution is notably skewed
            if skew > 0:
                alpha *= 1.2
            else:
                beta *= 1.2

        return max(0.1, alpha), max(0.1, beta)

    def _quantile_objective(self, params):
        """
        Enhanced objective function for beta parameter optimization
        incorporating confidence and multiple quantiles
        """
        alpha, beta = params

        # Basic quantile matching
        q25_err = (stats.beta.ppf(0.25, alpha, beta) - self.norm_q25) ** 2
        q75_err = (stats.beta.ppf(0.75, alpha, beta) - self.norm_q75) ** 2
        median_err = (stats.beta.ppf(0.5, alpha, beta) - self.norm_median) ** 4

        confidence_weight = 1 + self.expert.confidence

        # Basic error
        error = median_err**confidence_weight + q25_err + q75_err

        # Add tail quantile matching if available
        q05_err = (
            (stats.beta.ppf(0.05, alpha, beta) - self.norm_q05) ** 4
            if self.norm_q05
            else 0
        )
        error += q05_err**confidence_weight

        q95_err = (
            (stats.beta.ppf(0.95, alpha, beta) - self.norm_q95) ** 4
            if self.norm_q95
            else 0
        )
        error += q95_err**confidence_weight

        # Add shape penalty to prevent "fat" curves
        mode = (alpha - 1) / (alpha + beta - 2) if alpha + beta > 2 else 0.5
        mode_penalty = ((mode - self.norm_median) ** 2) * self.expert.confidence
        # mode_penalty = 0

        # Penalty for extreme parameters (prevents "fat" curves)
        param_penalty = 0.01 * (alpha**2 + beta**2) / (alpha + beta) ** 2
        # param_penalty = 0

        return error + mode_penalty + param_penalty

    def fit(self):
        """
        Fit beta distribution to expert opinions with improved optimization
        TODO - make sure fitting also takes into account IQR and median. The area under curve should be aroudn this and penalties when no.
        """
        # Get initial parameter estimates
        initial_params = self._estimate_initial_params()

        # Define bounds to prevent extreme values
        # Higher confidence = tighter bounds
        param_bound = 100 * (2 - self.expert.confidence)
        bounds = ((0.1, param_bound), (0.1, param_bound))

        # Try multiple optimization methods
        methods = ["L-BFGS-B", "SLSQP", "TNC"]
        best_result = None
        best_error = np.inf

        for method in methods:
            try:
                result = minimize(
                    self._quantile_objective,
                    x0=initial_params,
                    bounds=bounds,
                    method=method,
                )

                if result.success and result.fun < best_error:
                    best_result = result
                    best_error = result.fun

            except Exception as e:
                warnings.warn(f"Optimization failed with method {method}: {str(e)}")

        if best_result is None:
            raise RuntimeError("All optimization methods failed")

        self.beta_params = tuple(best_result.x)
        return self

    def sample(self, n_samples=1000):
        """Generate samples from the fitted distribution"""
        if self.beta_params is None:
            raise ValueError("Model must be fitted first")

        normalized_samples = stats.beta.rvs(*self.beta_params, size=n_samples)
        return normalized_samples * self.range + self.lower

    def pdf(self, x):
        """Get PDF values at given points"""
        if self.beta_params is None:
            raise ValueError("Model must be fitted first")

        x_norm = (np.array(x) - self.lower) / self.range
        return stats.beta.pdf(x_norm, *self.beta_params) / self.range

    def quantile(self, q):
        """Get quantile values"""
        if self.beta_params is None:
            raise ValueError("Model must be fitted first")

        return stats.beta.ppf(q, *self.beta_params) * self.range + self.lower

    def plot_pdf(
        self,
        num_points=1000,
        figsize=(12, 8),
        show_quantiles=True,
        show_confidence=True,
        title=None,
    ):
        """
        Plot the fitted PDF with optional quantile markers and confidence intervals.

        Parameters:
        -----------
        num_points : int
            Number of points to plot for the PDF curve
        figsize : tuple
            Figure size (width, height)
        show_quantiles : bool
            Whether to show the expert's quantile inputs
        show_confidence : bool
            Whether to show confidence intervals
        title : str
            Custom title for the plot
        """
        if self.beta_params is None:
            raise ValueError("Model must be fitted first")

        fig, ax = plt.subplots(figsize=figsize)

        # Generate points for PDF curve
        x = np.linspace(self.lower, self.upper, num_points)
        y = self.pdf(x)

        # Plot main PDF
        ax.plot(x, y, "b-", lw=2, label="Fitted Distribution")

        if show_quantiles:
            # Plot quantile markers
            quantiles = [
                (0.25, self.expert.q25, "Q25"),
                (0.5, self.expert.median, "Median"),
                (0.75, self.expert.q75, "Q75"),
            ]

            if self.expert.q05 is not None:
                quantiles.append((0.05, self.expert.q05, "Q05"))
            if self.expert.q95 is not None:
                quantiles.append((0.95, self.expert.q95, "Q95"))

            # Plot vertical lines for quantiles
            for q, val, label in quantiles:
                y_height = self.pdf(val)
                ax.vlines(val, 0, y_height, colors="gray", linestyles=":", alpha=0.5)
                ax.plot(val, y_height, "x", color="gray")
                # Add text annotation slightly above the point
                ax.annotate(
                    f"{label}\n({val:.2f})",
                    xy=(val, y_height),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color="gray",
                )
                # Plot distribution-derived quantiles
            dist_quantiles = [self.quantile(q) for q, _, _ in quantiles]
            for q, val in zip([q for q, _, _ in quantiles], dist_quantiles):
                y_height = self.pdf(val)
                ax.vlines(val, 0, y_height, colors="red", linestyles="--", alpha=1.0)
                ax.plot(val, y_height, "o", color="red")
                ax.annotate(
                    f"Dist {q:.2f}\n({val:.2f})",
                    xy=(val, y_height),
                    xytext=(-20, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color="black",
                )

        if show_confidence:
            # Show confidence intervals
            central_prob = self.expert.confidence
            lower_q = (1 - central_prob) / 2
            upper_q = 1 - lower_q

            ci_lower = self.quantile(lower_q)
            ci_upper = self.quantile(upper_q)

            # Add shaded confidence region
            x_conf = np.linspace(ci_lower, ci_upper, 100)
            y_conf = self.pdf(x_conf)
            ax.fill_between(
                x_conf,
                y_conf,
                alpha=0.2,
                color="g",
                label=f"{central_prob*100:.0f}% Confidence Interval",
            )

            # Add confidence bound markers
            ax.axvline(ci_lower, color="g", linestyle=":", alpha=0.5)
            ax.axvline(ci_upper, color="g", linestyle=":", alpha=0.5)

        # Add beta distribution parameters to plot
        param_text = f"β({self.beta_params[0]:.2f}, {self.beta_params[1]:.2f})"
        ax.text(
            0.95,
            0.95,
            param_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Customize plot
        if title is None:
            title = "Expert Opinion PDF with Fitted Beta Distribution"
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add a bit of padding to the x-axis
        x_padding = (self.upper - self.lower) * 0.05
        ax.set_xlim(self.lower - x_padding, self.upper + x_padding)

        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)

        return fig, ax


# Example usage with plotting
if __name__ == "__main__":
    # Create expert input
    expert_input = ExpertInput(
        median=3.25, q25=3.20, q75=3.3, q05=3.0, q95=3.35, confidence=0.5
    )

    # expert_input = ExpertInput(
    #     median=3.25, q25=3.20, q75=3.33, q05=3.10, q95=3.35, confidence=0.5
    # )
    # Fit model
    model = ExpertOpinion(expert_input)
    model.fit()

    # Create and show plot
    fig, ax = model.plot_pdf(
        show_quantiles=True,
        show_confidence=True,
    )
    plt.tight_layout()
    plt.show()

    banker = {
        "q05": 3.10,
        "q25": 3.20,
        "median": 3.25,
        "q75": 3.33,
        "q95": 3.35,
        "confidence": 0.50,
    }
    youtuber = {
        "q05": 2.90,
        "q25": 3.10,
        "median": 3.20,
        "q75": 3.28,
        "q95": 3.30,
        "confidence": 0.25,
    }
    fx_trader = {
        "q05": 3.20,
        "q25": 3.25,
        "median": 3.28,
        "q75": 3.35,
        "q95": 3.40,
        "confidence": 0.75,
    }

    experts = [banker, youtuber, fx_trader]
    dfs = []

    for expert in experts:
        e_input = ExpertInput(**expert)
        e_model = ExpertOpinion(e_input)
        e_model.fit()
        # Create and show plot
        fig, ax = e_model.plot_pdf(
            show_quantiles=True,
            show_confidence=True,
        )
        plt.tight_layout()
        plt.show()

        # generate samples
        df = pd.DataFrame(e_model.sample(10_000), columns=["fx_rate"])
        dfs.append(df)
        # ? how to merge to dfs?

    dfs = pd.concat(dfs, axis=0)
    dfs_summary = dfs.describe()
    print(dfs_summary)

    # Plot
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 6))
    sns.histplot(
        dfs["fx_rate"],
        bins=30,
        kde=True,
        alpha=0.75,
        edgecolor="white",
        line_kws={"color": "green", "lw": 2},
    )
    plt.xlabel("FX Rate")
    plt.ylabel("Count")
    plt.title("Aggregate of Expert Opinions", fontsize=16)
    plt.tight_layout()
    plt.show()
