import numpy as np
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings


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
        self.lower = min(all_quantiles)
        self.upper = max(all_quantiles)
        self.range = self.upper - self.lower

        # Normalize all quantiles
        self.norm_q25 = (self.expert.q25 - self.lower) / self.range
        self.norm_q75 = (self.expert.q75 - self.lower) / self.range
        self.norm_median = (self.expert.median - self.lower) / self.range

        if self.expert.q05 is not None:
            self.norm_q05 = (self.expert.q05 - self.lower) / self.range
        if self.expert.q95 is not None:
            self.norm_q95 = (self.expert.q95 - self.lower) / self.range

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
        median_err = (stats.beta.ppf(0.5, alpha, beta) - self.norm_median) ** 2

        # Weight median more heavily based on confidence
        median_weight = 1 + self.expert.confidence

        # Basic error
        error = median_err * median_weight + q25_err + q75_err

        # Add tail quantile matching if available
        if hasattr(self, "norm_q05"):
            q05_err = (stats.beta.ppf(0.05, alpha, beta) - self.norm_q05) ** 2
            error += q05_err

        if hasattr(self, "norm_q95"):
            q95_err = (stats.beta.ppf(0.95, alpha, beta) - self.norm_q95) ** 2
            error += q95_err

        # Add shape penalty to prevent "fat" curves
        mode = (alpha - 1) / (alpha + beta - 2) if alpha + beta > 2 else 0.5
        mode_penalty = ((mode - self.norm_median) ** 2) * self.expert.confidence

        # Penalty for extreme parameters (prevents "fat" curves)
        param_penalty = 0.01 * (alpha**2 + beta**2) / (alpha + beta) ** 2

        return error + mode_penalty + param_penalty

    def fit(self):
        """
        Fit beta distribution to expert opinions with improved optimization
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


# Example usage
expert_input = ExpertInput(
    median=3.25, q25=3.22, q75=3.28, q05=3.20, q95=3.35, confidence=0.8
)

model = ExpertOpinion(expert_input)
model.fit()

# Get some statistics
samples = model.sample(1000)
predicted_median = model.quantile(0.5)
predicted_q25 = model.quantile(0.25)
predicted_q75 = model.quantile(0.75)
print(samples)
