"""
# Quantile-Based Expert Opinion Modeling

## Overview
The goal is to transform expert opinions about future values (like FX rates, stock prices, or any financial metric) into a proper probability distribution. Instead of asking experts for complex statistical parameters, we ask them for intuitive quantile estimates - essentially, their views on different scenarios and their likelihood.

## Core Approach
Our method uses five key pieces of information from experts:
- Median (50th percentile): Their "best guess" of the future value
- 25th and 75th percentiles: Values they think have a 1-in-4 chance of occurring
- 5th and 95th percentiles (optional): Their views on "extreme but plausible" scenarios
- Confidence level: How sure they are about their estimates

We then fit this information to a Beta distribution, which is particularly good at modeling bounded random variables and can capture various shapes (symmetric, skewed, peaked, or flat).

## Why This Works
1. **Intuitive Input**: Experts are more comfortable and accurate when thinking in terms of scenarios ("I'm 95% sure it won't go above X") rather than statistical parameters like standard deviations or skewness.

2. **Mathematical Rigor**: While the inputs are intuitive, the underlying mathematics is robust:
   - Beta distributions can represent a wide range of shapes
   - The fitting process ensures the distribution respects all the expert's views
   - The resulting distribution is smooth and well-behaved, suitable for further mathematical operations

3. **Confidence Integration**: The expert's confidence level helps control how "peaked" or "spread out" the final distribution is, reflecting their uncertainty.

## Applications
The resulting distributions can be used for:
- Risk assessment and scenario analysis
- Option pricing and derivatives modeling
- Portfolio optimization
- Trading strategy development
- Consensus building when multiple experts are involved

## Advantages Over Other Approaches
Unlike previous methods that used triangular distributions or kernel density estimation (KDE), our quantile-based approach:
- Is more intuitive for experts to provide inputs
- Directly captures tail risks (through 5th and 95th percentiles)
- Produces a parametric distribution that's easy to work with mathematically
- Avoids the smoothing and bandwidth selection issues associated with KDE
- Results in more stable and reliable distributions

## Technical Implementation
We use numerical optimization to find Beta distribution parameters that best match the expert's quantile estimates. The optimization includes:
- Smart initialization based on the provided quantiles
- Multiple optimization methods to ensure robust fitting
- Penalties to prevent unrealistic distribution shapes
- Built-in validation of expert inputs for consistency

## Practical Usage
The model provides various ways to work with the fitted distribution:
- Generate random samples for simulation
- Calculate probabilities for any range of values
- Visualize the distribution and compare it with expert inputs
- Combine opinions from multiple experts

This approach bridges the gap between expert knowledge and quantitative modeling, providing a rigorous yet practical way to incorporate subjective views into financial analysis.
"""

'''
# Understanding Confidence in Expert Opinion Models

## Intuitive Understanding

Think of confidence as "how sure are you about your estimates?" It affects the shape of our probability distribution in two key ways:

1. **Peak Sharpness**: 
   - High confidence → Sharper, more concentrated distribution
   - Low confidence → Flatter, more spread-out distribution

2. **Tail Behavior**:
   - High confidence → Thinner tails (extreme outcomes less likely)
   - Low confidence → Fatter tails (extreme outcomes more likely)

### Real-World Example

Imagine two FX traders providing forecasts for EUR/USD in one month:

**Trader A (High Confidence, 0.9)**
- Current rate: 1.10
- "I'm very confident the rate will be between 1.08 and 1.12"
- Median: 1.10
- Q25-Q75: 1.09-1.11
- Q05-Q95: 1.08-1.12

**Trader B (Low Confidence, 0.5)**
- Current rate: 1.10
- "There's a lot of uncertainty right now..."
- Median: 1.10
- Q25-Q75: 1.07-1.13
- Q05-Q95: 1.05-1.15

Even with the same median, their different confidence levels result in very different distributions.

## Mathematical Implementation

Our implementation translates confidence into mathematical constraints in several ways:

### 1. Parameter Bounds Control

```python
def _get_parameter_bounds(self):
    """Adjust parameter bounds based on confidence"""
    # Higher confidence = tighter bounds on parameters
    max_param = 100 * (2 - self.expert.confidence)
    return ((0.1, max_param), (0.1, max_param))
```

For example:
- Confidence = 0.9 → max_param = 110 (tight bounds)
- Confidence = 0.5 → max_param = 150 (looser bounds)

### 2. Objective Function Weighting

```python
def _quantile_objective(self, params):
    alpha, beta = params
    
    # Basic errors
    q25_err = (stats.beta.ppf(0.25, alpha, beta) - self.norm_q25) ** 2
    q75_err = (stats.beta.ppf(0.75, alpha, beta) - self.norm_q75) ** 2
    median_err = (stats.beta.ppf(0.5, alpha, beta) - self.norm_median) ** 2
    
    # Weight median more heavily based on confidence
    median_weight = 1 + self.expert.confidence
    
    # Confidence affects shape penalty
    shape_penalty = ((alpha + beta) * (1 - self.expert.confidence))
    
    return (median_err * median_weight + 
            q25_err + q75_err + 
            shape_penalty)
```

### 3. Initial Parameter Estimation

```python
def _estimate_initial_params(self):
    # Use IQR to estimate spread
    iqr = self.norm_q75 - self.norm_q25
    
    # Estimate variance using IQR and confidence
    # Lower confidence = higher variance
    variance = (iqr / 1.35) ** 2 * (2 - self.expert.confidence)
    
    # Method of moments adjusted by confidence
    mean = self.norm_median
    mean_term = mean * (1 - mean) / variance - 1
    
    alpha = mean * mean_term
    beta = (1 - mean) * mean_term
    
    return max(0.1, alpha), max(0.1, beta)
```

## Visual Demonstration

Let's add a method to visualize how confidence affects the distribution:

```python
def plot_confidence_comparison(self, confidences=[0.3, 0.6, 0.9], figsize=(12, 6)):
    """Compare distributions with different confidence levels"""
    fig, ax = plt.subplots(figsize=figsize)
    
    for conf in confidences:
        # Create temporary model with this confidence
        temp_expert = ExpertInput(
            median=self.expert.median,
            q25=self.expert.q25,
            q75=self.expert.q75,
            q05=self.expert.q05,
            q95=self.expert.q95,
            confidence=conf
        )
        temp_model = ExpertOpinion(temp_expert)
        temp_model.fit()
        
        # Plot PDF
        x = np.linspace(self.lower, self.upper, 1000)
        y = temp_model.pdf(x)
        ax.plot(x, y, label=f'Confidence = {conf:.1f}')
    
    ax.set_title('Impact of Confidence on Distribution Shape')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    return fig, ax
```

## Mathematical Effects

Confidence affects our Beta distribution in three key ways:

1. **Parameter Magnitude**:
   - Higher confidence → Larger α and β parameters
   - This creates a more peaked distribution
   
2. **Parameter Ratio**:
   - Confidence affects how closely we stick to the expert's median
   - Higher confidence puts more weight on hitting the exact median

3. **Optimization Constraints**:
   - Higher confidence tightens the allowable parameter space
   - This prevents the optimization from finding solutions that are too spread out

## Practical Impact

The confidence parameter helps us:
1. Distinguish between strong and weak opinions
2. Properly weight different expert opinions when combining them
3. Adjust risk measures based on certainty levels
4. Create more realistic scenario analyses

https://claude.ai/chat/aee5df6a-d673-4c3f-8e2a-84c4d931d278
'''

import numpy as np
from scipy import stats
from scipy.optimize import minimize, curve_fit
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
        # self.lower = min(all_quantiles) - 0.05
        # self.upper = max(all_quantiles) + 0.05
        # self.range = self.upper - self.lower
        self._calculate_bounds()

        # Normalize all quantiles
        self.norm_q25 = (self.expert.q25 - self.lower) / self.range
        self.norm_q75 = (self.expert.q75 - self.lower) / self.range
        self.norm_median = (self.expert.median - self.lower) / self.range

        if self.expert.q05 is not None:
            self.norm_q05 = (self.expert.q05 - self.lower) / self.range
        if self.expert.q95 is not None:
            self.norm_q95 = (self.expert.q95 - self.lower) / self.range

    def _calculate_bounds(self):
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

    def _calculate_bounds_linear3(self):
        # Left side (Q05 to median)
        x_left = np.array([0.05, 0.25, 0.5])
        y_left = np.array([self.expert.q05, self.expert.q25, self.expert.median])
        slope_left, intercept_left = np.polyfit(x_left, y_left, 1)

        # Right side (median to Q95)
        x_right = np.array([0.5, 0.75, 0.95])
        y_right = np.array([self.expert.median, self.expert.q75, self.expert.q95])
        slope_right, intercept_right = np.polyfit(x_right, y_right, 1)

        # Extrapolate to get lower and upper bounds
        self.lower = slope_left * 0 + intercept_left
        self.upper = slope_right * 1 + intercept_right
        self.range = self.upper - self.lower

    def _calculate_bounds_logreg(self):
        x_left = [0.05, 0.25, 0.5]
        y_left = [self.expert.q05, self.expert.q25, self.expert.median]
        params_left, _ = curve_fit(logistic_function, x_left, y_left, p0=[1, 1, 0.25])

        x_right = [0.5, 0.75, 0.95]
        y_right = [self.expert.median, self.expert.q75, self.expert.q95]
        params_right, _ = curve_fit(
            logistic_function, x_right, y_right, p0=[1, 1, 0.75]
        )

        self.lower = logistic_function(0, *params_left)
        self.right = logistic_function(1, *params_right)
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

        # Weight median more heavily based on confidence
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
                ax.vlines(val, 0, y_height, colors="r", linestyles="--", alpha=0.5)
                ax.plot(val, y_height, "ro")
                # Add text annotation slightly above the point
                ax.annotate(
                    f"{label}\n({val:.2f})",
                    xy=(val, y_height),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
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


from scipy.stats import norm
from scipy.optimize import minimize


class MixtureNormalExpertOpinion:
    def __init__(self, expert_input: ExpertInput):
        self.expert = expert_input
        self.mixture_params = None
        self._calculate_bounds()

    def _objective_function(self, params):
        mu1, sigma1, mu2, sigma2, w = params
        mixture_cdf = lambda x: w * norm.cdf(x, mu1, sigma1) + (1 - w) * norm.cdf(
            x, mu2, sigma2
        )

        errors = []
        if hasattr(self.expert, "q05"):
            errors.append((mixture_cdf(self.expert.q05) - 0.05) ** 2)
        errors.append((mixture_cdf(self.expert.q25) - 0.25) ** 2)
        errors.append((mixture_cdf(self.expert.median) - 0.5) ** 2)
        errors.append((mixture_cdf(self.expert.q75) - 0.75) ** 2)
        if hasattr(self.expert, "q95"):
            errors.append((mixture_cdf(self.expert.q95) - 0.95) ** 2)

        return sum(errors)

    def fit(self):
        initial_guess = [
            self.expert.median,
            (self.expert.q75 - self.expert.q25) / 2,
            self.expert.median,
            (self.expert.q75 - self.expert.q25),
            0.5,
        ]
        bounds = [
            (self.lower, self.upper),
            (0, None),
            (self.lower, self.upper),
            (0, None),
            (0, 1),
        ]

        result = minimize(
            self._objective_function, initial_guess, bounds=bounds, method="L-BFGS-B"
        )
        self.mixture_params = result.x
        return self

    def pdf(self, x):
        mu1, sigma1, mu2, sigma2, w = self.mixture_params
        return w * norm.pdf(x, mu1, sigma1) + (1 - w) * norm.pdf(x, mu2, sigma2)

    # Add methods for sampling, plotting, etc.


# Example usage with plotting
if __name__ == "__main__":
    # Create expert input
    expert_input = ExpertInput(
        median=3.25, q25=3.22, q75=3.28, q05=3.1, q95=3.55, confidence=0.5
    )

    # Fit model
    model = ExpertOpinion(expert_input)
    model.fit()

    # Create and show plot
    fig, ax = model.plot_pdf(
        show_quantiles=True,
        show_confidence=True,
        title="Expert Opinion: Asset Price Distribution",
    )
    plt.tight_layout()
    plt.show()
