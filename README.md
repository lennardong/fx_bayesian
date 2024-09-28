# Living the Good-Enough Life: lifestyle arbitrage through FX

## Whats the story?
Here's a basic story:

1. I have to move a non-trivial sum of $$ from SGD to MYR. 
2. This has to happen within the next $d$ months. 
3. When not used, the funds will be in some kind of financial product.

How do I make the money do some work for me in the process?

Breaking it down, there are 3 key components to this story:
- **SGD  Returns** : $t \cdot r_1$: This represents the returns if we keep the money in SGD for time $t$.
- **MYR Returns** : $(d-t) \cdot r_2(p)$: This is the returns from investing in MYR after exchanging at time $t$.
- **Exchange Rate Returns** : $\Delta e$: This captures the impact of exchange rate fluctuations.

Thus, the net capital gain can be expressed as:
$$
y = t \cdot r_1 + (d-t) \cdot r_2(p) + \Delta e
$$

Where:
- $y$: net capital gain (%)
- $t$: annualized duration of funds in SGD, simplified to 3 time horizons:
  - $t \in \{0. 0.25, 0.5\}$
- $d$: annualized duration until funds needed, where $d=0.5$
- $r_1$: annualized rate of return in SGD fund
- $r_2(p)$: annualized rate of return in MYR for product $p \in P$
  - $r_2: P \to \mathbb{R}$
  - $P = \{\text{money\_market}, \text{bonds}, \text{fixed\_deposit}\}$
- $\Delta e$: percentage change in exchange rate from present rate, caluclated as:$\frac{e_t - e_0}{e_0} \times 100\%$
  - $e_t$ is the exchange rate at time $t$
  - $e_0$ is the present exchange rate at time of solving. 

For simplicity, I disregard transfer and service fees because they're relatively small here. 

Overall, I think this captures it? Cool. Lets move on to turn this model into an optimization problem.

## What am I solving?

To maximize the net gain $y$, two key decisions need to be made:
1. When do I make FX transfer? ($t$)
2. What product do I use after the FX transfer? ($p$)

Re-written in principled form:
$$
\max_{t,p} y(t,p) = t \cdot r_1 + (d-t) \cdot r_2(p) + \Delta e
$$

Where the model constraints are:
- $t \in \{0, 0.25, 0.5\}$
- $p \in P = \{\text{money\_market}, \text{fixed\_deposit}\}$
- $0 \leq t \leq d$
- $d = 0.5$


## How do I solve this? 

To solve this, I need to build out the following: 
1. Model exchange rate $\Delta e$ using Bayesian methods for $t \in \{0, 0.25, 0.5\}$, accounting for different levels of confidence.
2. Estimate $r_1$ and $r_2(p)$ for each $p \in P$
3. Formulate as a discrete optimization problem
4. Solve for optimal $(t^*, p^*)$ using exhaustive search or monte-carlo methods:
   - For each $t \in \{0, 0.25, 0.5\}$ and $p \in P$, calculate $y(t,p)$
   - Select $(t^*, p^*)$ that maximizes $y(t,p)$
5. Conduct sensitivity analysis on key parameters
6. Validate results with historical data and out-of-sample testing

You are starting to sweat. This decision feels like a game of luck because WHO KNOWS WHAT THE FUTURE HOLDS?! 

## Task 1: Build model for $\Delta e$

### $e$: Exchange Rate Volatility and Projections 

**FX Patterns**

The first step is understanding the problem: how volatile are the FX rates? What exactly is at stake?  There are various challenges in this, but one significant challenge is modelling $\Delta e$ which is highly volatile at present.  As a quick check, of the FX movements for the past 100 days using data from `AlphaVantage` shows the craziness in this period: 

![FX Weekly](./assets/plot_historic_fx_weeklywhisker.png)

### Estimating $e_{future}$

The general form for a bayesian model is:

$$P(\theta|D) \propto P(D|\theta) \cdot P(\theta)$$

Where

- $\theta$ represents the parameters of our exchange rate model. 
- $D$ is the observed exchange rate data
- $P(\theta|D)$ is the posterior probability
- $P(D|\theta)$ is the likelihood
- $P(\theta)$ is the prior probability

Steps

- Define the prior beta distribution $P(\theta)$ based on weighted expert opinions
- Define the likelihood function $P(D|\theta)$ based on the observed exchange rate data
- Calculate the posterior distribution $P(\theta|D)$ using Bayes' theorem
- Use the posterior distribution to make predictions for future exchange rates

**Parameter Estimation of Expert Opinions**

Here are the various expert opinions:

```python
# fx_min, fx_max, fx_mean, weightage
dads_friend = (3.30, 3.35, 3.33, 0.5)
hsbc_rm = (3.25, 3.40, 3.28, 0.7)
bloomberg_futures = (3.33, 3.34, 3.335, 0.4)
```

Given the above, estimate $\alpha$ and $\beta$ for a beta distribution.

**Take 1: Estimation via Normalized Variance**
This method attempts to capture both the expert's expected value and their uncertainty in a single beta distribution. I normalize the expert's expected value to fit within the [0, 1] range of the beta distribution and estimate the variance based on the expert's confidence weight. The variance assumes a lowerbound of 0.25. A small factor added to 1 to avoid zero-division error


Normalize the mean:
$$\mu_{normalized} = \frac{F_{expected} - F_{low}}{F_{high} - F_{low}}$$
Estimate variance (adjusted based on feedback):
$$\sigma^2 = (1 - weight) \cdot 0.25$$
Use the method of moments to estimate α and β:
$$\alpha = \left(\frac{1 - \mu}{\sigma^2} - \frac{1}{\mu}\right) \cdot \mu^2$$
$$\beta = \alpha \cdot \frac{1}{\mu - 1}$$


```python
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
    var = (1.001-weight) * 0.25
    alpha = (
        (1-mu_norm) / var
        - 1 / mu_norm
    ) * mu_norm ** 2

    beta = alpha * (1 / mu_norm - 1)
    return alpha, beta
```
One key issue is we assume fx_low and fx_high is the absolute upper and lower bounds. In reality, this is more an expression of confidence.

**Take 2: Estimation via Concentration Parameter**

This method focuses on using the expert's confidence to directly influence the concentration of the beta distribution. I use the weight to determine a concentration parameter, which controls how peaked or spread out the distribution is.

For a beta distribution, the mean μ should match the normalized expected value (E), so: 

$$\mu = E_{norm} = \frac{\alpha}{\alpha + \beta}$$

The sum of α and β is also a measure of the concentration of the distribution.
The higher the concentration, the more likely we are to get values close to the mean from the distribution.
In simple terms, its a measure of "certainty" in the expert's opinion.

$$
\alpha + \beta  = c
$$
Therefore:
$\alpha = cE$, $\beta = c(1-E)$

Here are some key heuristics and interpretations:

- Higher c values (e.g., 100+) indicate high certainty, resulting in narrow, peaked distributions.
- Lower c values (e.g., 1-10) suggest low certainty, leading to flatter, more spread-out distributions.
- A c value of 2 represents a uniform distribution, indicating maximum uncertainty.
- As c approaches infinity, the distribution approximates a normal distribution.

```python
def estimate_parameters_v3(fx_low, fx_high, fx_expected, weight):
    range_width = fx_high - fx_low
    normalized_expected = (fx_expected - fx_low) / range_width
    
    # Use weight directly to control the concentration
    concentration = 10 + 90 * weight  # This will give c values between 10 and 100
    
    alpha = concentration * normalized_expected
    beta = concentration * (1 - normalized_expected)
    
    return alpha, beta
```

**Take 3: Estimation via Simulation**
TODO
ask claude for MLE: fx_min and max should not be absolute bounds. 
Instead, they are an expression of a 95% confidence interval in the bounds. 

does the model account for this? 
if not update. 
also, add docstrings to the methods to explain what they do. 
functions should be simplified to methods under the class

Or: try estimation via synthetic data: https://search.r-project.org/CRAN/refmans/EnvStats/html/ebeta.html


### $r$: Expected Returns from Investment Products

## Building a Model
