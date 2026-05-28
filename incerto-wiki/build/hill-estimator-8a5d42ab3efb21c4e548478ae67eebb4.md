# Hill Estimator

## Statement

Let $X_1,\dots,X_n$ be positive iid observations with a Pareto-type right tail.
Write the ascending order statistics as

$$
X_{1:n}\le\cdots\le X_{n:n}.
$$

For $1\le k<n$, the Hill estimator of the extreme-value index $\xi$ is

$$
\widehat\xi_{k,n}
=
\frac{1}{k}\sum_{i=1}^k
\log X_{n-i+1:n}
-
\log X_{n-k:n}.
$$

For Pareto-type tails with $\xi=1/\alpha$, the corresponding tail-exponent
estimate is

$$
\widehat\alpha_{k,n}=\frac{1}{\widehat\xi_{k,n}}.
$$

Under standard iid, regular variation, and threshold-growth conditions
($k\to\infty$ and $k/n\to0$), the Hill estimator is consistent for $\xi$.

## Intuition

The estimator looks only at the largest $k$ observations and asks how far they
sit above the threshold $X_{n-k:n}$ on a log scale. For an exact Pareto tail,
log-excesses over a threshold are exponential with mean $1/\alpha$. The Hill
estimate is the sample mean of those log-excesses.

The hard part is not the formula; it is choosing $k$. Too small, and the
estimate is noisy because it uses very few extremes. Too large, and the
estimate is biased because body observations contaminate the tail fit.

## Proof

For an exact Pareto Type I tail with cutoff $u$,

$$
\mathbb P(X>x\mid X>u)=\left(\frac{u}{x}\right)^\alpha,
\qquad x\ge u.
$$

Therefore, for $Y=\log(X/u)$ and $y\ge0$,

$$
\mathbb P(Y>y\mid X>u)
=
\mathbb P(X>ue^y\mid X>u)
=
e^{-\alpha y}.
$$

So $Y$ is exponential with mean $1/\alpha=\xi$. If the threshold is fixed and
the exceedances are exact Pareto, the average log-excess estimates $\xi$.

The Hill estimator replaces the fixed threshold by the random order statistic
$X_{n-k:n}$ and uses the $k$ observations above it. Consistency for regularly
varying tails is a standard extreme-value result under intermediate-threshold
conditions. This page cites the general theorem rather than re-proving the
empirical-process argument.

## Python

```python
import numpy as np
from scipy.stats import pareto
from incerto.estimators import hill_alpha_estimator, hill_stability

rng = np.random.default_rng(20260523)
alpha = 1.5
sample = pareto.rvs(b=alpha, size=5000, random_state=rng)

print(hill_alpha_estimator(sample, k=100))

stability = hill_stability(sample, k_values=range(20, 500, 20))
print(np.column_stack([stability["k"], stability["alpha"]])[:5])
```

```{raw} html
<div class="incerto-widget incerto-widget-grid" data-incerto-widget="hill-stability">
  <div class="incerto-widget-controls">
    <div class="incerto-widget-control-row">
      <label for="hill-alpha">alpha</label>
      <output id="hill-alpha-value" data-hill-alpha-value for="hill-alpha">1.50</output>
      <input id="hill-alpha" data-hill-alpha type="range" min="0.80" max="3.00" step="0.05" value="1.50">
    </div>
    <div class="incerto-widget-control-row">
      <label for="hill-n">draws</label>
      <output id="hill-n-value" data-hill-n-value for="hill-n">2500</output>
      <input id="hill-n" data-hill-n type="range" min="1000" max="6000" step="500" value="2500">
    </div>
  </div>
  <div class="incerto-widget-stats" aria-live="polite">
    <div class="incerto-widget-stat">
      <strong>k=25</strong>
      <span data-hill-k-small></span>
    </div>
    <div class="incerto-widget-stat">
      <strong>k=100</strong>
      <span data-hill-k-mid></span>
    </div>
    <div class="incerto-widget-stat">
      <strong>k=300</strong>
      <span data-hill-k-large></span>
    </div>
  </div>
  <figure class="incerto-widget-plot">
    <svg data-hill-plot aria-label="Hill alpha estimate over k"></svg>
  </figure>
</div>
```

## Caveats

- A flat-looking Hill plot is evidence, not proof, of a Pareto tail.
- The estimator assumes positive tail observations. For losses or two-sided
  returns, transform the data to the relevant positive tail first.
- Dependence, rounding, truncation, and mixtures can produce misleading
  stability regions.
- Reporting one $\widehat\alpha$ without the chosen threshold, $k$, and a
  sensitivity plot hides the main modeling choice.

## References

- Hill, "A Simple General Approach to Inference About the Tail of a
  Distribution" [@hill1975simple].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Taleb, *Statistical Consequences of Fat Tails*, Chapter 5
  [@taleb2020scoft].

## Backlinks

- Depends on: [Regular Variation](../theorems/regular-variation.md) and
  [Pareto Distribution](../distributions/pareto.md).
- Used by: [Pareto Moment Existence](../theorems/pareto-moment-existence.md),
  [S&P 500 Tail Fit](../examples/sp500-tail.md), and the
  [Chapter 5 reading guide](../../reading-guides/taleb-scoft/ch5.md).
