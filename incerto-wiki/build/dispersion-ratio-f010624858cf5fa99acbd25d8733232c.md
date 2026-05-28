---
kernelspec:
  name: python3
  display_name: Python 3
---

# Dispersion Ratio Under Fat Tails

## Statement

Let $X_1,\dots,X_n$ be a sample, and compare the sample standard deviation with
absolute-deviation scale estimates.  The repository uses normal-scaled versions

$$
\operatorname{MeanAD}
=
\sqrt{\frac{\pi}{2}}\frac1n\sum_{i=1}^n |X_i-\bar X|,
$$

and

$$
\operatorname{MedianAD}
=
\frac{\operatorname{median}_i |X_i-\operatorname{median}(X)|}
{\Phi^{-1}(0.75)}.
$$

For Gaussian data, both are calibrated to estimate the same $\sigma$ as the
standard deviation.  Under fat-tailed or contaminated data, the ratios

$$
\frac{\operatorname{Std}}{\operatorname{MeanAD}},
\qquad
\frac{\operatorname{Std}}{\operatorname{MedianAD}}
$$

can rise because squared deviations react more strongly to extremes than
absolute deviations.

## Intuition

The standard deviation squares observations before averaging.  A single extreme
point can therefore dominate the estimate.  Mean absolute deviation still sees
the point, but only linearly.  Median absolute deviation may ignore the point
almost completely if it does not move the sample median or the middle absolute
deviation.

The ratio is a quick way to see disagreement between scale notions.  It should
not be treated as a universal tail classifier; it is a stress signal telling
you that the chosen norm matters.

## Proof

For $X\sim N(\mu,\sigma^2)$,

$$
\mathbb E|X-\mu|
=
\sigma\sqrt{\frac{2}{\pi}}.
$$

Thus multiplying the mean absolute deviation by $\sqrt{\pi/2}$ calibrates it
to $\sigma$ under a normal model.  Also,

$$
\operatorname{median}(|X-\mu|)
=
\sigma\Phi^{-1}(0.75),
$$

so dividing the raw median absolute deviation by $\Phi^{-1}(0.75)$ gives the
same normal calibration.

These identities are normal-model calibrations, not tail theorems.  If the
sample has extreme observations, the standard deviation's quadratic loss gives
those observations much more weight than either absolute-deviation statistic.
If the variance is infinite, as with a Cauchy distribution, the sample standard
deviation is not estimating a finite population quantity.

## Python

The simulation reproduces the Chapter 4 contrast with deterministic samples.

```{code-cell} python
:label: dispersion-ratio-python-check

import numpy as np
from scipy.stats import cauchy, t

from incerto.distributions import univariate_variance_gamma
from incerto.stats import mean_absolute_deviation, median_absolute_deviation


def dispersion_ratios(sample):
    std = np.std(sample)
    mean_ad = mean_absolute_deviation(sample)
    median_ad = median_absolute_deviation(sample)
    median_ratio = np.inf if median_ad == 0 else std / median_ad
    return std / mean_ad, median_ratio


rng = np.random.default_rng(20260523)
n = 200_000
samples = {
    "normal": rng.normal(size=n),
    "variance-gamma": univariate_variance_gamma.rvs(
        1.0, 0.0, 0.5, size=n, random_state=rng
    ),
    "student-t(3)": t.rvs(df=3, size=n, random_state=rng),
    "cauchy": cauchy.rvs(size=n, random_state=rng),
}

extreme = np.ones(n)
extreme[0] = 1_000_000.0
samples["single-outlier"] = extreme

print("sample           Std/MeanAD  Std/MedianAD")
for name, sample in samples.items():
    mean_ratio, median_ratio = dispersion_ratios(sample)
    print(f"{name:15s} {mean_ratio:8.2f} {median_ratio:8.2f}")
```

The normal ratios should be close to `1`.  The heavy-tailed and contaminated
samples produce larger ratios; the single-outlier example makes the median
absolute deviation equal to zero, so the second ratio is infinite.

## Caveats

- A high ratio is a diagnostic, not a fitted tail exponent.
- The mean absolute deviation and median absolute deviation are scaled here to
  match a Gaussian standard deviation.  Other conventions use unscaled values.
- Median absolute deviation can be too robust for payoff questions where a
  rare outlier dominates the quantity of interest.
- In a Cauchy sample, the ratios are sample-path objects; the population
  variance does not exist.
- Dependence and volatility clustering can move these ratios even when the
  one-step marginal distribution is unchanged.

## References

- Taleb, *Statistical Consequences of Fat Tails*, Chapter 4
  [@taleb2020scoft].
- Rousseeuw and Croux, "Alternatives to the Median Absolute Deviation"
  [@rousseeuw1993alternatives].

## Backlinks

- Depends on: [Variance Gamma Distribution](../distributions/variance-gamma.md)
  and [Body, Shoulders, and Tails](../methods/body-shoulder-tail.md).
- Used by: the planned Chapter 4 reading guide.
