---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: dispersion-ratio
    type: empirical-example
    depends_on:
      - variance-gamma
      - body-shoulder-tail
    tags:
      - robust-scale
      - tail-risk
---

# Dispersion Ratio Under Fat Tails

## Statement

Let $x_1,\dots,x_n$ be observed data, and compare the sample standard
deviation with absolute-deviation scale estimates.  The repository uses
normal-scaled versions

$$
\operatorname{MeanAD}
=
\sqrt{\frac{\pi}{2}}\frac1n\sum_{i=1}^n |x_i-\bar x|,
$$

and

$$
\operatorname{MedianAD}
=
\frac{\operatorname{median}_i |x_i-\operatorname{median}(x)|}
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

The sample notation $X_1,\dots,X_n$ and expectation notation follow the shared
[notation table](../../notation/index.md).  The scale symbol $\sigma$ and the
names `MeanAD` and `MedianAD` are local conventions for this page.

## Examples

- For Gaussian data, the standard deviation, normal-scaled mean absolute
  deviation, and normal-scaled median absolute deviation target the same
  $\sigma$, so the ratios should be near $1$.
- For Student-t or Pareto samples, large observations can push the standard
  deviation much higher than the absolute-deviation scales.
- In a single-outlier sample, the median absolute deviation can be zero while
  the standard deviation is large, so the second ratio is infinite.

## Calibration

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

## Simulation

The simulation compares Gaussian, Student-t, and Pareto samples across sample
sizes.  It reports repeated-simulation medians of the two dispersion ratios.

```{code-cell} python
:label: dispersion-ratio-simulation
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

from incerto.distributions import pareto_type1
from incerto.figures import FIGURE_SIZES, set_theme, style_axes
from incerto.stats import mean_absolute_deviation, median_absolute_deviation

set_theme()


def dispersion_ratios(sample):
    std = np.std(sample)
    mean_ad = mean_absolute_deviation(sample)
    median_ad = median_absolute_deviation(sample)
    median_ratio = np.inf if median_ad == 0 else std / median_ad
    return std / mean_ad, median_ratio


def ratio_quantiles(draw, n, reps=240, seed=20260617):
    rng = np.random.default_rng(seed)
    ratios = np.empty((reps, 2))
    for i in range(reps):
        ratios[i] = dispersion_ratios(draw(rng, n))
    return np.quantile(ratios, [0.5, 0.9], axis=0)


draws = {
    "normal": lambda rng, n: rng.normal(size=n),
    "student-t(3)": lambda rng, n: rng.standard_t(df=3, size=n),
    "Pareto(1.5)": lambda rng, n: pareto_type1.rvs(1.5, size=n, random_state=rng),
}

n_values = np.array([200, 1_000, 5_000, 20_000])
results = {
    name: np.array(
        [ratio_quantiles(draw, n, seed=20260617 + i * 100 + n) for i, n in enumerate(n_values)]
    )
    for name, draw in draws.items()
}

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"])

for name, qs in results.items():
    axes[0].plot(n_values, qs[:, 0, 0], marker="o", label=name)
    axes[1].plot(n_values, qs[:, 0, 1], marker="o", label=name)

axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.1)
axes[0].set_xscale("log")
axes[0].set_xlabel("sample size n")
axes[0].set_ylabel("median Std/MeanAD")
axes[0].set_title("Mean absolute deviation ratio")
axes[0].legend()

axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.1)
axes[1].set_xscale("log")
axes[1].set_xlabel("sample size n")
axes[1].set_ylabel("median Std/MedianAD")
axes[1].set_title("Median absolute deviation ratio")
axes[1].legend()

style_axes(axes, grid_axis="both")
plt.show()
```

**What to notice.** The Gaussian ratios stay near one by construction.  The
Student-t and Pareto ratios are larger because the standard deviation is more
sensitive to extremes than the absolute-deviation scales.  The ratio is a
stress signal, not a fitted tail exponent.

## Caveats

- A high ratio is a diagnostic, not a fitted tail exponent.
- The mean absolute deviation and median absolute deviation are scaled here to
  match a Gaussian standard deviation.  Other conventions use unscaled values.
- Median absolute deviation can be too robust for payoff questions where a
  rare outlier dominates the quantity of interest.
- In an infinite-variance sample, the ratios are sample-path objects; the
  population variance does not exist.
- Dependence and volatility clustering can move these ratios even when the
  one-step marginal distribution is unchanged.

## References

- Rousseeuw and Croux, "Alternatives to the Median Absolute Deviation"
  [@rousseeuw1993alternatives].

## Backlinks

- Depends on: [Variance Gamma Distribution](../distributions/variance-gamma.md)
  and [Body, Shoulders, and Tails](../methods/body-shoulder-tail.md).
- Used by: robust-scale examples and tail-risk diagnostics.

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/examples/dispersion-ratio.md`. Last verified: 2026-06-17. Checked against cited calibration identities and executable dispersion-ratio simulations.
:::
<!-- incerto-provenance:end -->
