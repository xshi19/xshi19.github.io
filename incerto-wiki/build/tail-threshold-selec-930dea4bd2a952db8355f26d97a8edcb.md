---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: tail-threshold-selection
    type: method
    depends_on:
      - extreme-value-index
      - hill-estimator
      - mean-excess-function
      - generalized-pareto
    tags:
      - estimation
      - diagnostics
      - tail-risk
---

# Tail Threshold Selection

## Statement

Tail threshold selection is the modeling step that chooses where the body of a
sample ends and where a tail estimator or peaks-over-threshold model begins.
For a threshold $u$, the exceedance sample is

$$
\{x_i-u:x_i>u\}.
$$

For a [Hill estimator](hill-estimator.md), the equivalent tuning parameter is
the number $k$ of upper order statistics, with empirical threshold
$x_{n-k:n}$.  Hill is designed for positive, regularly varying right tails
with extreme-value index $\xi>0$; for a Pareto-type tail,
$\alpha=1/\xi$.  A threshold selection report should include at least:

- the candidate thresholds or $k$ values;
- the number of exceedances retained;
- Hill or EVI stability over the candidate range;
- mean-excess or GPD-shape stability where relevant;
- the reason the selected range was kept or rejected.

The goal is not to find a universally correct threshold.  It is to make the
bias-variance tradeoff visible.

The symbols $u$, $k$, $x_{n-k:n}$, and $\xi$ follow the shared
[notation table](../../notation/index.md).

## Bias-variance intuition

Every tail model asks the same awkward question: how far out is "tail"?  Set
the threshold too low and the estimator is biased by body observations.  Set
it too high and the estimator is dominated by too few extremes.  A good
threshold analysis therefore looks for a region where several diagnostics stop
moving violently while enough exceedances remain to estimate anything at all.

This page is distinct from the
[body-shoulder-tail diagnostic](body-shoulder-tail.md).  Body/shoulder geometry
describes how a distribution's density responds to variance mixing.  Threshold
selection is an empirical modeling decision for tail estimation.

## Diagnostics

For a one-sided positive sample:

1. Choose a grid of candidate thresholds, often empirical quantiles.
2. Record exceedance counts at each threshold.
3. Plot Hill estimates over $k$ or threshold values.
4. Plot the [mean-excess function](../theorems/mean-excess-function.md) for the
   same threshold range.
5. If using a GPD model, fit shape and scale over multiple thresholds.
6. Prefer a range where estimates are reasonably stable and exceedance counts
   are not too small.

The accepted range should be reported, not hidden.  Downstream quantities such
as [moment existence](../theorems/pareto-moment-existence.md), return levels, or
expected shortfall can be highly sensitive to $\xi$.  Variance existence
changes at $\xi=1/2$, while mean and expected-shortfall existence change at
$\xi=1$; high return-level estimates are sensitive to $\xi$ but do not have
mathematical singularities specifically at those two values.

## Synthetic worked example

The example below constructs a sample with a bounded body below the true tail
threshold $u=3$ and a [Pareto tail](../distributions/pareto.md) above it.  The
vertical line marks the construction's true threshold; in real data this line
is not known.

```{code-cell} python
:label: tail-threshold-selection-diagnostics
:tags: [hide-input]

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

from incerto.distributions import pareto_type1
from incerto.estimators import hill_stability, mean_excess
from incerto.figures import COLORS, FIGURE_SIZES, set_theme, style_axes

set_theme()

rng = np.random.default_rng(20260617)
n = 30_000
tail_probability = 0.18
alpha = 1.7
xi_true = 1 / alpha
true_threshold = 3.0

is_tail = rng.random(n) < tail_probability
sample = np.empty(n)
sample[~is_tail] = 1 + 2 * rng.random(np.sum(~is_tail))
sample[is_tail] = true_threshold * pareto_type1.rvs(
    alpha,
    size=np.sum(is_tail),
    random_state=rng,
)

ks = np.unique(np.geomspace(20, int(0.35 * n), 100).astype(int))
hill = hill_stability(sample, ks)
threshold_by_k = np.sort(sample)[-(ks + 1)]

thresholds = np.quantile(sample, np.linspace(0.75, 0.99, 35))
me = mean_excess(sample, thresholds, min_exceedances=80)
theoretical_me_thresholds = thresholds[thresholds >= true_threshold]
theoretical_mean_excess = xi_true * theoretical_me_thresholds / (1 - xi_true)

gpd_thresholds = np.quantile(sample, np.linspace(0.80, 0.985, 18))
gpd_xi = []
gpd_modified_scale = []
gpd_exceedances = []
fit_warning_count = 0
for threshold in gpd_thresholds:
    excesses = sample[sample > threshold] - threshold
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xi_hat, loc_hat, beta_hat = genpareto.fit(excesses, floc=0.0)
    fit_warning_count += len(caught)
    if abs(loc_hat) > 1e-10:
        raise RuntimeError("GPD fit did not respect the fixed zero location")
    gpd_xi.append(xi_hat)
    gpd_modified_scale.append(beta_hat - xi_hat * threshold)
    gpd_exceedances.append(excesses.size)

gpd_xi = np.array(gpd_xi)
gpd_modified_scale = np.array(gpd_modified_scale)
gpd_exceedances = np.array(gpd_exceedances)

accepted_mask = (gpd_thresholds >= true_threshold) & (gpd_exceedances >= 700)
accepted_thresholds = gpd_thresholds[accepted_mask]
accepted_counts = gpd_exceedances[accepted_mask]
accepted_xi = gpd_xi[accepted_mask]
if accepted_thresholds.size == 0:
    raise RuntimeError("No thresholds satisfied the accepted-range rule")

report_threshold = accepted_thresholds[0]
report_tail_sample = sample[sample > report_threshold]
report_k = report_tail_sample.size
xi_report = np.mean(np.log(report_tail_sample / report_threshold))
xi_se = xi_report / np.sqrt(report_k)
xi_ci = xi_report + np.array([-1.96, 1.96]) * xi_se

fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["three_panel"])

axes[0].plot(threshold_by_k, hill["alpha"], color=COLORS["green"])
axes[0].axhline(alpha, color=COLORS["accent"], linestyle="--", linewidth=1.1)
axes[0].axvline(true_threshold, color="black", linestyle=":", linewidth=1.1)
axes[0].set_xlabel("empirical threshold")
axes[0].set_ylabel(r"Hill $\widehat\alpha$")
axes[0].set_title("Hill stability")

axes[1].plot(me["threshold"], me["mean_excess"], color=COLORS["teal"])
axes[1].plot(
    theoretical_me_thresholds,
    theoretical_mean_excess,
    color=COLORS["accent"],
    linestyle="--",
    linewidth=1.1,
    label="exact Pareto line",
)
axes[1].axvline(true_threshold, color="black", linestyle=":", linewidth=1.1)
axes[1].set_xlabel("threshold u")
axes[1].set_ylabel("mean excess")
axes[1].set_title("Mean-excess diagnostic")
axes[1].legend()
count_axis = axes[1].twinx()
count_axis.plot(
    me["threshold"],
    me["exceedances"],
    color=COLORS["muted"],
    linestyle=":",
    linewidth=1.1,
    label="exceedances",
)
count_axis.set_ylabel("exceedances")

xi_line = axes[2].plot(gpd_thresholds, gpd_xi, marker="o", label=r"$\hat\xi$")
modified_axis = axes[2].twinx()
modified_line = modified_axis.plot(
    gpd_thresholds,
    gpd_modified_scale,
    marker="o",
    color=COLORS["umber"],
    label=r"$\hat\beta-\hat\xi u$",
)
axes[2].axhline(xi_true, color=COLORS["accent"], linestyle="--", linewidth=1.1)
modified_axis.axhline(0.0, color=COLORS["umber"], linestyle="--", linewidth=1.0)
axes[2].axvline(true_threshold, color="black", linestyle=":", linewidth=1.1)
axes[2].set_xlabel("threshold u")
axes[2].set_ylabel(r"$\hat\xi$")
modified_axis.set_ylabel(r"modified scale")
axes[2].set_title("GPD stability")
axes[2].legend(
    xi_line + modified_line,
    [line.get_label() for line in xi_line + modified_line],
)

style_axes([*axes, count_axis, modified_axis], grid_axis="both")
plt.show()

print("Threshold selection report")
print(
    f"Accepted threshold range: {accepted_thresholds[0]:.3f} "
    f"to {accepted_thresholds[-1]:.3f}"
)
print(
    f"Exceedances across range: {int(accepted_counts[0])} "
    f"to {int(accepted_counts[-1])}"
)
print(
    f"Tail estimate at u={report_threshold:.3f}: "
    f"xi={xi_report:.3f} "
    f"(approx. 95% Hill interval {xi_ci[0]:.3f}, {xi_ci[1]:.3f}); "
    f"alpha={1 / xi_report:.3f}"
)
print(
    f"GPD xi sensitivity across accepted range: "
    f"{np.min(accepted_xi):.3f} to {np.max(accepted_xi):.3f}"
)
print(f"GPD fit warnings captured: {fit_warning_count}")
```

**What to notice.** Thresholds below the true tail cutoff pull body data into
the diagnostics.  Thresholds far above the cutoff leave fewer exceedances and
more noise.  The theoretical mean-excess line is only drawn for thresholds at
or above the true Pareto cutoff.  In the GPD panel, $\hat\beta-\hat\xi u$ is
the threshold-stable modified scale; the simpler $\hat\beta/u\approx\hat\xi$
check is specific to an exact Pareto tail.  The practical target is not a
single perfect value, but a defensible range where multiple diagnostics are
reasonably stable.

## Caveats

- Threshold selection is a modeling judgment, not a theorem.  Different
  diagnostics can disagree.
- The Hill estimator is a right-tail regular-variation tool for $\xi>0$, not a
  general estimator for Gumbel-type or bounded-tail domains.
- The ratio $\hat\beta(u)/u$ is an exact-Pareto diagnostic.  For a general GPD
  threshold model, threshold stability is
  $\beta(u')=\beta(u)+\xi(u'-u)$, so $\beta(u)-\xi u$ is the stable quantity.
- Very high thresholds reduce bias but can leave too few exceedances for
  stable estimation.
- Dependence, volatility clustering, rounding, reporting limits, and
  truncation can all make threshold diagnostics misleading.
- A selected threshold for one sample window or horizon should not be reused
  mechanically for a different dataset.
- Thresholds for right-tail positive data do not automatically apply to
  two-sided returns or losses without an explicit transformation.

## References

- Hill, "A Simple General Approach to Inference About the Tail of a
  Distribution" [@hill1975simple].
- Davison and Smith, "Models for Exceedances over High Thresholds"
  [@davison1990models].
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [@coles2001introduction].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].

## Backlinks

- Depends on: [Extreme-Value Index Estimation](extreme-value-index.md),
  [Hill Estimator](hill-estimator.md),
  [Mean Excess Function](../theorems/mean-excess-function.md), and
  [Generalized Pareto Distribution](../distributions/generalized-pareto.md).
- Related: [Body, Shoulders, and Tails](body-shoulder-tail.md) is a density
  geometry diagnostic, not a threshold-selection rule.
- Used by: [S&P 500 Tail Diagnostics](../examples/sp500-tail.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/methods/tail-threshold-selection.md`. Last verified: 2026-06-24. Checked against cited sources, diagnostic workflow, and executable threshold-stability plots.
:::
<!-- incerto-provenance:end -->
