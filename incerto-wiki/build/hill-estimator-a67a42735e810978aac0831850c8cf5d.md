---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: hill-estimator
    type: method
    prerequisites:
      - pareto
      - regular-variation
      - extreme-value-index
    tags:
      - tail-index
      - estimation
---

# Hill Estimator

## Overview

How heavy is the upper tail, and how sensitive is the answer to where we
start fitting it? Hill estimates the exponent of a positive
[Pareto-type tail](../distributions/pareto.md) from the largest observations.
Choosing how many observations to retain is part of the estimate.

Read the Pareto and [regular variation](../theorems/regular-variation.md)
pages first. The formula uses logarithms and sorted observations; the
[extreme-value index](extreme-value-index.md) $\xi$ is the reciprocal of the
Pareto exponent $\alpha$. For data analysis, begin with
[threshold selection](tail-threshold-selection.md) and check dependence and
measurement limits before interpreting a stable-looking curve.

(hill-estimator-definition)=
## Estimator

Let $x_1,\dots,x_n$ be observed right-tail data with $x_i>0$ for every $i$,
and write the ascending realized order statistics as

$$
x_{1:n}\le \cdots \le x_{n:n}.
$$

For $1\le k<n$, the Hill estimate of the extreme-value index is

$$
\widehat\xi_{k,n}
=
\frac1k\sum_{j=1}^{k}
\log\left(\frac{x_{n-j+1:n}}{x_{n-k:n}}\right).
$$

For a [Pareto-type](../distributions/pareto.md) right tail with exponent
$\alpha$, the corresponding tail exponent estimate is

$$
\widehat\alpha_{k,n}=\frac{1}{\widehat\xi_{k,n}}.
$$

The tuning parameter $k$ is the number of upper order statistics used.  A Hill
stability plot graphs $\widehat\xi_{k,n}$ or $\widehat\alpha_{k,n}$ over a
range of $k$ values.  On this page, we name the plotted coordinate explicitly:
$\xi$ is the extreme-value index, while $\alpha=1/\xi$ is the Pareto tail
exponent.

The symbols $x_{i:n}$, $k$, $\widehat\xi$, and $\widehat\alpha$ follow the
shared [notation table](../../notation/index.md).

## When to use it

Hill is a right-tail estimator for strictly positive observations in a
Pareto-type regime, meaning the survival tail is expected to be
[regularly varying](../theorems/regular-variation.md).  It is appropriate only
after deciding what data transformation makes the tail positive and
one-sided.

Small $k$ means the threshold is very high, so the estimate uses only the most
extreme observations and has high variance.  Large $k$ lowers the threshold,
which adds data but can mix non-tail observations into the calculation.  A
stable region is a practical compromise between those two failures.

Report a range of $k$ values, the corresponding thresholds, and the sensitivity
of the estimated exponent. A plateau can guide further checks, but it does not
establish regular variation or independence.

(hill-pareto-calibration)=
## Exact Pareto calibration

For a Pareto Type I variable,

$$
\mathbb P(X>x)=\left(\frac{x_m}{x}\right)^\alpha,\qquad x\ge x_m.
$$

Define $Y=\log(X/x_m)$.  Then

$$
\mathbb P(Y>y)
=
\mathbb P(X>x_m e^y)
=e^{-\alpha y},
$$

so $Y$ is exponential with mean $1/\alpha$.  Equivalently,

$$
\mathbb E\left[\log\left(\frac{X}{u}\right)\mid X>u\right]
=\frac1\alpha
$$

for any Pareto threshold $u\ge x_m$.  The Hill estimator replaces this
conditional expectation by the empirical average of log-excesses above the
random threshold $X_{n-k:n}$, or by $x_{n-k:n}$ after the sample is realized.

For exact Pareto samples this explains why $\widehat\xi_{k,n}$ targets
$1/\alpha$.  For a positive iid sample whose right tail is regularly varying
with extreme-value index $\xi>0$, the standard consistency result uses an
intermediate sequence $k=k_n$ with

$$
k_n\to\infty,\qquad \frac{k_n}{n}\to0.
$$

Under these assumptions, $\widehat\xi_{k_n,n}$ converges in probability to
$\xi$.  The proof of that full theorem belongs to extreme-value asymptotics;
we use the exact Pareto calibration and cite the general result in Resnick
[@resnick2007heavy]. Hill's original estimator is described in the paper
[@hill1975simple], pp. 1163–1174,
[doi:10.1214/aos/1176343247](https://doi.org/10.1214/aos/1176343247).

## Stability diagnostics

The package implementation exposes the single-$k$ estimator and the stability
curve used for plotting.  The example compares a pure Pareto sample with a
sample that has a bounded body plus the same Pareto tail.

```{code-cell} python
:label: hill-estimator-stability-diagnostic
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

from incerto.distributions import pareto_type1
from incerto.estimators import hill_alpha_estimator, hill_stability
from incerto.figures import COLORS, FIGURE_SIZES, set_theme, style_axes

set_theme()

alpha = 1.5
n = 20_000
rng = np.random.default_rng(20260617)

pure_sample = pareto_type1.rvs(alpha, size=n, random_state=rng)

body_tail_sample = np.empty(n)
body_mask = rng.random(n) < 0.82
body_tail_sample[body_mask] = 1 + 3 * rng.random(np.sum(body_mask)) ** 0.7
body_tail_sample[~body_mask] = 4 * pareto_type1.rvs(
    alpha,
    size=np.sum(~body_mask),
    random_state=rng,
)

ks = np.unique(np.geomspace(5, int(0.35 * n), 90).astype(int))
pure = hill_stability(pure_sample, ks)
body_tail = hill_stability(body_tail_sample, ks)

fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
ax.semilogx(
    pure["k"],
    pure["alpha"],
    color=COLORS["green"],
    label="exact Pareto",
)
ax.semilogx(
    body_tail["k"],
    body_tail["alpha"],
    color=COLORS["umber"],
    label="bounded body plus Pareto tail",
)
ax.axhline(alpha, color=COLORS["accent"], ls="--", lw=1.0, label="true alpha")
ax.set_xlabel("upper order statistics k")
ax.set_ylabel(r"estimated tail exponent $\widehat\alpha$")
ax.set_title("Hill stability plot")
ax.legend()
style_axes(ax, grid_axis="both")
plt.show()

single_k_estimate = hill_alpha_estimator(pure_sample, k=500)
print(f"Hill alpha estimate for k=500 on the exact Pareto sample: {single_k_estimate:.3f}")
```

The pure Pareto curve fluctuates around the true exponent.
The bounded-body sample bends when $k$ becomes large enough to pull body
observations into the tail calculation.  For empirical data, a plateau is
evidence to inspect, not a certificate.

## Failure modes

- Hill is a right-tail estimator for strictly positive data: every realized
  input must satisfy $x_i>0$.  Transform or split two-sided data before using
  it.
- The estimator is threshold-sensitive.  A reported $\widehat\alpha$ should
  include the selected $k$, the threshold, and a stability plot.
- A plateau is suggestive, not a proof of a Pareto tail.  Dependence,
  mixtures, truncation, and measurement limits can all manufacture or destroy
  apparent stability.
- Estimating $\alpha$ and plugging it into moments is dangerous near moment
  boundaries.  Small estimation error around $\alpha=1$ or $\alpha=2$ can
  change whether a mean or variance is treated as finite.
- The reciprocal $\widehat\alpha=1/\widehat\xi$ is unstable when
  $\widehat\xi$ is close to zero.  Thin-tail regimes need different tools.

## References

- Hill, "A Simple General Approach to Inference About the Tail of a
  Distribution" [@hill1975simple].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].

## Backlinks

- Depends on: [Pareto Distribution](../distributions/pareto.md),
  [Regular Variation](../theorems/regular-variation.md), and the canonical
  order statistic notation in [Notation](../../notation/index.md).
- Used by: [Extreme Value Index Estimation](extreme-value-index.md),
  [Tail Threshold Selection](tail-threshold-selection.md), and
  [S&P 500 Tail Diagnostics](../examples/sp500-tail.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/methods/hill-estimator.md`. Last verified: 2026-06-17. Checked against cited sources, exact Pareto calibration, and executable Hill stability diagnostics.
:::
<!-- incerto-provenance:end -->
