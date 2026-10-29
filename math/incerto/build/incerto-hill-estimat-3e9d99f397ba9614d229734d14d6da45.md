---
title: Hill Estimator
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

## Overview

How heavy is the upper tail, and how sensitive is the answer to where we
start fitting it? Hill estimates the exponent of a positive
[Pareto-type tail](./incerto-pareto.md) from the largest observations.
Choosing how many observations to retain is part of the estimate.

Read the Pareto and [regular variation](./incerto-regular-variation.md)
pages first. The formula uses logarithms and sorted observations; the
[extreme-value index](./incerto-extreme-value-index.md) $\xi$ is the reciprocal of the
Pareto exponent $\alpha$. For data analysis, inspect several thresholds and check dependence and
measurement limits before interpreting a stable-looking curve.
[Tail threshold selection](./incerto-tail-threshold-selection.md) develops these checks.

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

For a [Pareto-type](./incerto-pareto.md) right tail with exponent
$\alpha$, the corresponding tail exponent estimate is

$$
\widehat\alpha_{k,n}=\frac{1}{\widehat\xi_{k,n}}.
$$

The tuning parameter $k$ is the number of upper order statistics used.  A Hill
stability plot graphs $\widehat\xi_{k,n}$ or $\widehat\alpha_{k,n}$ over a
range of $k$ values.  On this page, we name the plotted coordinate explicitly:
$\xi$ is the extreme-value index, while $\alpha=1/\xi$ is the Pareto tail
exponent.

Here $x_{i:n}$ is the $i$th ascending observation, and $k$ selects the top
$k$ values above the order-statistic threshold. If all selected values equal
that threshold, $\widehat\xi=0$ and there is no finite reciprocal estimate.
Ties can also make the strict exceedance count smaller than $k$.

## When to use it

Hill is a right-tail estimator for strictly positive observations in a
Pareto-type regime, meaning the survival tail is expected to be
[regularly varying](./incerto-regular-variation.md).  It is appropriate only
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

Define $Y=\log(X/x_m)$. For $y\ge0$,

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
@resnick2007heavy. Hill's original estimator is
described in @hill1975simple,
pp. 1163–1174.

## Stability diagnostics

A stability diagnostic repeats the same calculation over $k$ and reports both
the threshold and the chosen tail coordinate. The arithmetic is visible even
in a fixed four-observation example.

(hill-estimator-stability-diagnostic)=
Take the fixed ordered observations $(1,2,4,8)$. The Hill calculation gives:

| $k$ | Threshold $x_{n-k:n}$ | $\widehat\xi_{k,n}$ | $\widehat\alpha_{k,n}$ |
| --- | --- | --- | --- |
| $1$ | $4$ | $\log 2$ | $1/\log 2$ |
| $2$ | $2$ | $3\log 2/2$ | $2/(3\log 2)$ |
| $3$ | $1$ | $2\log 2$ | $1/(2\log 2)$ |

This small deterministic example shows how the selected threshold changes
the estimate. It is an illustration of the arithmetic, with no claim that the
four values are a random Pareto sample.

In a distribution with a bounded body and a Pareto tail, increasing $k$ far
enough pulls body observations into the calculation. For empirical data,
a plateau is evidence to inspect, not a certificate.

(hill-synthetic-stability)=
## Seeded stability comparison

```{figure} ../assets/figures/incerto/hill-stability.svg
:alt: Hill extreme-value-index estimates versus retained count for exact Pareto and a body-tail mixture, with known truth and a third panel mapping counts to observed thresholds.

Two iid samples of $30{,}000$ observations: exact Pareto with $x_m=1$,
$\alpha=1.7$ (PCG64 seed 20261020), and the
[uniform-body/Pareto-tail model](incerto-tail-threshold-selection.md#tail-threshold-selection-diagnostics)
(seed 20261021). The mixture is uniform on $[1,3]$ with probability
$0.82$ and Pareto with minimum $3$, exponent $1.7$, otherwise.
Both tail targets are $\xi=1/1.7$. The marked count of 5,344 is realized,
not its population expectation. The third panel maps $k$ to
$x_{n-k:n}$; these continuous draws have no ties, so strict exceedance
counts equal $k$. Increasing $k$ eventually includes the mixture body.
A stable-looking segment supplies no unique threshold choice.
```

Use [`pareto_type1`](incerto-api-reference.md#api-pareto-type1) and
[`hill_stability`](incerto-api-reference.md#api-hill-stability) to reproduce
selected points of the first panel:

```python
import numpy as np
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.estimators import hill_stability

x = pareto_type1.rvs(1.7, size=30000, random_state=np.random.default_rng(20261020))
hill = hill_stability(x, [50, 500, 5000])
for k, xi in zip(hill["k"], hill["xi"]):
    print(f"k={int(k):d}: xi={xi:.4g}, truth={1/1.7:.4g}")
```

**Output**

```text
k=50: xi=0.627, truth=0.5882
k=500: xi=0.5728, truth=0.5882
k=5000: xi=0.58, truth=0.5882
```

Regenerate all panels with `uv run python demos/incerto/tail_estimation.py`.
The same mixture sample supplies the coordinated threshold diagnostics.

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
  Distribution" @hill1975simple.
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  @embrechts1997modelling.
- Resnick, *Heavy-Tail Phenomena* @resnick2007heavy.

## Backlinks

- Depends on: [Pareto Distribution](./incerto-pareto.md),
  [Regular Variation](./incerto-regular-variation.md), and the shared tail notation in [Shared notation](https://xshi19.github.io/math/notation/).
- Used by: [Extreme Value Index Estimation](./incerto-extreme-value-index.md),
  [Tail Threshold Selection](./incerto-tail-threshold-selection.md), and
  [S&P 500 Tail Diagnostics](./incerto-sp500-tail.md).
