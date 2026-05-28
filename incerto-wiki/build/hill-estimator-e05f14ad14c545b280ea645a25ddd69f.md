---
kernelspec:
  name: python3
  display_name: Python 3
---

# Hill Estimator

## Statement

Let $X_1,\dots,X_n$ be positive observations, and write the ascending order
statistics as

$$
X_{1:n}\le \cdots \le X_{n:n}.
$$

For $1\le k<n$, the Hill estimator of the extreme-value index is

$$
\widehat\xi_{k,n}
=
\frac1k\sum_{j=1}^{k}
\log\left(\frac{X_{n-j+1:n}}{X_{n-k:n}}\right).
$$

For a Pareto-type right tail with exponent $\alpha$, the corresponding tail
exponent estimate is

$$
\widehat\alpha_{k,n}=\frac{1}{\widehat\xi_{k,n}}.
$$

The tuning parameter $k$ is the number of upper order statistics used.  A Hill
stability plot graphs $\widehat\alpha_{k,n}$ or $\widehat\xi_{k,n}$ over a range
of $k$ values.  The goal is not to find a magic point; it is to see whether
there is a credible threshold region where the estimate is not mostly an
artifact of the chosen cutoff.

## Intuition

The estimator uses log spacings above a high threshold.  Small $k$ means the
threshold is very high, so the estimate uses only the most extreme observations
and has high variance.  Large $k$ lowers the threshold, which adds data but can
mix non-tail observations into the calculation.  A stable region is a practical
compromise between those two failures.

This is why Hill estimates should be plotted against $k$.  A single number
without the stability plot hides the most important modeling choice.  The exact
Pareto model has the right tail everywhere.  A bounded-body model can share the
same Pareto tail above a cutoff, but the plot bends when $k$ becomes large
enough to include too much of the body.

## Proof

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
random threshold $X_{n-k:n}$.

For exact Pareto samples this explains why $\widehat\xi_{k,n}$ targets
$1/\alpha$.  For regularly varying tails, the standard consistency result uses
an intermediate sequence $k=k_n$ with

$$
k_n\to\infty,\qquad \frac{k_n}{n}\to0.
$$

Under the usual iid tail assumptions, $\widehat\xi_{k_n,n}$ converges in
probability to the extreme-value index $\xi=1/\alpha$.  The proof of that full
theorem belongs to extreme-value asymptotics; this page uses the exact Pareto
calibration and cites the general result.

## Python

The package implementation exposes the single-$k$ estimator and the stability
curve used for plotting.

```{code-cell} python
:label: hill-estimator-python-check

import numpy as np
import matplotlib.pyplot as plt

from incerto.distributions import pareto_type1
from incerto.estimators import hill_alpha_estimator, hill_stability
from incerto.figures import COLORS, FIGURE_SIZES, set_theme, style_axes

set_theme()

alpha = 1.5
n = 20_000
rng = np.random.default_rng(20260523)

pure_sample = pareto_type1.rvs(alpha, size=n, random_state=rng)

body_tail_sample = np.empty(n)
body_mask = rng.random(n) < 0.82
body_tail_sample[body_mask] = 1 + 3 * rng.random(np.sum(body_mask)) ** 0.7
body_tail_sample[~body_mask] = 4 * pareto_type1.rvs(
    alpha, size=np.sum(~body_mask), random_state=rng
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
ax.set_ylabel("estimated alpha")
ax.set_title("Hill stability plot")
ax.legend()
style_axes(ax, grid_axis="both")
plt.show()

single_k_estimate = hill_alpha_estimator(pure_sample, k=500)
print(f"Hill alpha estimate for k=500 on the exact Pareto sample: {single_k_estimate:.3f}")
```

For a pure Pareto sample, the curve should fluctuate around the true exponent.
For empirical data, the more important question is whether there is a plateau
that survives reasonable changes in $k$.

## Caveats

- Hill is a right-tail estimator for positive data.  Transform or split
  two-sided data before using it.
- The estimator is threshold-sensitive.  A reported $\widehat\alpha$ should
  include the selected $k$, the threshold, and a stability plot.
- A plateau is suggestive, not a proof of a Pareto tail.  Dependence,
  mixtures, truncation, and measurement limits can all manufacture or destroy
  apparent stability.
- Estimating $\alpha$ and plugging it into moments is dangerous near moment
  boundaries.  Small estimation error around $\alpha=1$ or $\alpha=2$ can change
  whether a mean or variance is treated as finite.
- The reciprocal $\widehat\alpha=1/\widehat\xi$ is unstable when
  $\widehat\xi$ is close to zero.  Thin-tail regimes need different tools.

## References

- Hill, "A Simple General Approach to Inference About the Tail of a
  Distribution" [@hill1975simple].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].
- Taleb, *Statistical Consequences of Fat Tails*, Chapter 3
  [@taleb2020scoft].

## Backlinks

- Depends on: [Pareto Distribution](../distributions/pareto.md),
  [Regular Variation](../theorems/regular-variation.md), and the canonical
  order statistic notation in [Notation](../../notation/index.md).
- Used by: [Chapter 3 reading guide](../../reading-guides/taleb-scoft/ch3.md)
  and future plug-in tail estimation examples.
