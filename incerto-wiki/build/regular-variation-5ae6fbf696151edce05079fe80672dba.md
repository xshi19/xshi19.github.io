---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: regular-variation
    type: theorem
    prerequisites: [pareto]
    tags:
      - fat-tails
      - asymptotics
---

# Regular Variation

## Overview

An exact [Pareto tail](../distributions/pareto.md) has the same probability
ratio whenever we multiply a threshold by a fixed amount. Regular variation
asks for this behavior only in the limit as the threshold grows. It allows
slow corrections to a power law while retaining a limiting tail exponent.

Read the Pareto survival plot first for a concrete example. This page uses
limits and positive measurable functions; the proof below establishes the
power-times-slowly-varying characterization directly from the definition.

(regular-variation-definition)=
## Definition and characterization

A positive measurable function $f$ is regularly varying at infinity with index
$\rho$, written $f\in RV_\rho$, if, for every fixed $t>0$,

$$
\lim_{x\to\infty}\frac{f(tx)}{f(x)}=t^\rho.
$$

The case $\rho=0$ is called slow variation: a positive measurable function
$L$ is slowly varying when $L(tx)/L(x)\to1$ for every fixed $t>0$.  Equivalently,
regularly varying functions can be written as

$$
f(x)=x^\rho L(x)
$$

with $L$ slowly varying.  A [survival function](../../notation/index.md)
$\bar F$ has a regularly varying right tail with exponent $\alpha>0$, written
$\bar F\in RV_{-\alpha}$, when

$$
\lim_{x\to\infty}\frac{\bar F(tx)}{\bar F(x)}=t^{-\alpha},
\qquad t>0.
$$

The symbols $L$, $\bar F$, $\alpha$, $x_m$, and $RV_{-\alpha}$ follow the
shared [notation table](../../notation/index.md).  The multiplier $t$ is local
to this page.

## What the ratio means

Regular variation is the mathematical version of
[Pareto-style](../distributions/pareto.md) scale invariance.  At high
thresholds, multiplying the threshold by $t$ has an asymptotically stable
effect on exceedance probabilities.  The slowly varying factor $L$ allows
departures from an exact [Pareto law](../distributions/pareto.md) while
preserving the same tail exponent.

The ratio statement is stronger than saying that large observations are more
frequent than under a Gaussian baseline.  It says that the relative penalty for
raising a large threshold settles to a power $t^{-\alpha}$.

## Examples and non-examples

- Constant functions are slowly varying.  If $L(x)=c>0$, then
  $L(tx)/L(x)=1$ for every $x$ and $t>0$.
- Logarithmic corrections are slowly varying.  For fixed
  $\beta\in\mathbb R$, $L(x)=(\log x)^\beta$ on $x>1$ satisfies
  $L(tx)/L(x)\to1$.
- A nonzero power is not slowly varying.  If $L(x)=x^\beta$, then
  $L(tx)/L(x)=t^\beta$, which equals $1$ for all $t$ only when $\beta=0$.
- Lognormal right tails are heavy-tailed and subexponential, but not regularly
  varying: their fixed-multiplier survival ratios do not settle to
  $t^{-\alpha}$ for any finite $\alpha$.
- Exponential tails are neither regularly varying nor subexponential.  For a multiplier $t>1$, their
  survival ratios decay exponentially in $x$.

## Proof of the characterization

We prove the algebraic characterization used throughout the wiki: a regularly
varying function is a power times a slowly varying function.  The proof is only
an unwinding of the ratio definition.  Deeper representation theorems for
slowly varying functions are cited through [Karamata's theorem](karamata.md).

If $f(x)=x^\rho L(x)$ with $L$ slowly varying, then for fixed $t>0$,

$$
\frac{f(tx)}{f(x)}
=
t^\rho\frac{L(tx)}{L(x)}
\to t^\rho.
$$

Conversely, if $f\in RV_\rho$, define $L(x)=x^{-\rho}f(x)$.  Then

$$
\frac{L(tx)}{L(x)}
=
t^{-\rho}\frac{f(tx)}{f(x)}
\to 1,
$$

so $L$ is slowly varying and $f(x)=x^\rho L(x)$.

For survival tails, set $\rho=-\alpha$ in the same equivalence:
$\bar F(x)=x^{-\alpha}L(x)$, with $L(x)=x^\alpha\bar F(x)$ slowly varying.
No separate survival-tail argument is needed.

For the [Pareto distribution](../distributions/pareto.md) with lower cutoff
$x_m$,

$$
\bar F(x)=x_m^\alpha x^{-\alpha},
$$

so $L(x)=x_m^\alpha$ is constant and therefore slowly varying.

## Numerical ratio diagnostics

The next diagnostic plots the fixed-multiplier ratio
$\bar F(tx)/\bar F(x)$ over increasing thresholds.  It compares an exact Pareto
tail, a Pareto tail with a slowly varying logarithmic correction, and a
lognormal contrast.

```{code-cell} python
:label: regular-variation-ratio-diagnostic
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

from incerto.figures import FIGURE_SIZES, set_theme, style_axes

set_theme()

alpha = 1.5
beta = 1.0
t = 2.0
x_grid = np.geomspace(np.e, 200, 400)


def pareto_tail(x):
    return x ** (-alpha)


def log_corrected_tail(x):
    return x ** (-alpha) * np.log(x) ** beta


ratio_curves = {
    "exact Pareto": pareto_tail(t * x_grid) / pareto_tail(x_grid),
    "Pareto x log correction": log_corrected_tail(t * x_grid)
    / log_corrected_tail(x_grid),
    "lognormal contrast": lognorm.sf(t * x_grid, s=1.0)
    / lognorm.sf(x_grid, s=1.0),
}

survival_x = np.geomspace(np.e, 1e5, 400)

fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.6), constrained_layout=True)

for label, ratio in ratio_curves.items():
    axes[0].plot(x_grid, ratio, label=label)
axes[0].axhline(t ** (-alpha), color="black", linestyle="--", linewidth=1.2)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("threshold x")
axes[0].set_ylabel(r"ratio $\bar F(2x)/\bar F(x)$")
axes[0].set_title("Multiplier-ratio diagnostic")
axes[0].legend()

axes[1].loglog(survival_x, pareto_tail(survival_x), label="exact Pareto")
axes[1].loglog(
    survival_x,
    log_corrected_tail(survival_x),
    label="Pareto x log correction",
)
axes[1].set_xlabel("threshold x")
axes[1].set_ylabel(r"survival probability $\bar F(x)$")
axes[1].set_title("Log-log survival shapes")
axes[1].legend()

style_axes(axes, grid_axis="both")
plt.show()
```

The exact Pareto ratio is flat at $2^{-\alpha}$.  The
log-corrected tail approaches that same target slowly, while the lognormal
contrast keeps drifting downward.  A finite diagnostic curve can
suggest regular variation, but the theorem is about the asymptotic limit.

## Caveats

- Regular variation is an asymptotic property.  It does not say that every
  moderate observation follows a power law.
- Estimating $\alpha$ from finite samples is threshold-sensitive; the
  [Hill estimator](../methods/hill-estimator.md) and log-log plots are
  diagnostics, not certificates.
- A regularly varying right tail with $\alpha>0$ is subexponential under
  standard conditions; see [Subexponentiality](subexponentiality.md) for the
  one-big-jump principle.
- Moment implications require assumptions on the full tail and should point to
  [Karamata's theorem](karamata.md) or to a Pareto-specific proof.

## References

- Bingham, Goldie, and Teugels, *Regular Variation* [@bingham1987regular].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].

## Backlinks

- Used by: [Pareto Distribution](../distributions/pareto.md),
  [Double Pareto Distribution](../distributions/double-pareto.md),
  [Subexponentiality](subexponentiality.md),
  [Tail Class Catalog](../distributions/tail-class-catalog.md),
  [Pareto Moment Existence](pareto-moment-existence.md),
  [Generalized Central Limit Theorem](generalized-central-limit-theorem.md),
  and [Hill Estimator](../methods/hill-estimator.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/theorems/regular-variation.md`. Last verified: 2026-06-25. Checked against cited sources, scoped characterization proof, and executable ratio diagnostics.
:::
<!-- incerto-provenance:end -->
