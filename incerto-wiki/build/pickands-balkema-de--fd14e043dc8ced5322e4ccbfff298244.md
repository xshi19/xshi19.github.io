---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: pickands-balkema-de-haan
    type: theorem
    depends_on:
      - regular-variation
      - extreme-value-index
      - generalized-pareto
      - generalized-extreme-value
    tags:
      - extreme-value-theory
      - peaks-over-threshold
---

# Pickands-Balkema-de Haan Theorem

## Statement

Let $F$ be a distribution function with finite or infinite right endpoint
$x_F=\sup\{x:F(x)<1\}$.  For a high threshold $u<x_F$, define the conditional
excess distribution

$$
F_u(y)=\mathbb P(X-u\le y\mid X>u)
=\frac{F(u+y)-F(u)}{1-F(u)},
\qquad 0\le y<x_F-u.
$$

If $F$ is in the maximum domain of attraction of an extreme-value law with
extreme-value index $\xi$, then there is a positive scale function $\beta(u)$
such that

$$
\sup_{0\le y<x_F-u}
\left|F_u(y)-G_{\xi,\beta(u)}(y)\right|
\to 0,
\qquad u\uparrow x_F.
$$

The comparison is over the valid excess support.  The generalized Pareto term
has scale $\beta>0$ and is

$$
G_{\xi,\beta}(y)
=
1-\left(1+\frac{\xi y}{\beta}\right)^{-1/\xi},
\qquad \xi\ne0,
$$

on the support where $1+\xi y/\beta>0$, and

$$
G_{0,\beta}(y)=1-e^{-y/\beta}
$$

for $\xi=0$.  The theorem is the asymptotic justification for the
peaks-over-threshold model: sufficiently high threshold exceedances are modeled
by a [generalized Pareto distribution](../distributions/generalized-pareto.md).

The symbols $F$, $X$, $\xi$, $\beta$, $u$, and $x_F$ follow the shared
[notation table](../../notation/index.md).

## Threshold intuition

Block-maxima theory says that properly normalized maxima have only a small
number of possible limiting shapes, collected in the
[Generalized Extreme-Value Distribution](../distributions/generalized-extreme-value.md).
The Pickands-Balkema-de Haan theorem says the matching threshold view has the
same discipline: once we condition on being far enough into the tail, the
remaining excess has an approximately
[Generalized Pareto Distribution](../distributions/generalized-pareto.md).

For fat-tail work, the case $\xi>0$ is the main bridge.  It corresponds to a
[regularly varying](regular-variation.md) right tail with exponent
$\alpha=1/\xi$.  The theorem does not say every high observation is generated
by a clean [Pareto law](../distributions/pareto.md).  It says the excess
distribution over moving high thresholds has a universal limit shape under the
same domain-of-attraction assumptions used in extreme-value theory.

## Exact Pareto excess special case

The full Pickands-Balkema-de Haan theorem is cited here, not proved.  The
calculation below proves the exact Pareto special case: once the threshold is
above the lower cutoff, the excess distribution is already generalized Pareto,
not merely asymptotically close to it.

The full theorem is a classical result of Balkema and de Haan and,
independently, Pickands.  Its proof uses the equivalence between convergence of
normalized maxima and convergence of normalized threshold excesses.  We
record the statement and use exact Pareto algebra as a checkable special
case.

If $X$ is Pareto Type I with lower cutoff $x_m$ and tail exponent $\alpha$,
then for $u\ge x_m$,

$$
\mathbb P(X-u>y\mid X>u)
=
\frac{\mathbb P(X>u+y)}{\mathbb P(X>u)}
=
\left(1+\frac{y}{u}\right)^{-\alpha}.
$$

This is exactly a generalized Pareto survival function with

$$
\xi=\frac1\alpha,
\qquad
\beta(u)=\frac{u}{\alpha}.
$$

Thus exact Pareto tails do not merely approach the generalized Pareto form;
they have it at every threshold above $x_m$.

## Simulation and fit

The simulation below samples an exact Pareto tail.  The left panel overlays
scaled excess survival curves above several thresholds.  The right panel fits
a generalized Pareto distribution to exceedances and compares the fitted
parameters with the exact Pareto target.

```{code-cell} python
:label: pbdh-threshold-excess-fit
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

from incerto.distributions import pareto_type1
from incerto.figures import FIGURE_SIZES, set_theme, style_axes

set_theme()

alpha = 1.7
xi_true = 1 / alpha
rng = np.random.default_rng(20260617)
sample = pareto_type1.rvs(alpha, size=140_000, random_state=rng)

survival_quantiles = [0.90, 0.95, 0.975, 0.99]
z_grid = np.geomspace(0.02, 10, 160)

fit_quantiles = np.linspace(0.85, 0.99, 15)
xi_hat = []
scale_ratio = []

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"])

for q in survival_quantiles:
    u = np.quantile(sample, q)
    scaled_excess = (sample[sample > u] - u) / u
    empirical_sf = np.array([np.mean(scaled_excess > z) for z in z_grid])
    axes[0].loglog(z_grid, empirical_sf, label=fr"$q={q:.3f}$")

axes[0].loglog(
    z_grid,
    (1 + z_grid) ** (-alpha),
    color="black",
    linestyle="--",
    linewidth=1.1,
    label="exact Pareto excess",
)
axes[0].set_xlabel(r"scaled excess $y/u$")
axes[0].set_ylabel(r"$P((X-u)/u>y/u\mid X>u)$")
axes[0].set_title("Threshold-excess survival")
axes[0].legend()

for q in fit_quantiles:
    u = np.quantile(sample, q)
    excesses = sample[sample > u] - u
    xi, loc, beta = genpareto.fit(excesses, floc=0)
    xi_hat.append(xi)
    scale_ratio.append(beta / u)

axes[1].plot(fit_quantiles, xi_hat, marker="o", label=r"$\hat \xi$")
axes[1].plot(fit_quantiles, scale_ratio, marker="o", label=r"$\hat\beta/u$")
axes[1].axhline(xi_true, color="black", linestyle="--", linewidth=1.1)
axes[1].set_xlabel("threshold quantile")
axes[1].set_ylabel("estimate")
axes[1].set_title("GPD fit stability")
axes[1].legend()

style_axes(axes, grid_axis="both")
plt.show()
```

**What to notice.** For an exact Pareto law, scaled excess curves at different
thresholds collapse toward the same survival shape.  The fitted
$\widehat\xi$ and $\widehat\beta/u$ fluctuate around $1/\alpha$.  Real data
rarely behaves this cleanly, which is why threshold diagnostics matter.

## Caveats

- The theorem is asymptotic in the threshold.  It does not choose the threshold
  for a finite dataset.
- Dependence, seasonality, volatility clustering, rounding, truncation, and
  mixtures can all distort threshold exceedances.
- A generalized Pareto fit is a tail model, not a proof that the full
  distribution is Pareto.
- When $\xi\ge1$, the fitted generalized Pareto model has no finite mean.  When
  $\xi\ge1/2$, it has no finite variance.  These moment boundaries are often
  the practical reason the shape estimate matters.
- A high threshold reduces bias but increases variance because fewer
  exceedances remain.

## References

- Balkema and de Haan, "Residual Life Time at Great Age"
  [@balkema1974residual].
- Pickands, "Statistical Inference Using Extreme Order Statistics"
  [@pickands1975statistical].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [@coles2001introduction].

## Backlinks

- Depends on: [Regular Variation](regular-variation.md),
  [Extreme-Value Index](../methods/extreme-value-index.md),
  [Generalized Pareto Distribution](../distributions/generalized-pareto.md),
  [Generalized Extreme-Value Distribution](../distributions/generalized-extreme-value.md),
  and the threshold notation in [Notation](../../notation/index.md).
- Used by: [Mean Excess Function](mean-excess-function.md) and
  [S&P 500 Tail Diagnostics](../examples/sp500-tail.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/theorems/pickands-balkema-de-haan.md`. Last verified: 2026-06-25. Checked against cited sources, scoped exact Pareto excess derivation, and executable GPD threshold diagnostics.
:::
<!-- incerto-provenance:end -->
