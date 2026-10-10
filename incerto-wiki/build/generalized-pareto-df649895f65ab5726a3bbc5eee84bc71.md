---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: generalized-pareto
    type: distribution
    depends_on:
      - extreme-value-index
    tags:
      - extreme-value-theory
      - peaks-over-threshold
      - tail-risk
---

# Generalized Pareto Distribution

## Statement

The generalized Pareto distribution is the standard limit family for threshold
excesses.  In the wiki's notation, an excess variable $Y\ge0$ has shape
$\xi$ and scale $\beta>0$ when

$$
G_{\xi,\beta}(y)
=
1-\left(1+\frac{\xi y}{\beta}\right)^{-1/\xi},
\qquad \xi\ne0,
$$

on the support where $1+\xi y/\beta>0$.  The boundary case is

$$
G_{0,\beta}(y)=1-e^{-y/\beta}.
$$

For $\xi\ge0$, the support is $y\ge0$.  For $\xi<0$, the support is
$0\le y\le -\beta/\xi$, so the tail has a finite endpoint.  The survival
function for $\xi\ne0$ is

$$
\bar G_{\xi,\beta}(y)
=
\left(1+\frac{\xi y}{\beta}\right)^{-1/\xi}.
$$

The mean exists only for $\xi<1$ and equals $\beta/(1-\xi)$.  The variance
exists only for $\xi<1/2$ and equals

$$
\frac{\beta^2}{(1-\xi)^2(1-2\xi)}.
$$

## Shape and endpoint intuition

The generalized Pareto distribution is the peaks-over-threshold counterpart of
the generalized extreme-value distribution for block maxima.  Once a threshold
is high enough, the distribution of the excess over that threshold is modeled
with one shape parameter $\xi$ and one scale parameter $\beta$.

The sign and size of $\xi$ carry the tail story.  Positive $\xi$ gives a
[Pareto-type](pareto.md) heavy tail.  Zero gives the exponential boundary.  Negative $\xi$
gives a finite endpoint.  In the fat-tail examples, the most important
boundary values are $\xi=1/2$ for variance and $\xi=1$ for the mean.

```{code-cell} python
:label: generalized-pareto-survival-plot
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

from incerto.figures import FIGURE_SIZES, set_theme, style_axes

set_theme()
y_grid = np.linspace(0, 8, 500)

fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
for xi in (-0.4, 0.0, 0.4):
    support = y_grid if xi >= 0 else y_grid[y_grid <= -1 / xi]
    ax.plot(
        support,
        genpareto.sf(support, c=xi, scale=1.0),
        label=fr"$\xi={xi}$",
    )

ax.set_xlabel("excess y")
ax.set_ylabel(r"survival $\bar G(y)$")
ax.set_title("GPD survival and finite endpoint regimes")
ax.legend()
style_axes(ax)
plt.show()
```

**What to notice.** Positive $\xi$ gives the slowest-decaying curve, $\xi=0$
is the exponential boundary, and negative $\xi$ ends at the finite endpoint
$-\beta/\xi$.

## Threshold-excess stability and Pareto special case

The Pareto Type I distribution is an exact generalized Pareto excess model.
If $X$ has Pareto tail exponent $\alpha$ and threshold $u\ge x_m$, then

$$
\mathbb P(X-u>y\mid X>u)
=
\left(1+\frac{y}{u}\right)^{-\alpha}.
$$

This is the generalized Pareto survival function with

$$
\xi=\frac1\alpha,
\qquad
\beta(u)=\frac{u}{\alpha}.
$$

The [Pickands-Balkema-de Haan theorem](../theorems/pickands-balkema-de-haan.md)
explains why the same family appears asymptotically for a much larger class of
threshold exceedances.

## Numerical checks

SciPy's `genpareto` uses the same shape sign convention as this page: the
shape argument `c` equals $\xi$.

```{code-cell} python
:tags: [hide-input]
:label: generalized-pareto-python-check

import numpy as np
from scipy.stats import genpareto

xi = 0.4
beta = 2.0
y = np.array([0.0, 1.0, 5.0, 10.0])

survival = genpareto.sf(y, c=xi, scale=beta)
manual = (1 + xi * y / beta) ** (-1 / xi)
mean, variance = genpareto.stats(c=xi, scale=beta, moments="mv")

print("GPD survival check:")
for value, observed, expected in zip(y, survival, manual):
    print(f"  y={value:4.1f}  scipy={observed:.6f}  formula={expected:.6f}")
print(f"Mean={float(mean):.3f}; variance={float(variance):.3f}")
```

## Caveats

- A GPD fit is a tail model for exceedances, not a model for the full
  distribution body.
- The threshold $u$ is not chosen by the theorem.  Shape and scale estimates
  should be inspected across a range of thresholds.
- When $\xi\ge1$, the fitted excess distribution has no finite mean.  When
  $\xi\ge1/2$, it has no finite variance.
- Dependence, truncation, censoring, seasonality, and mixture effects can
  distort threshold-excess fits.

## References

- Balkema and de Haan, "Residual Life Time at Great Age"
  [@balkema1974residual].
- Pickands, "Statistical Inference Using Extreme Order Statistics"
  [@pickands1975statistical].
- Davison and Smith, "Models for Exceedances over High Thresholds"
  [@davison1990models].
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [@coles2001introduction].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].

## Backlinks

- Depends on: [Extreme-Value Index Estimation](../methods/extreme-value-index.md)
  and the shared GPD notation in [Notation](../../notation/index.md).
- Used by: [Pickands-Balkema-de Haan Theorem](../theorems/pickands-balkema-de-haan.md),
  [Mean Excess Function](../theorems/mean-excess-function.md), and
  [Tail Threshold Selection](../methods/tail-threshold-selection.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/distributions/generalized-pareto.md`. Last verified: 2026-06-06. Checked against cited sources, page support or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
