---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: generalized-extreme-value
    type: distribution
    depends_on:
      - extreme-value-index
    tags:
      - extreme-value-theory
      - block-maxima
      - tail-risk
---

# Generalized Extreme-Value Distribution

## Statement

The generalized extreme-value distribution is the standard limit family for
normalized block maxima.  With location $\mu$, scale $\sigma>0$, shape
$\xi$, and standardized variable $z=(x-\mu)/\sigma$, its CDF is

$$
H_{\xi,\mu,\sigma}(x)
=
\exp\left[-\left(1+\xi z\right)^{-1/\xi}\right],
\qquad \xi\ne0,
$$

on the support where $1+\xi z>0$.  The boundary case is the Gumbel law,

$$
H_{0,\mu,\sigma}(x)
=
\exp\left[-e^{-z}\right].
$$

The shape parameter $\xi$ is the same extreme-value index used in threshold
excesses: $\xi>0$ is [Frechet-type](frechet.md) heavy-tail behavior,
$\xi=0$ is Gumbel-type exponential boundary behavior, and $\xi<0$ is
Weibull-type finite endpoint.
The symbols $X_i$, $M_n$, $\alpha$, $\xi$, and $x_m$ follow the shared
[notation table](../../notation/index.md); $\mu$, $\sigma$, and $z$ are local
to this distributional parameterization.

## Maxima intuition

The distribution is to block maxima what the [Generalized Pareto Distribution](generalized-pareto.md)
is to threshold exceedances.  If we split data into blocks, take the maximum in
each block, and normalize those maxima appropriately, non-degenerate limits
fall into this three-shape family.

For fat-tail work, $\xi>0$ is the main case.  A [Pareto tail](pareto.md) with exponent
$\alpha$ has $\xi=1/\alpha$, so the fitted block-maximum shape should point to
the same tail coordinate as a Hill or GPD threshold analysis when the modeling
assumptions are reasonable.

## Shape regimes

The three shape regimes can be seen directly from the CDF family.  The plotted
location and scale are fixed at $\mu=0$ and $\sigma=1$ so that only the shape
parameter changes.

```{code-cell} python
:label: generalized-extreme-value-cdf-plot
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme

from incerto.figures import FIGURE_SIZES, set_theme, style_axes

set_theme()
x_grid = np.linspace(-3, 6, 500)

fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
for xi in (-0.3, 0.0, 0.3):
    support = x_grid[1 + xi * x_grid > 0] if xi != 0 else x_grid
    ax.plot(
        support,
        genextreme.cdf(support, c=-xi),
        label=fr"$\xi={xi}$",
    )

ax.set_xlabel("x")
ax.set_ylabel("CDF H(x)")
ax.set_title("GEV shape regimes")
ax.legend()
style_axes(ax)
plt.show()
```

**What to notice.** The $\xi<0$ curve has a finite right endpoint, the
$\xi=0$ curve is the Gumbel boundary, and $\xi>0$ leaves a heavy right tail.

## Pareto block maxima example

If $X_1,\dots,X_n$ are iid [Pareto Type I](pareto.md) with lower cutoff $x_m$ and exponent
$\alpha$, then for $x\ge x_m$,

$$
\mathbb P(M_n\le x)
=
\left(1-\left(\frac{x_m}{x}\right)^\alpha\right)^n.
$$

With the normalization $a_n=x_m n^{1/\alpha}$,

$$
\mathbb P(M_n/a_n\le y)
=
\left(1-\frac{1}{ny^\alpha}\right)^n
\to
\exp(-y^{-\alpha}),\qquad y>0.
$$

This is the [Frechet member](frechet.md) of the GEV family with
$\xi=1/\alpha$ after a standard location-scale reparameterization.

## Numerical fit diagnostic

SciPy's `genextreme` uses the opposite sign convention for the shape parameter:
`c = -xi`.  The code below fits block maxima from a Pareto sample and reports
the converted estimate.

```{code-cell} python
:tags: [hide-input]
:label: generalized-extreme-value-python-check

import numpy as np
from scipy.stats import genextreme

from incerto.distributions import pareto_type1

alpha = 1.8
xi_true = 1 / alpha
rng = np.random.default_rng(20260606)

blocks = pareto_type1.rvs(alpha, size=(2_000, 250), random_state=rng)
maxima = blocks.max(axis=1)

c_hat, loc_hat, scale_hat = genextreme.fit(maxima)
xi_hat = -c_hat

print(f"True xi from Pareto exponent: {xi_true:.3f}")
print(f"GEV block-maxima fit: xi_hat={xi_hat:.3f}, scale={scale_hat:.3f}")
```

## Caveats

- Block maxima discard within-block information.  Threshold methods may use
  extremes more efficiently, but they require a threshold choice.
- Blocks should be chosen with dependence and seasonality in mind.  Overlapping
  or strongly dependent blocks can make fitted uncertainty too optimistic.
- A GEV fit describes maxima, not the entire parent distribution.
- SciPy's sign convention differs from the $\xi$ convention used in this wiki:
  `genextreme` shape `c` equals `-xi`.

## References

- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [@coles2001introduction].
- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [@feller1971introduction].

## Backlinks

- Depends on: [Extreme-Value Index Estimation](../methods/extreme-value-index.md).
- Used by: [Frechet Distribution and Frechet-Type Limits](frechet.md),
  [Pickands-Balkema-de Haan Theorem](../theorems/pickands-balkema-de-haan.md)
  as the block-maxima counterpart to threshold excess limits, and by future
  return-level pages.

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/distributions/generalized-extreme-value.md`. Last verified: 2026-07-04. Checked against cited sources, page support or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
