---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: frechet
    type: distribution
    depends_on:
      - generalized-extreme-value
      - pareto
      - regular-variation
    tags:
      - extreme-value-theory
      - block-maxima
      - fat-tails
---

# Frechet Distribution and Frechet-Type Limits

## Statement

The Frechet distribution is the positive-shape extreme-value law for normalized
maxima with heavy right tails.  In the standard one-parameter form with shape
$\alpha>0$, its CDF is

$$
\Phi_\alpha(y)
=
\begin{cases}
0, & y\le0,\\
\exp(-y^{-\alpha}), & y>0.
\end{cases}
$$

For $y>0$, the density is

$$
\phi_\alpha(y)
=
\alpha y^{-\alpha-1}\exp(-y^{-\alpha}).
$$

It is the $\xi>0$ member of the
[generalized extreme-value distribution](generalized-extreme-value.md).  With
$\xi=1/\alpha$, location $\mu=1$, and scale $\sigma=\xi$,

$$
H_{\xi,1,\xi}(y)
=
\exp\left[-y^{-1/\xi}\right]
=
\Phi_\alpha(y),
\qquad y>0.
$$

The phrase Frechet-type refers to distributions whose normalized maxima
converge to this law.  In the usual right-tail setting, this is the
[regularly varying](../theorems/regular-variation.md) maximum-domain-of-attraction
case: a [Pareto-type](pareto.md) survival tail with exponent $\alpha$ has
extreme-value index $\xi=1/\alpha>0$.  The symbols $F$, $\bar F$, $X_i$,
$M_n$, $\alpha$, $\xi$, and $x_m$ follow the shared
[notation table](../../notation/index.md); $\Phi_\alpha$ and $\phi_\alpha$
are local notation for this page.

## Maxima interpretation

Frechet-type behavior is a statement about extremes, not necessarily about the
full parent distribution.  If $X_1,\dots,X_n$ are iid and the right tail is
regularly varying with exponent $\alpha>0$, then an appropriate positive
normalization $a_n$ puts the maximum

$$
M_n=\max_{1\le i\le n} X_i
$$

on the Frechet scale.  The full regular-variation domain-of-attraction theorem
is cited in the references; the exact Pareto calculation below shows the core
mechanism.

For [Pareto Type I](pareto.md) with lower cutoff $x_m$ and exponent $\alpha$,
take $a_n=x_m n^{1/\alpha}$.  Then for fixed $y>0$ and all large enough $n$,

$$
\mathbb P(M_n/a_n\le y)
=
\left(1-\frac{1}{ny^\alpha}\right)^n
\to
\exp(-y^{-\alpha})
=
\Phi_\alpha(y).
$$

The same shape parameter is therefore seen in two coordinates:

$$
\alpha \quad \text{for the Pareto-type survival exponent},
\qquad
\xi=\frac1\alpha \quad \text{for the extreme-value index}.
$$

## Tail behavior

Although the Frechet law is a limit law for maxima, it is itself heavy-tailed.
Its survival function satisfies

$$
\bar\Phi_\alpha(y)
=
1-\exp(-y^{-\alpha})
\sim
y^{-\alpha},
\qquad y\to\infty.
$$

Thus the Frechet survival tail is regularly varying with index $-\alpha$.
This is why the $\xi>0$ GEV regime is also the Pareto-type regime: the
block-maximum limit and the parent survival tail share the same reciprocal
coordinate $\xi=1/\alpha$.

## Numerical check

The next check simulates exact Pareto block maxima, rescales by
$a_n=x_m n^{1/\alpha}$, and compares the empirical CDF with the Frechet limit.

```{code-cell} python
:label: frechet-pareto-maxima-check
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

from incerto.distributions import pareto_type1
from incerto.figures import COLORS, FIGURE_SIZES, set_theme, style_axes

set_theme()

alpha = 1.6
x_m = 1.0
block_size = 600
n_blocks = 2_500
rng = np.random.default_rng(20260704)

blocks = pareto_type1.rvs(
    alpha,
    scale=x_m,
    size=(n_blocks, block_size),
    random_state=rng,
)
scaled_maxima = blocks.max(axis=1) / (x_m * block_size ** (1 / alpha))

y_grid = np.geomspace(0.25, 5.0, 240)
frechet_cdf = np.exp(-(y_grid ** (-alpha)))
empirical_cdf = (
    np.searchsorted(np.sort(scaled_maxima), y_grid, side="right") / n_blocks
)

fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
ax.plot(y_grid, frechet_cdf, color=COLORS["accent"], label="Frechet limit")
ax.plot(
    y_grid,
    empirical_cdf,
    color=COLORS["green"],
    linestyle="--",
    label="scaled Pareto maxima",
)
ax.set_xscale("log")
ax.set_xlabel("scaled maximum y")
ax.set_ylabel("CDF")
ax.set_title("Pareto block maxima approach a Frechet law")
ax.legend()
style_axes(ax, grid_axis="both")
plt.show()

max_error = np.max(np.abs(empirical_cdf - frechet_cdf))
print(f"Maximum grid CDF error: {max_error:.3f}")
print(f"Extreme-value index xi = 1/alpha = {1 / alpha:.3f}")
```

**What to notice.** The scaled maxima follow the Frechet CDF because the parent
tail is exactly Pareto.  For real data or second-order tails, convergence can
be slower and threshold or block-size diagnostics still matter.

## Caveats

- Frechet-type is a domain-of-attraction label.  It does not say the original
  observations themselves follow a Frechet distribution.
- The $\xi>0$ conclusion is asymptotic.  Finite samples can look Frechet-like
  over one range and deviate elsewhere because of body contamination,
  dependence, truncation, censoring, or second-order tail behavior.
- GEV block-maxima fitting uses a location-scale family.  The unit Frechet law
  above is a convenient standard representative of the positive-shape class.

## References

- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [@coles2001introduction].

## Backlinks

- Depends on: [Generalized Extreme-Value Distribution](generalized-extreme-value.md),
  [Pareto Distribution](pareto.md), and
  [Regular Variation](../theorems/regular-variation.md).
- Used by: [Extreme Value Index Estimation](../methods/extreme-value-index.md)
  and the positive-shape regime of
  [Generalized Extreme-Value Distribution](generalized-extreme-value.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/distributions/frechet.md`. Last verified: 2026-07-04. Checked against cited sources, page proof or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
