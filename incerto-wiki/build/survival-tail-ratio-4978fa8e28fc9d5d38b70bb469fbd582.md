---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: survival-tail-ratio
    type: method
    depends_on:
      - regular-variation
      - subexponentiality
    related:
      - tail-class-catalog
      - max-to-sum-ratio
    tags:
      - diagnostics
      - fat-tails
      - tail-risk
---

# Survival Tail Ratio

## Statement

For a right-tail survival function $\bar F(x)=\mathbb P(X>x)$, the survival
tail-ratio diagnostic is

$$
Q_F(x)=\frac{\bar F(x)^2}{\bar F(2x)},
$$

where $\bar F(2x)>0$.  With iid copies $X_1$ and $X_2$, the numerator is

$$
\bar F(x)^2=\mathbb P(X_1>x,\ X_2>x).
$$

Thus $Q_F(x)$ compares a two-coordinate moderate-extreme event with a
one-coordinate double-threshold event.  It is a tail-geometry diagnostic, not
an estimator by itself.

The symbols $X$, $F$, $\bar F$, $x$, $\alpha$, and $x_m$ follow the shared
[notation table](../../notation/index.md).

## Calibration cases

The diagnostic separates three useful geometries:

| Tail model | $Q_F(x)$ behavior | Reading |
| --- | --- | --- |
| Pareto, $\bar F(x)=(x_m/x)^\alpha$ | $2^\alpha(x_m/x)^\alpha\to0$ | A single doubled observation is eventually more likely than two independent observations above $x$. |
| Exponential, $\bar F(x)=e^{-\lambda x}$ | $1$ | Two $x$-exceedances and one $2x$-exceedance have the same exponential cost. |
| Standard normal | $\sim \sqrt{2/\pi}\,e^{x^2}/x\to\infty$ | Two moderate extremes are much more likely than one doubled extreme. |

The Pareto calculation is exact once $x\ge x_m$.  The normal calculation uses
Mills' ratio for the Gaussian survival tail [@feller1971introduction].

The [regular variation](../theorems/regular-variation.md) connection is simple.
If $\bar F\in RV_{-\alpha}$ with $\alpha>0$, then

$$
\frac{\bar F(2x)}{\bar F(x)}\to 2^{-\alpha}.
$$

Therefore

$$
Q_F(x)
=
\frac{\bar F(x)}{\bar F(2x)/\bar F(x)}
\sim
2^\alpha\bar F(x)
\to0.
$$

So regularly varying tails fall on the one-big-jump side of this diagnostic.
That agrees with [subexponentiality](../theorems/subexponentiality.md), though
$Q_F(x)\to0$ is only a diagnostic ratio and not the full convolution-tail
definition.

## Numerical diagnostic

The plot below compares the ratio for a Pareto tail, Student-t tails, an
exponential tail, and a Gaussian tail.  The curves are distribution-level
checks, not fits to data.

```{code-cell} python
:label: survival-tail-ratio-diagnostic
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy, expon, norm, t as student_t

from incerto.distributions import pareto_type1
from incerto.figures import FIGURE_SIZES, set_theme, style_axes

set_theme()

x_grid = np.geomspace(1.0, 12.0, 240)


def tail_ratio(sf, x):
    survival_x = sf(x)
    survival_2x = sf(2.0 * x)
    return survival_x**2 / survival_2x


curves = {
    r"Pareto $\alpha=1.5$": tail_ratio(lambda x: pareto_type1.sf(x, 1.5), x_grid),
    r"Student-t df=3": tail_ratio(student_t(df=3).sf, x_grid),
    "Cauchy": tail_ratio(cauchy.sf, x_grid),
    "Exponential": tail_ratio(expon.sf, x_grid),
    "Normal": tail_ratio(norm.sf, x_grid),
}

fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
for label, ratio in curves.items():
    ax.plot(x_grid, ratio, label=label)

ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("threshold x")
ax.set_ylabel(r"$Q_F(x)=\bar F(x)^2/\bar F(2x)$")
ax.set_title("Survival tail-ratio diagnostic")
ax.legend()
style_axes(ax, grid_axis="both")
plt.show()

print("Diagnostic value at x=12")
for label, ratio in curves.items():
    print(f"{label:<22} Q_F(12) = {ratio[-1]:.4g}")
```

**What to notice.** The Pareto, Student-t, and Cauchy curves keep moving
toward zero.  The exponential curve is exactly flat at one.  The normal curve
explodes upward because a doubled Gaussian deviation is far more expensive
than two independent single deviations.

## Caveats

- $Q_F(x)$ is a distribution-level diagnostic.  A finite empirical estimate can
  be dominated by sampling noise because both the numerator and denominator
  involve rare events.
- The iid interpretation of $\bar F(x)^2$ fails under dependence.  Clustered
  extremes need their own dependence model before this ratio can be read as a
  joint-event probability.
- The ratio uses right tails.  Two-sided returns, losses, and absolute values
  require an explicit transformation before the diagnostic is applied.
- $Q_F(x)\to0$ is compatible with one-big-jump geometry, but it is not a
  replacement for the convolution-tail definition of
  [Subexponentiality](../theorems/subexponentiality.md).

## References

- Taleb, *Statistical Consequences of Fat Tails* [@taleb2020scoft].
- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [@feller1971introduction].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].

## Backlinks

- Depends on: [Regular Variation](../theorems/regular-variation.md) and
  [Subexponentiality](../theorems/subexponentiality.md).
- Related catalog: [Tail Class Catalog](../distributions/tail-class-catalog.md).
- Related diagnostic: [Max-to-Sum Ratio](max-to-sum-ratio.md).
- Used by: [Iso-Density Tail Geometry](iso-density-tail-geometry.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/methods/survival-tail-ratio.md`. Last verified: 2026-06-22. Checked against cited sources, page proof or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
