---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: body-shoulder-tail
    type: method
    depends_on:
      - normal-mixture
    tags:
      - diagnostics
      - tail-risk
---

# Body, Shoulders, and Tails

## Statement

For a centered normal density parameterized by its variance

$$
g_v(x)=
\frac{1}{\sqrt{2\pi v}}
\exp\left(-\frac{x^2}{2v}\right),
$$

the body-shoulder-tail diagnostic asks where a small, mean-preserving
randomization of $v$ adds density and where it removes density.  A local
version is the sign of

$$
\frac{\partial^2 g_v(x)}{\partial v^2}.
$$

For the standard normal density, this second derivative changes sign at

$$
\pm\sqrt{3-\sqrt6},
\qquad
\pm\sqrt{3+\sqrt6}.
$$

These four points split the line into left tail, left shoulder, peak, right
shoulder, and right tail regions.  Positive variance curvature means a local
variance mixture raises density there; negative variance curvature means it
lowers density there.

Density symbols such as $f$ and $g$ follow the shared
[notation table](../../notation/index.md).
The variance parameter $v$ is local to this diagnostic.

## Examples

- For the standard normal density under variance perturbation, the inner
  shoulder boundaries are about $\pm0.742$, and the outer tail boundaries are
  about $\pm2.334$.
- If one perturbs the normal scale $\sigma$ rather than the variance $v$, the
  corresponding boundaries are different: about $\pm0.662$ and $\pm2.136$ for
  the standard normal.
- For Student-t densities, a scale-curvature version applies, but the boundary
  values depend on the degrees of freedom.
- These boundaries are not Pareto thresholds; threshold selection for
  [regular variation](../theorems/regular-variation.md) is a separate problem.

## Diagnostic intuition

If a Gaussian variance is randomized while keeping the center fixed, the
resulting mixture does not simply "spread out" everywhere.  It tends to add mass near the
peak and in the far tails, while taking mass from the shoulders.  The shoulders
are the moderate-deviation region that looks ordinary under a single scale but
is depleted when variance uncertainty is introduced.

This is a diagnostic for finite-sample geometry, not a tail-index estimator.
It explains why stochastic volatility can create both a sharper center and
fatter-looking tails without producing a [Pareto tail](../distributions/pareto.md).
Choosing a threshold for tail estimation remains the separate judgment handled
by [Tail Threshold Selection](tail-threshold-selection.md).

## Derivation of the variance-curvature boundaries

We derive the standard-normal variance-curvature boundaries by differentiating
the density with respect to variance and solving the resulting quadratic in
$x^2$.  The Taylor expansion explains the local mixture interpretation.  The
Student-t and scale-perturbation boundaries later in the page are numerical
comparisons rather than general threshold-selection claims.

For the normal density with variance parameter $v$, write

$$
g_v(x)=
\frac{1}{\sqrt{2\pi v}}
\exp\left(-\frac{x^2}{2v}\right).
$$

Set $y=x/\sqrt v$.  A direct differentiation gives

$$
\frac{\partial g_v(x)}{\partial v}
=
\frac{g_v(x)}{2v}(y^2-1),
$$

and a second differentiation gives

$$
\frac{\partial^2 g_v(x)}{\partial v^2}
=
\frac{g_v(x)}{4v^2}(y^4-6y^2+3).
$$

At $v=1$, the sign changes where

$$
x^4-6x^2+3=0.
$$

Solving the quadratic in $x^2$ gives

$$
x^2=3\pm\sqrt6.
$$

This is a local Taylor diagnostic.  If

$$
V=v_0+\varepsilon,\qquad \mathbb E[\varepsilon]=0,\qquad
\operatorname{Var}(\varepsilon)=\tau^2,
$$

and the perturbation is small enough for the expansion to be informative, then

$$
\mathbb E[g_V(x)]-g_{v_0}(x)
=
\frac{\tau^2}{2}
\frac{\partial^2g_v(x)}{\partial v^2}\bigg|_{v=v_0}
+o(\tau^2).
$$

For a large finite mixture, the exact crossings need not equal the local
curvature boundaries.

The figure below overlays the standard normal density with a simple
[normal variance mixture](../distributions/normal-mixture.md).  It also shows
how the two-sided survival bends before any asymptotic tail model is chosen.

```{code-cell} python
:label: body-shoulder-tail-region-plot
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

from incerto.distributions import simple_normal_mixture
from incerto.figures import COLORS, FIGURE_SIZES, set_theme, style_axes

from incerto.stats import normal_variance_peak_shoulder_tail, t_peak_shoulder_tail

set_theme()

x = np.linspace(-5, 5, 600)
right_x = np.linspace(0, 5, 400)
bounds = normal_variance_peak_shoulder_tail()
student_bounds = t_peak_shoulder_tail(df=3)
survival_boundary = np.sqrt(3.0)


def density_difference(z):
    return simple_normal_mixture.pdf(z, a=0.8) - norm.pdf(z)


crossings = np.array(
    [
        brentq(density_difference, 0.05, 1.0),
        brentq(density_difference, 1.0, 3.5),
    ]
)

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"])

axes[0].plot(x, norm.pdf(x), label="standard normal")
axes[0].plot(
    x,
    simple_normal_mixture.pdf(x, a=0.8),
    label="variance mixture",
)
for boundary in bounds:
    axes[0].axvline(boundary, color=COLORS["muted"], linestyle=":", linewidth=1.0)
for crossing in crossings:
    axes[0].axvline(crossing, color=COLORS["accent"], linestyle="-.", linewidth=0.9)
    axes[0].axvline(-crossing, color=COLORS["accent"], linestyle="-.", linewidth=0.9)
axes[0].annotate("peak", xy=(0, norm.pdf(0)), xytext=(-0.35, 0.44))
axes[0].annotate("shoulder", xy=(1.2, norm.pdf(1.2)), xytext=(0.9, 0.25))
axes[0].annotate("tail", xy=(2.8, norm.pdf(2.8)), xytext=(2.7, 0.09))
axes[0].set_xlabel("x")
axes[0].set_ylabel("density")
axes[0].set_title("Body, shoulders, and tails")
axes[0].legend()

axes[1].semilogy(right_x, 2 * norm.sf(right_x), label="normal |X| survival")
axes[1].semilogy(
    right_x,
    2 * simple_normal_mixture.sf(right_x, a=0.8),
    label="mixture |X| survival",
)
axes[1].axvline(
    survival_boundary,
    color=COLORS["muted"],
    linestyle=":",
    linewidth=1.0,
)
axes[1].set_xlabel("|x| threshold")
axes[1].set_ylabel(r"$P(|X|>x)$")
axes[1].set_title("Curvature before tail modeling")
axes[1].legend()

style_axes(axes, grid_axis="both")
plt.show()

print(f"Normal variance-curvature density boundaries: {np.round(bounds, 3)}")
print(f"Exact density crossings for a=0.8: +/-{np.round(crossings, 3)}")
print(f"Normal |X| survival variance-curvature boundary: {survival_boundary:.3f}")
print(f"Student-t(df=3) scale-curvature boundaries: {np.round(student_bounds, 3)}")
```

**What to notice.** Variance mixing raises the center and far tails while
lowering the shoulders.  The dotted vertical lines are local
variance-curvature density boundaries.  The dash-dot lines mark the exact
crossings of the displayed $a=0.8$ finite mixture, which need not coincide
with the local boundaries.  The survival-panel line is the corresponding local
variance-curvature boundary for two-sided survival, not a Pareto threshold.

## Caveats

- The diagnostic assumes a symmetric normal variance perturbation.  Skewed or
  multimodal distributions need a different interpretation.
- The region labels are not universal definitions of "body" or "tail".  They
  are tied here to local variance perturbations.  Perturbing scale instead of
  variance changes the numerical boundaries.
- A shoulder/tail boundary is not a threshold for Pareto estimation.  It
  describes density geometry, not regular variation.
- The finite-difference version used in exploratory plots depends on the size
  of the variance perturbation.  The formulas here are the local limit.

## References

- Taleb, *Statistical Consequences of Fat Tails* [@taleb2020scoft].

## Backlinks

- Depends on: [Normal Variance Mixture](../distributions/normal-mixture.md)
  and the canonical density notation in [Notation](../../notation/index.md).
- Used by: [Dispersion Ratio Under Fat Tails](../examples/dispersion-ratio.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/methods/body-shoulder-tail.md`. Last verified: 2026-06-25. Checked against cited sources, scoped variance-curvature derivation, and executable region diagnostics.
:::
<!-- incerto-provenance:end -->
