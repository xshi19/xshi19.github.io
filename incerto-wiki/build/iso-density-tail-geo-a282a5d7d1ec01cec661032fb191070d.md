---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: iso-density-tail-geometry
    type: method
    depends_on:
      - survival-tail-ratio
      - subexponentiality
    related:
      - body-shoulder-tail
      - max-to-sum-ratio
    tags:
      - diagnostics
      - fat-tails
      - geometry
---

# Iso-Density Tail Geometry

## Statement

Iso-density tail geometry studies the level sets of the iid joint density

$$
f_{X_1,X_2}(x_1,x_2)=f(x_1)f(x_2)
$$

near a large-deviation constraint such as $x_1+x_2=s$.  The diagnostic asks
whether the most likely configurations along the constraint sit near the
diagonal, where both coordinates are moderately large, or near the axes, where
one coordinate is large and the other is ordinary.

The symbols $X_1$, $X_2$, $f$, and $x$ follow the shared
[notation table](../../notation/index.md).  The large-sum level $s$ is local
to this page.

## Normal versus Cauchy geometry

For two iid standard normal variables, the joint density is proportional to

$$
\exp\left[-\frac{x_1^2+x_2^2}{2}\right].
$$

On the line $x_1+x_2=s$,

$$
x_1^2+x_2^2
=
2\left(x_1-\frac{s}{2}\right)^2+\frac{s^2}{2},
$$

so the joint density is maximized at the equal split
$(s/2,s/2)$.  A large Gaussian sum is geometrically a many-moderate-deviation
event.

For two iid standard Cauchy variables,

$$
f(x)=\frac{1}{\pi(1+x^2)}.
$$

Compare the equal split $(s/2,s/2)$ with an axial split $(0,s)$.  The axial to
equal density ratio is

$$
\frac{f(0)f(s)}{f(s/2)^2}
=
\frac{(1+s^2/4)^2}{1+s^2}
\sim
\frac{s^2}{16}
\to\infty.
$$

Along a large-sum line, the Cauchy geometry increasingly favors one large
coordinate rather than two equal coordinates.  This is the density-contour
version of the one-big-jump intuition in
[Subexponentiality](../theorems/subexponentiality.md) and the
[Survival Tail Ratio](survival-tail-ratio.md).

## Contour diagnostic

The figure overlays a large-sum line on standard normal and standard Cauchy
joint-density contours.  The marked diagonal point is $(s/2,s/2)$; the two
axis-near points are $(0,s)$ and $(s,0)$.

```{code-cell} python
:label: iso-density-tail-geometry-contours
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy, norm

from incerto.figures import COLORS, FIGURE_SIZES, set_theme, style_axes

set_theme()

limit = 7.0
s = 6.0
grid = np.linspace(-limit, limit, 320)
x1, x2 = np.meshgrid(grid, grid)
normal_log_joint = norm.logpdf(x1) + norm.logpdf(x2)
cauchy_log_joint = cauchy.logpdf(x1) + cauchy.logpdf(x2)

line_x = np.linspace(max(-limit, s - limit), min(limit, s + limit), 200)
line_y = s - line_x

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"])

for ax, log_joint, title in [
    (axes[0], normal_log_joint, "Normal joint density"),
    (axes[1], cauchy_log_joint, "Cauchy joint density"),
]:
    levels = np.linspace(np.max(log_joint) - 12.0, np.max(log_joint) - 1.0, 8)
    ax.contour(x1, x2, log_joint, levels=levels, colors=COLORS["accent"], linewidths=0.9)
    ax.plot(line_x, line_y, color=COLORS["brick"], linewidth=1.4, label=fr"$x_1+x_2={s:g}$")
    ax.scatter([s / 2], [s / 2], color=COLORS["green"], zorder=3, label="equal split")
    ax.scatter([0, s], [s, 0], color=COLORS["gold"], zorder=3, label="axis split")
    ax.set_aspect("equal")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    ax.legend(loc="upper right")

style_axes(axes, grid_axis="both")
plt.show()


def normal_axial_to_equal(level):
    return np.exp(-(level**2) / 4.0)


def cauchy_axial_to_equal(level):
    return (1.0 + (level / 2.0) ** 2) ** 2 / (1.0 + level**2)


print("Axial density divided by equal-split density")
for level in (4.0, 8.0, 12.0):
    print(
        f"s={level:>4.1f}  "
        f"normal={normal_axial_to_equal(level):>10.3g}  "
        f"cauchy={cauchy_axial_to_equal(level):>10.3g}"
    )
```

**What to notice.** The Gaussian contours make the diagonal point the cheapest
large-sum configuration.  The Cauchy contours increasingly favor the axis-near
points as $s$ grows.  The printed ratios show the same flip in density
geometry.

## Caveats

- Iso-density geometry requires a density.  Discrete, singular, censored, or
  heavily rounded data need a different diagnostic.
- Density at a point is not probability mass.  The contour picture explains
  local geometry; probability statements require integration over regions.
- The iid product-density formula fails under dependence.  Volatility
  clustering, common factors, and contagion can rotate or bend the contours.
- The examples here are symmetric and centered.  Skewed losses or one-sided
  positive variables should first be put into a problem-specific coordinate
  system.

## References

- Taleb, *Statistical Consequences of Fat Tails* [@taleb2020scoft].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].

## Backlinks

- Depends on: [Survival Tail Ratio](survival-tail-ratio.md) and
  [Subexponentiality](../theorems/subexponentiality.md).
- Related diagnostic: [Max-to-Sum Ratio](max-to-sum-ratio.md).
- Related density geometry: [Body, Shoulders, and Tails](body-shoulder-tail.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/methods/iso-density-tail-geometry.md`. Last verified: 2026-06-22. Checked against cited sources, page proof or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
