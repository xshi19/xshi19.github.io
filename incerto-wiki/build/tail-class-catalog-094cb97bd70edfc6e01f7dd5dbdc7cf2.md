---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: tail-class-catalog
    type: distribution
    depends_on:
      - subexponentiality
      - regular-variation
    tags:
      - fat-tails
      - asymptotics
      - distributions
---

# Tail Class Catalog

## Main facts

We catalog common right-tail examples by two asymptotic properties:
[subexponentiality](../theorems/subexponentiality.md) and
[regular variation](../theorems/regular-variation.md).  It is a diagnostic
catalog, not a proof page.  Standard references for the classifications are
Bingham, Goldie, and Teugels for regular variation, and Embrechts,
Klueppelberg, and Mikosch plus Foss, Korshunov, and Zachary for
subexponential tails [@bingham1987regular; @embrechts1997modelling;
@foss2013heavy].

Let $\bar F(x)=\mathbb P(X>x)$ be the survival function.  In the table,
"regularly varying" means $\bar F\in RV_{-\alpha}$ for some $\alpha>0$, so
$\bar F(tx)/\bar F(x)\to t^{-\alpha}$ for fixed $t>0$.  The symbols $X$,
$F$, $\bar F$, $x$, and $\alpha$ follow the shared
[notation table](../../notation/index.md).

| Distribution | Subexponential? | Regularly varying? | Diagnostic tail behavior |
| --- | --- | --- | --- |
| [Pareto](pareto.md), $\bar F(x)=(x_m/x)^\alpha$ | Yes | Yes | Exact power tail. |
| Lognormal | Yes | No | Heavier than any exponential, lighter than any power. |
| Weibull, $\bar F(x)=\exp(-x^\beta)$ with $0<\beta<1$ | Yes | No | Stretched exponential tail. |
| Exponential | No | No | Memoryless light tail. |
| Gamma | No | No | Exponential tail with a polynomial factor. |

For the lognormal row, "heavier than any exponential, lighter than any power"
means the survival function satisfies

$$
e^{cx}\bar F(x)\to\infty
\quad\text{for every }c>0,
\qquad
x^p\bar F(x)\to0
\quad\text{for every }p>0.
$$

The inclusion direction to remember is:

$$
\bar F\in RV_{-\alpha},\ \alpha>0
\quad\Longrightarrow\quad
F\text{ is subexponential},
$$

for distributions supported on $[0,\infty)$, or under the standard
corresponding right-tail assumptions.  The converse fails: lognormal and
stretched-Weibull tails are subexponential without being regularly varying.

## Reusable diagnostics

The two numerical checks used below are intentionally finite-sample
diagnostics, not proofs.

- The regular-variation diagnostic evaluates $\bar F(tx)/\bar F(x)$ over
  increasing thresholds.  It only needs a survival function.
- The subexponential diagnostic computes
  $\mathbb P(X_1+X_2>x)/(2\bar F(x))$ by a one-dimensional convolution
  integral.  For the continuous SciPy distributions in this catalog, this is
  more accurate and less noisy than rare-event Monte Carlo.

```{code-cell} python
:label: tail-class-catalog-diagnostics
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma as gamma_dist, lognorm, weibull_min

from incerto.distributions import pareto_type1
from incerto.figures import FIGURE_SIZES, set_theme, style_axes
from incerto.tail_diagnostics import (
    iid_two_sum_tail,
    regular_variation_ratio,
)

set_theme()

alpha = 1.5
weibull_beta = 0.5
t = 2.0
x_grid = np.geomspace(1.0, 60.0, 180)
sum_x_grid = np.geomspace(2.01, 500.0, 180)
pareto_rv = pareto_type1(alpha)
lognormal_rv = lognorm(s=1.0, scale=1.0)
weibull_rv = weibull_min(c=weibull_beta, scale=1.0)
exponential_rv = expon(scale=1.0)
gamma_shape2_rv = gamma_dist(a=2.0, scale=1.0)

distributions = [
    ("Pareto", "Pareto", pareto_rv, pareto_rv.logsf),
    ("Lognormal", "Lognormal", lognormal_rv, lognormal_rv.logsf),
    (
        r"Weibull $\beta=0.5$",
        "Weibull beta=0.5",
        weibull_rv,
        weibull_rv.logsf,
    ),
    ("Exponential", "Exponential", exponential_rv, lambda z: -np.asarray(z)),
    (
        "Gamma shape 2",
        "Gamma shape 2",
        gamma_shape2_rv,
        lambda z: -np.asarray(z) + np.log1p(np.asarray(z)),
    ),
]
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
markers = ["o", "s", "^", "D", "v"]

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"])
summary_rows = []

for index, (plot_label, print_label, distribution, logsf) in enumerate(distributions):
    rv_result = regular_variation_ratio(distribution.sf, x_grid, multiplier=t)
    axes[0].plot(
        rv_result["threshold"],
        rv_result["ratio"],
        label=plot_label,
        linestyle=linestyles[index],
        marker=markers[index],
        markevery=30,
        linewidth=1.4,
    )

    sum_result = iid_two_sum_tail(distribution, sum_x_grid)
    if not np.all(sum_result["success"]):
        raise RuntimeError(f"Quadrature did not converge for {print_label}")
    axes[1].plot(
        sum_result["x"],
        sum_result["ratio"],
        label=plot_label,
        linestyle=linestyles[index],
        marker=markers[index],
        markevery=30,
        linewidth=1.4,
    )

    largest_threshold = sum_x_grid[-1]
    log_rv_at_largest = float(
        logsf(t * largest_threshold)
        - logsf(largest_threshold)
    )
    summary_rows.append(
        (
            print_label,
            largest_threshold,
            sum_result["ratio"][-1],
            log_rv_at_largest,
            np.max(sum_result["nfev"]),
        )
    )

axes[0].axhline(t ** (-alpha), color="black", linestyle="--", linewidth=1.0)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("threshold x")
axes[0].set_ylabel(r"$\bar F(2x)/\bar F(x)$")
axes[0].set_title("Multiplier survival ratio")
axes[0].legend()

axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("threshold x")
axes[1].set_ylabel(r"$P(X_1+X_2>x)/(2P(X>x))$")
axes[1].set_title("Two-summand tail ratio")
axes[1].legend()

style_axes(axes, grid_axis="both")
plt.show()

print("Largest-threshold deterministic diagnostic")
print(
    f"{'distribution':<18} {'x':>10} {'sum ratio':>11} "
    f"{'log10 RV':>11} {'max nfev':>9}"
)
for label, threshold, sum_ratio, log_rv_ratio, max_nfev in summary_rows:
    print(
        f"{label:<18} {threshold:10.3f} {sum_ratio:11.3f} "
        f"{log_rv_ratio / np.log(10):11.3f} {int(max_nfev):9d}"
    )
```

**What to notice.** The Pareto multiplier ratio is flat at $2^{-\alpha}$,
which is drawn as the dashed line.  The lognormal, Weibull, exponential, and
gamma multiplier ratios keep drifting downward, so they are not regularly
varying.  This first panel alone does not distinguish subexponential
lognormal or stretched-Weibull tails from exponential-type tails; combine it
with the convolution diagnostic.  In the sum-tail panel, all distributions are
evaluated on the same threshold grid.  The heavy-tailed subexponential
examples move toward the target level one, while the exponential-type examples
move away from it.

## How to read the numbers

The largest-threshold table reports the deterministic quadrature calculation at
the largest threshold in the common sum-tail grid.  The survival multiplier
ratio is reported as $\log_{10}\{\bar F(2x)/\bar F(x)\}$, computed from
`logsf`, so values below ordinary floating-point range are not displayed as
mathematical zeros.  The exact asymptotic statements are about $x\to\infty$;
we show only the direction of travel.

The two diagnostics test different ideas.  A distribution can pass the
subexponential ratio while failing the regular-variation ratio.  Lognormal and
stretched-Weibull tails are the standard examples.

## Caveats

- These numerical curves are illustrations.  They do not certify a tail class
  from data.
- The sum-tail diagnostic depends on the distribution's numerical `logpdf` and
  `logsf` implementations.  If those underflow to non-finite values, a
  distribution-specific log-tail formula is needed.
- A survival ratio near a power law over a finite range is not a proof of
  regular variation.
- The subexponential definition used here is the right-tail iid version for
  nonnegative summands.  Two-sided and dependent settings need additional
  assumptions.

## Appendix: convolution calculation

This appendix proves the elementary two-summand convolution decomposition used
by the subexponential diagnostic.  We work with independent continuous
variables, then specialize to an iid lower-bounded distribution.  Discrete,
dependent, and two-sided variants require separate assumptions.

For independent continuous random variables $X_1$ and $X_2$ with densities
$f_1$, $f_2$ and survival functions $\bar F_1$, $\bar F_2$,

$$
\mathbb P(X_1+X_2>x)
=
\int_{-\infty}^{\infty} f_1(y)\bar F_2(x-y)\,dy.
$$

To see the iid reduction, take a common density $f$, survival function
$\bar F$, and lower support endpoint $a$, with $x>2a$.  Split the event
$\{X_1+X_2>x\}$ into three cases, ignoring probability-zero boundary points:
first $X_2\le x/2$, second $X_1\le x/2$, and third both variables exceed
$x/2$.  In the first case, conditioning on $X_2=y$ with $a\le y\le x/2$
leaves the requirement $X_1>x-y$, so this part contributes
$\int_a^{x/2} f(y)\bar F(x-y)\,dy$.  The second case contributes the same
quantity by iid symmetry.  The remaining upper-right square has probability
$\bar F(x/2)^2$ by independence.

The figure shows this partition for the nonnegative-support case $a=0$.  The
display window is finite, but the shaded upper and right tails continue beyond
the plotted boundary.

```{code-cell} python
:label: tail-class-catalog-convolution-split
:tags: [hide-input]

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

from incerto.figures import COLORS, FIGURE_SIZES, set_theme

set_theme()

x = 10.0
half_x = x / 2.0
display_limit = 12.5

fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])

left_integral_color = COLORS["accent_light"]
right_integral_color = COLORS["green"]
square_color = COLORS["gold"]

# Condition on X_2=y with 0 <= y <= x/2, then require X_1 > x-y.
ax.add_patch(
    Polygon(
        [(0, x), (half_x, half_x), (half_x, display_limit), (0, display_limit)],
        closed=True,
        facecolor=left_integral_color,
        edgecolor=left_integral_color,
        linewidth=0.8,
        alpha=0.22,
    )
)

# The symmetric copy conditions on X_1=y with 0 <= y <= x/2.
ax.add_patch(
    Polygon(
        [(x, 0), (half_x, half_x), (display_limit, half_x), (display_limit, 0)],
        closed=True,
        facecolor=right_integral_color,
        edgecolor=right_integral_color,
        linewidth=0.8,
        alpha=0.20,
    )
)

# The remaining case is {X_1 > x/2, X_2 > x/2}.
ax.add_patch(
    Rectangle(
        (half_x, half_x),
        display_limit - half_x,
        display_limit - half_x,
        facecolor=square_color,
        edgecolor=square_color,
        linewidth=0.8,
        alpha=0.18,
    )
)

ax.plot([0, x], [x, 0], color=COLORS["ink"], linewidth=1.1)
ax.axvline(half_x, color=COLORS["accent"], linestyle=(0, (4, 4)), linewidth=0.8)
ax.axhline(half_x, color=COLORS["accent"], linestyle=(0, (4, 4)), linewidth=0.8)

ax.set_xlim(-0.4, display_limit)
ax.set_ylim(-0.4, display_limit)
ax.set_aspect("equal")
ax.set_xlabel(r"$X_2$")
ax.set_ylabel(r"$X_1$")
ax.set_xticks([0, half_x, x])
ax.set_xticklabels(["0", r"$x/2$", r"$x$"])
ax.set_yticks([0, half_x, x])
ax.set_yticklabels(["0", r"$x/2$", r"$x$"])
ax.set_title(r"Partition of $\{X_1+X_2>x\}$ by the $x/2$ cuts")
ax.grid(False)
ax.legend(
    handles=[
        Patch(
            facecolor=left_integral_color,
            edgecolor=left_integral_color,
            alpha=0.22,
            label=r"$\int_a^{x/2} f(y)\bar F(x-y)\,dy$",
        ),
        Patch(
            facecolor=right_integral_color,
            edgecolor=right_integral_color,
            alpha=0.20,
            label=r"symmetric copy",
        ),
        Patch(
            facecolor=square_color,
            edgecolor=square_color,
            alpha=0.18,
            label=r"$\bar F(x/2)^2$",
        ),
        Line2D([0], [0], color=COLORS["ink"], linewidth=1.1, label=r"$X_1+X_2=x$"),
        Line2D(
            [0],
            [0],
            color=COLORS["accent"],
            linestyle=(0, (4, 4)),
            linewidth=0.8,
            label=r"$x/2$ cuts",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=2,
)
plt.show()
```

For iid variables with lower support endpoint $a$, symmetry gives the more
useful half-line formula

$$
\mathbb P(X_1+X_2>x)
=
2\int_a^{x/2} f(y)\bar F(x-y)\,dy+\bar F(x/2)^2.
$$

The plotted diagnostic divides by $2\bar F(x)$.  The package helper
`iid_two_sum_tail` therefore evaluates

$$
R(x)
=
\int_a^{x/2}
\exp\{\log f(y)+\log\bar F(x-y)-\log\bar F(x)\}\,dy
+
\frac12
\exp\{2\log\bar F(x/2)-\log\bar F(x)\}.
$$

This normalization matters.  The absolute tail probability
$\mathbb P(X_1+X_2>x)$ can be tiny, but the normalized integrand stays on the
scale of the plotted ratio.  The implementation uses SciPy's vectorized
[tanh-sinh quadrature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.tanhsinh.html)
with `log=True`, which handles endpoint singularities such as the Weibull
density with shape $0.5$ and returns log-integrals before any final
exponentiation.  The output includes both `log_tail` and `log_ratio` so the
logarithmic quantities remain available even when ordinary tail probabilities
underflow.

Two closed-form checks calibrate the calculation:

$$
X\sim\operatorname{Exp}(1)
\quad\Longrightarrow\quad
R(x)=\frac{1+x}{2},
$$

and

$$
X\sim\operatorname{Gamma}(2,1)
\quad\Longrightarrow\quad
R(x)=
\frac{\operatorname{Gamma}(4,1).\operatorname{sf}(x)}
{2\,\operatorname{Gamma}(2,1).\operatorname{sf}(x)}.
$$

## References

- Bingham, Goldie, and Teugels, *Regular Variation*
  [@bingham1987regular].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Foss, Korshunov, and Zachary, *An Introduction to Heavy-Tailed and
  Subexponential Distributions* [@foss2013heavy].

## Backlinks

- Depends on: [Subexponentiality](../theorems/subexponentiality.md) and
  [Regular Variation](../theorems/regular-variation.md).
- Example distribution: [Pareto Distribution](pareto.md).
- Related diagnostic: [Max-to-Sum Ratio](../methods/max-to-sum-ratio.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/distributions/tail-class-catalog.md`. Last verified: 2026-06-25. Checked against cited sources, scoped convolution calculation, and executable tail-class diagnostics.
:::
<!-- incerto-provenance:end -->
