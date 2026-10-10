---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: pareto-moment-existence
    type: theorem
    prerequisites: [pareto]
    related: [karamata]
    tags:
      - moments
      - pareto
---

# Pareto Moment Existence

(pareto-moment-theorem)=
## Statement

Let $X$ have a [Pareto Type I distribution](../distributions/pareto.md) with
lower cutoff $x_m>0$ and tail exponent $\alpha>0$:

$$
\bar F(x)=\mathbb P(X>x)=\left(\frac{x_m}{x}\right)^\alpha,
\qquad x\ge x_m.
$$

For any moment order $p>0$,

$$
\mathbb E[X^p] < \infty
\quad\Longleftrightarrow\quad
p<\alpha.
$$

When $p<\alpha$,

$$
\mathbb E[X^p]=\frac{\alpha x_m^p}{\alpha-p}.
$$

At $p=\alpha$, the truncated moment diverges logarithmically.  For
$p>\alpha$, it diverges as a power of the upper cutoff.  In particular, the
Pareto mean exists only for $\alpha>1$, and the variance exists only for
$\alpha>2$.

The symbols $\bar F$, $\alpha$, $x_m$, $p$, and $\mathbb E$ follow the shared
[notation table](../../notation/index.md).

## Proof by direct integration

We prove the Pareto Type I moment boundary and the moment formula by direct
integration.  The same calculation also explains the logarithmic boundary and
power-divergent regimes in the truncated-moment section.  The broader
regularly varying moment test is handled by [Karamata's Theorem](karamata.md).

For $x\ge x_m$, the Pareto density is

$$
f(x)=\alpha x_m^\alpha x^{-(\alpha+1)}.
$$

For $p>0$,

$$
\mathbb E[X^p]
=
\int_{x_m}^{\infty}x^p\alpha x_m^\alpha x^{-(\alpha+1)}\,dx
=
\alpha x_m^\alpha
\int_{x_m}^{\infty}x^{p-\alpha-1}\,dx.
$$

(moment-exponent-threshold)=
The integral converges exactly when $p-\alpha-1<-1$, equivalently
$p<\alpha$.  Evaluating the convergent case gives

$$
\alpha x_m^\alpha
\cdot\frac{x_m^{p-\alpha}}{\alpha-p}
=
\frac{\alpha x_m^p}{\alpha-p}.
$$

The algebraic equivalence $p-\alpha-1<-1 \Longleftrightarrow p<\alpha$
is checked by `Incerto.moment_exponent_threshold`; see the
[Lean statement](../../formalization/lean-blueprint.md#checked-statements).
The integral criterion and evaluation above are not part of that
formalization.

This exact Pareto calculation is the simplest instance of the
[Karamata](karamata.md) moment test for regularly varying tails.

## Truncated-moment behavior

For a finite upper cutoff $b\ge x_m$, define

$$
M_p(b)=\mathbb E[X^p\mathbf 1_{\{X\le b\}}].
$$

Direct integration gives

$$
M_p(b)=
\begin{cases}
\dfrac{\alpha x_m^\alpha}{\alpha-p}
\left(x_m^{p-\alpha}-b^{p-\alpha}\right), & p<\alpha,\\[1.1em]
\alpha x_m^\alpha\log(b/x_m), & p=\alpha,\\[0.8em]
\dfrac{\alpha x_m^\alpha}{p-\alpha}
\left(b^{p-\alpha}-x_m^{p-\alpha}\right), & p>\alpha.
\end{cases}
$$

Thus the same theorem has three operational regimes: convergence to a finite
moment, logarithmic boundary growth, and power growth.

```{code-cell} python
:label: pareto-moment-existence-truncated-plot
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

from incerto.figures import FIGURE_SIZES, set_theme, style_axes

set_theme()

alpha = 1.5
x_m = 1.0
b_grid = np.geomspace(x_m, 1_000, 400)
orders = (1.0, 1.5, 2.0)


def truncated_moment_pareto(b, p, alpha, x_m=1.0):
    b = np.asarray(b, dtype=float)
    if np.isclose(p, alpha):
        return alpha * x_m**alpha * np.log(b / x_m)
    if p < alpha:
        return (
            alpha
            * x_m**alpha
            / (alpha - p)
            * (x_m ** (p - alpha) - b ** (p - alpha))
        )
    return (
        alpha
        * x_m**alpha
        / (p - alpha)
        * (b ** (p - alpha) - x_m ** (p - alpha))
    )


fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
for p_order in orders:
    ax.plot(
        b_grid,
        truncated_moment_pareto(b_grid, p_order, alpha, x_m=x_m),
        label=fr"$p={p_order}$",
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("upper cutoff b")
ax.set_ylabel(r"truncated moment $M_p(b)$")
ax.set_title(r"Pareto moment regimes for $\alpha=1.5$")
ax.legend()
style_axes(ax, grid_axis="both")
plt.show()
```

**What to notice.** For $p<\alpha$, the truncated moment levels off.  At
$p=\alpha$, it keeps growing, but only logarithmically.  For $p>\alpha$, the
upper tail contributes a visible power-law rise.

## Examples

- If $\alpha=0.8$, the mean is infinite because $1\ge\alpha$.
- If $\alpha=1.5$, the mean exists, but the variance is infinite because
  $2\ge\alpha$.
- If $\alpha=3$, the mean and variance exist, while the third raw moment is at
  the logarithmic boundary and diverges.

The theorem is exact, but sample estimates can still look misleading.  With
$\alpha$ close to a boundary, a finite run can appear calm until a new large
observation changes the empirical moment.

```{code-cell} python
:label: pareto-moment-existence-sample-check
:tags: [hide-input]

from incerto.distributions import pareto_type1

rng = np.random.default_rng(20260617)
alpha = 1.2
sample = pareto_type1.rvs(alpha, scale=x_m, size=50_000, random_state=rng)
running_mean = np.cumsum(sample) / np.arange(1, sample.size + 1)
theoretical_mean = alpha / (alpha - 1)
sample_second_moment = np.mean(sample**2)

print(f"Theoretical mean for alpha={alpha}: {theoretical_mean:.3f}")
print(f"Last five running means: {np.round(running_mean[-5:], 3)}")
print(
    "Sample second raw moment "
    f"(variance is infinite for alpha <= 2): {sample_second_moment:.3f}"
)
```

## Caveats

- Moment existence is a property of the generating distribution, not proof that
  a finite sample estimate will be accurate.
- When $\alpha$ is close to a boundary, convergence can be so slow that the
  formal moment is a poor operational summary.
- For two-sided heavy-tailed variables, check absolute moments or one-sided
  tails explicitly.  Symmetry can make a location parameter look finite while
  absolute exposure is infinite.
- Empirical Pareto fits require threshold checks.  A moment calculation using a
  fitted $\widehat\alpha$ inherits the uncertainty and bias of the tail fit.

## References

- Bingham, Goldie, and Teugels, *Regular Variation*
  [@bingham1987regular].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].

## Backlinks

- Depends on: [Pareto Distribution](../distributions/pareto.md) and
  [Karamata's Theorem](karamata.md).
- Used by: [LLN Failure Under Infinite Mean](lln-failure.md),
  [Pre-Asymptotic LLN Behavior](../examples/lln-preasymptotic.md), and
  [Max-to-Sum Ratio](../methods/max-to-sum-ratio.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/theorems/pareto-moment-existence.md`. Last verified: 2026-09-05. Checked against cited sources, scoped direct derivation, and executable truncated-moment diagnostics.
:::
<!-- incerto-provenance:end -->
