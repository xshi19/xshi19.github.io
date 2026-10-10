---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: cramer-condition
    type: theorem
    depends_on:
      - regular-variation
    tags:
      - thin-tails
      - large-deviations
      - exponential-moments
---

# Cramér Exponential-Moment Condition

## Statement

A random variable $X$ satisfies a right-tail Cramér condition when there is
some $\theta>0$ such that

$$
\mathbb E[e^{\theta X}]<\infty.
$$

For two-sided large-deviation statements, one usually asks for the moment
generating function $M(\theta)=\mathbb E[e^{\theta X}]$ to be finite in an
open neighborhood of $0$.  Equivalently, the cumulant generating function
$\Lambda(\theta)=\log \mathbb E[e^{\theta X}]$ is finite near zero, and
the Cramér rate function is

$$
I(x)=\sup_\theta\{\theta x-\Lambda(\theta)\}.
$$

We use the one-sided version when the question is upper-tail
concentration; the classical two-sided Cramér large-deviation theorem requires
stronger mgf control.

The condition gives the elementary Chernoff bound

$$
\mathbb P(X>x)\le e^{-\theta x}\mathbb E[e^{\theta X}],
$$

so the upper-tail probability is bounded above by an exponentially decaying
function.  Equivalently, $\bar F(x)=O(e^{-\theta x})$ for that value of
$\theta$.

The symbols $X$, $x$, $\mathbb E$, and $\alpha$ follow the shared
[notation table](../../notation/index.md).  The exponential-tilting parameter
$\theta$ is local to this page.

## Thin-tail intuition

The Cramér condition is a thin-tail gate.  If exponential moments exist, then
exponential tilting, Chernoff bounds, and classical large-deviation rates have
room to operate.  If no positive exponential moment exists, those tools may
give a false sense of security.

This is the clean contrast with [Pareto-type](../distributions/pareto.md) fat
tails.  A [regularly varying](regular-variation.md) tail can have many finite
ordinary moments, but multiplying by $e^{\theta X}$ eventually overwhelms
every power-law decay.  The failure of the Cramér condition is therefore
stronger than "variance is infinite"; it can also fail when the mean and
variance are finite.

## Chernoff bound and examples

We prove the displayed Chernoff bound from Markov's inequality.  The right-tail
and two-sided Cramer conditions are definitions or standard large-deviation
hypotheses; the larger Cramer theorem is cited rather than proved here.  The
normal, exponential, and Pareto cases below are direct checks of the condition.

The Chernoff bound follows from Markov's inequality applied to the nonnegative
random variable $e^{\theta X}$:

$$
\mathbb P(X>x)
=
\mathbb P(e^{\theta X}>e^{\theta x})
\le
e^{-\theta x}\mathbb E[e^{\theta X}].
$$

A standard normal random variable satisfies the two-sided version because
$\mathbb E[e^{\theta X}]=e^{\theta^2/2}$ for all real $\theta$.  An exponential
random variable with rate $\lambda$ satisfies the right-tail version only for
$0<\theta<\lambda$.

For a [Pareto Type I](../distributions/pareto.md) random variable,

$$
\mathbb E[e^{\theta X}]
=
\int_{x_m}^{\infty} e^{\theta x}\alpha x_m^\alpha x^{-(\alpha+1)}\,dx.
$$

For every $\theta>0$, the exponential factor dominates the polynomial decay,
so the integral diverges.  Thus Pareto tails fail the right-tail Cramér
condition for every positive $\theta$.

## Truncated exponential moment check

The next plot compares finite truncations of
$\mathbb E[e^{\theta X}\mathbf 1_{\{X\le c\}}]$.  The exponential example
approaches a finite limit when $\theta$ is below its rate.  The Pareto example
keeps growing as the cap increases.

```{code-cell} python
:label: cramer-condition-truncated-moment-check
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

from incerto.distributions import pareto_type1
from incerto.figures import FIGURE_SIZES, set_theme, style_axes

set_theme()

theta = 0.12
alpha = 2.5
caps = np.geomspace(2, 140, 60)

exponential_truncated = (1.0 - np.exp(-(1.0 - theta) * caps)) / (1.0 - theta)

pareto_truncated = []
for cap in caps:
    grid = np.linspace(1.0, cap, 5_000)
    integrand = np.exp(theta * grid) * pareto_type1.pdf(grid, alpha)
    pareto_truncated.append(np.trapezoid(integrand, grid))
pareto_truncated = np.array(pareto_truncated)

x_grid = np.geomspace(1.0, 80, 300)

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"])

axes[0].plot(caps, exponential_truncated, label="exponential, rate 1")
axes[0].plot(caps, pareto_truncated, label=fr"Pareto, $\alpha={alpha}$")
axes[0].axhline(1 / (1 - theta), color="black", linestyle="--", linewidth=1.1)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("truncation cap c")
axes[0].set_ylabel(r"$E[e^{\theta X}\mathbf{1}_{\{X\leq c\}}]$")
axes[0].set_title(fr"Truncated exponential moments, $\theta={theta}$")
axes[0].legend()

axes[1].semilogy(x_grid, np.exp(-x_grid), label="exponential survival")
axes[1].semilogy(x_grid, pareto_type1.sf(x_grid, alpha), label="Pareto survival")
axes[1].set_xlabel("threshold x")
axes[1].set_ylabel(r"survival $\bar F(x)$")
axes[1].set_title("Semi-log tail contrast")
axes[1].legend()

style_axes(axes, grid_axis="both")
plt.show()
```

**What to notice.** The exponential truncated moment settles at the analytic
limit $1/(1-\theta)$.  The Pareto truncated moment keeps increasing because no
positive exponential moment exists, even though ordinary moments up to order
$p<\alpha$ are finite.

## Caveats

- The condition is direction-specific unless the moment generating function is
  finite around both sides of zero.
- Failure of the Cramér condition does not by itself prove regular variation
  or subexponentiality.  A lognormal distribution has every positive polynomial
  moment finite but no positive exponential moment, so "all moments finite" is
  not enough for the Cramér condition.
- Empirical samples cannot prove that an exponential moment exists; a finite
  sample always has a finite empirical exponential average.
- Large-deviation results need more than the one-line condition here.  This
  page records the gate and the thin-tail contrast, not the full theorem.

## References

- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [@feller1971introduction].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Taleb, *Statistical Consequences of Fat Tails* [@taleb2020scoft].

## Backlinks

- Depends on: [Regular Variation](regular-variation.md).
- Used by: [Subexponentiality](subexponentiality.md) and thin-tail contrast
  pages.

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/theorems/cramer-condition.md`. Last verified: 2026-06-25. Checked against cited sources, scoped Chernoff-bound proof, and executable truncated-moment diagnostics.
:::
<!-- incerto-provenance:end -->
