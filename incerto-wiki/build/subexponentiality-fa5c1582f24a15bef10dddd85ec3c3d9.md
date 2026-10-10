---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: subexponentiality
    type: theorem
    depends_on:
      - regular-variation
    tags:
      - fat-tails
      - asymptotics
      - one-big-jump
---

# Subexponentiality

## Statement

Let $X_1,X_2,\dots$ be iid nonnegative random variables with distribution $F$
and survival function $\bar F(x)=\mathbb P(X>x)$.  The distribution is
subexponential when, for two independent copies,

$$
\frac{\mathbb P(X_1+X_2>x)}{2\mathbb P(X>x)} \to 1,
\qquad x\to\infty.
$$

Equivalently, for each fixed integer $n\ge1$, with
$S_n=X_1+\cdots+X_n$ and $M_n=\max_{1\le i\le n}X_i$,

$$
\mathbb P(S_n>x)
\sim
\mathbb P(M_n>x)
\sim n\bar F(x).
$$

Equivalently,

$$
\mathbb P(S_n>x,\; M_n\le x)=o(\bar F(x)).
$$

This is the one-big-jump principle: for a fixed number of summands, a very
large sum is usually caused by one summand being very large, not by all
summands moving moderately together.

The maximum-tail asymptotic
$\mathbb P(M_n>x)\sim n\bar F(x)$ is not the distinctive part of the theorem:
it follows for every iid distribution with an unbounded right tail and fixed
$n$.  The subexponential content is that the sum tail has the same first-order
asymptotic as the maximum tail.

[Regularly varying](regular-variation.md) survival tails with positive
exponent, $\bar F\in RV_{-\alpha}$ for $\alpha>0$, are a standard
subexponential class.  We cite that theorem and use the exact
[Pareto family](../distributions/pareto.md) as the wiki's main checkable
example.

The symbols $X_i$, $F$, $\bar F$, $n$, $\alpha$, and $x_m$ follow the shared
[notation table](../../notation/index.md).

## One-big-jump intuition

Subexponential does not mean "lighter than exponential," and it is unrelated
to the concentration-theory phrase "sub-exponential random variable" for a
light-tailed $\psi_1$ condition.  In this context it means the opposite: the
tail is heavy enough that convolution barely changes the leading-order tail
probability.

**If one big observation already explains the rare event, adding one more
independent observation roughly doubles the chance of seeing it.**

This is the mathematical version of a recurring Incerto warning.  In thin-tail
settings, a large aggregate deviation is often the result of many small
coordinated deviations.  In subexponential settings, the aggregate tail is
usually dominated by a single large term.  That changes how sums, ruin events,
insurance losses, and sample moments should be read.

## What we can prove directly

The statement collects several equivalent forms and one important sufficient
class.  We prove the elementary maximum-tail asymptotic and an exact Pareto
two-summand check.  The full equivalence theory and the theorem that regularly
varying tails are subexponential are cited, because their proofs need more
convolution-tail machinery than we develop here.

The theorem that regularly varying tails are subexponential goes back to
Chistyakov's convolution-tail criterion for sums of independent positive random
variables [@chistyakov1964sums].  Standard modern references with proofs are
Embrechts, Klueppelberg, and Mikosch, Appendix A3, and Foss, Korshunov, and
Zachary, Chapter 3 [@embrechts1997modelling; @foss2013heavy].  We do
not reprove the full theorem, but the exact [Pareto](../distributions/pareto.md)
case gives the right calibration.

### Universal maximum tail

For any distribution with unbounded right tail, $\bar F(x)\to0$ as
$x\to\infty$.  For fixed $n$, iid independence gives

$$
\mathbb P(M_n\le x)
=\mathbb P(X_1\le x,\dots,X_n\le x)
=F(x)^n
=\left(1-\bar F(x)\right)^n.
$$

Therefore

$$
\mathbb P(M_n>x)
=1-\left(1-\bar F(x)\right)^n
$$

Set $u=\bar F(x)$.  Since $u\to0$ and $n$ is fixed, the binomial expansion
gives

$$
1-(1-u)^n
=nu-\binom{n}{2}u^2+O(u^3),
$$

and therefore

$$
\frac{1-(1-u)^n}{nu}
=1-\frac{n-1}{2}u+O(u^2)
\to1.
$$

Substituting back,

$$
\mathbb P(M_n>x)
\sim n\bar F(x).
$$

This maximum formula only uses iid independence, fixed $n$, and
$\bar F(x)\to0$.  For nonnegative variables, $M_n>x$ implies
$S_n=X_1+\cdots+X_n>x$, so it is a lower bound for the sum tail.  What is
special about subexponential tails is the matching upper bound:

$$
\mathbb P(S_n>x,\;M_n\le x)=o(\bar F(x)).
$$

Those are the configurations where several observations are moderately large
but none exceeds $x$.  Subexponentiality says this residual event is negligible
at the scale of one tail probability.

### Pareto upper-bound check

For the Pareto tail $\bar F(x)=(x_m/x)^\alpha$ with $x\ge x_m$, the missing
upper-bound step can be checked directly for $n=2$.  Let

$$
A_x=\{X_1+X_2>x,\;M_2\le x\}.
$$

Choose $h=x^\gamma$ with $1/2<\gamma<1$.  The figure uses $\gamma=0.6$ and
shows the covering argument.  The residual event $A_x$ is the triangle inside
the square $[0,x]^2$ above the line $X_1+X_2=x$.  The cover region is larger
than $A_x$, but it is made of the two thin strips and the middle rectangle
$\{X_1>h,\;X_2>h\}$.

```{code-cell} python
:label: subexponentiality-cover-diagram
:tags: [hide-input]

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

from incerto.figures import COLORS, FIGURE_SIZES, set_theme

set_theme()

x = 10.0
gamma = 0.6
h = x**gamma
x_minus_h = x - h

fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])

cover_color = COLORS["accent_light"]
event_color = COLORS["green"]

# The union-bound cover:
# {x-h < X_1 <= x}, {x-h < X_2 <= x}, and {X_1 > h, X_2 > h}.
ax.add_patch(
    Rectangle(
        (0, x_minus_h),
        x,
        h,
        facecolor=cover_color,
        edgecolor="none",
        alpha=0.16,
    )
)
ax.add_patch(
    Rectangle(
        (x_minus_h, 0),
        h,
        x,
        facecolor=cover_color,
        edgecolor="none",
        alpha=0.16,
    )
)
ax.add_patch(
    Rectangle(
        (h, h),
        x - h,
        x - h,
        facecolor=cover_color,
        edgecolor="none",
        alpha=0.13,
    )
)

# The residual event A_x inside the square.
ax.add_patch(
    Polygon(
        [(0, x), (x, x), (x, 0)],
        closed=True,
        facecolor=event_color,
        edgecolor=event_color,
        linewidth=0.9,
        alpha=0.24,
    )
)

ax.add_patch(
    Rectangle(
        (0, 0),
        x,
        x,
        facecolor="none",
        edgecolor=COLORS["ink"],
        linewidth=0.9,
    )
)
ax.plot([0, x], [x, 0], color=COLORS["ink"], linewidth=1.1)

for mark in (h, x_minus_h):
    ax.axvline(mark, color=COLORS["accent"], linestyle=(0, (4, 4)), linewidth=0.8)
    ax.axhline(mark, color=COLORS["accent"], linestyle=(0, (4, 4)), linewidth=0.8)

ax.set_xlim(-0.4, x + 0.9)
ax.set_ylim(-0.4, x + 0.9)
ax.set_aspect("equal")
ax.set_xlabel(r"$X_2$")
ax.set_ylabel(r"$X_1$")
ax.set_xticks([0, h, x_minus_h, x])
ax.set_xticklabels(["0", r"$h$", r"$x-h$", r"$x$"])
ax.set_yticks([0, h, x_minus_h, x])
ax.set_yticklabels(["0", r"$h$", r"$x-h$", r"$x$"])
ax.set_title(r"A covering of $A_x=\{X_1+X_2>x,\ M_2\leq x\}$")
ax.grid(False)
ax.legend(
    handles=[
        Patch(facecolor=event_color, edgecolor=event_color, alpha=0.24, label=r"$A_x$"),
        Patch(
            facecolor=cover_color,
            edgecolor="none",
            alpha=0.16,
            label=r"cover: strips plus $\{X_1>h,\ X_2>h\}$",
        ),
        Line2D([0], [0], color=COLORS["ink"], linewidth=1.1, label=r"$X_1+X_2=x$"),
        Line2D(
            [0],
            [0],
            color=COLORS["accent"],
            linestyle=(0, (4, 4)),
            linewidth=0.8,
            label=r"$h=x^{0.6}$ and $x-h$",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=2,
)
plt.show()
```

Outside the cover region, a point has neither coordinate in $(x-h,x]$ and not
both coordinates above $h$.  Thus at least one coordinate is $\le h$, and the
other is $\le x-h$, so $X_1+X_2\le x$.  Therefore any point in $A_x$ must lie
in the cover: either one summand lies in $(x-h,x]$, or both summands
exceed $h$.  Hence

$$
\mathbb P(A_x)
\le
2\mathbb P(x-h<X\le x)
+\mathbb P(X_1>h,\;X_2>h).
$$

For the first term,

$$
\frac{\mathbb P(x-h<X\le x)}{\bar F(x)}
=
\frac{\bar F(x-h)-\bar F(x)}{\bar F(x)}
=
\left(\frac{x}{x-h}\right)^\alpha-1
\to0,
$$

because $h/x\to0$.  For the second term, independence gives

$$
\frac{\mathbb P(X_1>h,\;X_2>h)}{\bar F(x)}
=
\frac{\bar F(h)^2}{\bar F(x)}
=
x_m^\alpha x^{\alpha(1-2\gamma)}
\to0,
$$

because $\gamma>1/2$.  Thus $\mathbb P(A_x)=o(\bar F(x))$.  Since
$\{S_2>x\}$ is the disjoint union of $\{M_2>x\}$ and $A_x$,

$$
\mathbb P(X_1+X_2>x)\sim 2\bar F(x).
$$

The regularly varying theorem cited above generalizes this Pareto calculation:
the maximum formula is universal, while the negligible residual event is the
heavy-tail property.

A useful comparison table is:

| Distribution | Subexponential? | Regularly varying? |
| --- | --- | --- |
| Pareto | Yes | Yes |
| Lognormal | Yes | No |
| Weibull with shape $0<\beta<1$ | Yes | No |
| Exponential | No | No |

For a broader catalog with generated ratio diagnostics, see
[Tail Class Catalog](../distributions/tail-class-catalog.md).

The exponential is an exact nonexample.  If
$X_i\sim\operatorname{Exp}(\lambda)$, then the universal maximum formula still
holds:

$$
\mathbb P(M_2>x)=2e^{-\lambda x}-e^{-2\lambda x}
\sim2e^{-\lambda x}.
$$

But the sum tail is much larger:

$$
\mathbb P(X_1+X_2>x)=e^{-\lambda x}(1+\lambda x),
$$

so

$$
\frac{\mathbb P(X_1+X_2>x)}{2\mathbb P(X>x)}
=\frac{1+\lambda x}{2}\to\infty,
$$

not $1$.

The fixed-$n$ assumption is load-bearing.  If $n$ grows with $x$, additional
large-deviation regimes can appear.

## Simulation check

The simulation below compares the empirical tail of $X_1+X_2$ with
$2\mathbb P(X>x)$ for an exact [Pareto](../distributions/pareto.md) sample.
The second panel tracks how much of an extreme two-term sum is carried by its
larger component.

```{code-cell} python
:label: subexponentiality-one-big-jump-check
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

from incerto.distributions import pareto_type1
from incerto.figures import FIGURE_SIZES, set_theme, style_axes

set_theme()

alpha = 1.5
sample_size = 450_000
rng = np.random.default_rng(20260617)
x1 = pareto_type1.rvs(alpha, size=sample_size, random_state=rng)
x2 = pareto_type1.rvs(alpha, size=sample_size, random_state=rng)
two_sum = x1 + x2
dominant_share = np.maximum(x1, x2) / two_sum

thresholds = pareto_type1.ppf(np.linspace(0.85, 0.995, 28), alpha)
single_tail = pareto_type1.sf(thresholds, alpha)
sum_tail = np.array([np.mean(two_sum > threshold) for threshold in thresholds])
ratio = sum_tail / (2 * single_tail)
share_given_exceedance = np.array(
    [np.mean(dominant_share[two_sum > threshold]) for threshold in thresholds]
)

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"])

axes[0].plot(thresholds, ratio, marker="o")
axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.1)
axes[0].set_xscale("log")
axes[0].set_xlabel("threshold x")
axes[0].set_ylabel(r"$P(X_1+X_2>x)/(2P(X>x))$")
axes[0].set_title("One-big-jump tail ratio")

axes[1].plot(thresholds, share_given_exceedance, marker="o")
axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.1)
axes[1].set_xscale("log")
axes[1].set_ylim(0.5, 1.02)
axes[1].set_xlabel("threshold x")
axes[1].set_ylabel("mean larger-component share")
axes[1].set_title("Dominance inside large sums")

style_axes(axes, grid_axis="both")
plt.show()
```

**What to notice.** The ratio moves toward the subexponential target of one,
though the far tail is noisy because exceedances are rare.  Conditional on a
large two-term sum, the larger component tends to carry most of the aggregate.
The simulation illustrates the theorem; it does not prove subexponentiality for
an empirical dataset.

## Caveats

- The definition above is for nonnegative iid summands.  Two-sided or
  dependent data need extra assumptions before the one-big-jump reading is
  valid.
- The equivalence for $n$ summands holds for fixed $n$.  It is not a uniform
  statement over arbitrary growing horizons.
- Subexponentiality is a tail property.  It does not say the body of the
  distribution is Pareto, nor does it choose an empirical threshold.
- A simulation ratio near one is only a diagnostic.  A durable mathematical
  claim needs a theorem, a checked distributional assumption, or a citation.

## References

- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Bingham, Goldie, and Teugels, *Regular Variation* [@bingham1987regular].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].
- Taleb, *Statistical Consequences of Fat Tails* [@taleb2020scoft].

## Backlinks

- Depends on: [Regular Variation](regular-variation.md).
- Related catalog: [Tail Class Catalog](../distributions/tail-class-catalog.md).
- Related: [Max-to-Sum Ratio](../methods/max-to-sum-ratio.md).
- Related to: [LLN Failure Under Infinite Mean](lln-failure.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/theorems/subexponentiality.md`. Last verified: 2026-06-25. Checked against cited sources, scoped maximum-tail and Pareto partial proofs, and executable one-big-jump diagnostics.
:::
<!-- incerto-provenance:end -->
