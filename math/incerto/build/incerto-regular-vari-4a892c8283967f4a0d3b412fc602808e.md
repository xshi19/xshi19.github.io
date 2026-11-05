---
title: Regular Variation
options:
  concept:
    id: regular-variation
    type: theorem
    prerequisites: [pareto]
    tags:
      - fat-tails
      - asymptotics
---

## Overview

An exact [Pareto tail](./incerto-pareto.md) has the same probability
ratio whenever we multiply a threshold by a fixed amount. Regular variation
asks for this behavior only in the limit as the threshold grows. It allows
slow corrections to a power law while retaining a limiting tail exponent.

Read the Pareto survival example first for a concrete example. This page uses
limits and positive measurable functions; the proof below establishes the
power-times-slowly-varying characterization directly from the definition.

(regular-variation-definition)=
## Definition and characterization

A positive measurable function $f$ is regularly varying at infinity with index
$\rho$, written $f\in RV_\rho$, if, for every fixed $t>0$,

$$
\lim_{x\to\infty}\frac{f(tx)}{f(x)}=t^\rho.
$$

The case $\rho=0$ is called slow variation: a positive measurable function
$L$ is slowly varying when $L(tx)/L(x)\to1$ for every fixed $t>0$.  Equivalently,
regularly varying functions can be written as

$$
f(x)=x^\rho L(x)
$$

with $L$ slowly varying.  A survival function
$\bar F$ has a regularly varying right tail with exponent $\alpha>0$, written
$\bar F\in RV_{-\alpha}$, when

$$
\lim_{x\to\infty}\frac{\bar F(tx)}{\bar F(x)}=t^{-\alpha},
\qquad t>0.
$$

Here $\bar F(x)=\mathbb P(X>x)$, $L$ is the slowly varying factor,
$\alpha$ is the positive tail exponent, and $t$ is a fixed multiplier.

## What the ratio means

Regular variation is the mathematical version of
[Pareto-style](./incerto-pareto.md) scale invariance.  At high
thresholds, multiplying the threshold by $t$ has an asymptotically stable
effect on exceedance probabilities.  The slowly varying factor $L$ allows
departures from an exact [Pareto law](./incerto-pareto.md) while
preserving the same tail exponent.

The ratio statement is stronger than saying that large observations are more
frequent than under a Gaussian baseline.  It says that the relative penalty for
raising a large threshold settles to a power $t^{-\alpha}$.

## Examples and non-examples

- Constant functions are slowly varying.  If $L(x)=c>0$, then
  $L(tx)/L(x)=1$ for every $x$ and $t>0$.
- Logarithmic corrections are slowly varying.  For fixed
  $\beta\in\mathbb R$, $L(x)=(\log x)^\beta$ on $x>1$ satisfies
  $L(tx)/L(x)\to1$.
- A nonzero power is not slowly varying.  If $L(x)=x^\beta$, then
  $L(tx)/L(x)=t^\beta$, which equals $1$ for all $t$ only when $\beta=0$.
- Lognormal right tails are heavy-tailed and subexponential, but not regularly
  varying: their fixed-multiplier survival ratios do not settle to
  $t^{-\alpha}$ for any finite $\alpha$.
- Exponential tails are neither regularly varying nor subexponential.  For a multiplier $t>1$, their
  survival ratios decay exponentially in $x$.

## Proof of the characterization

We prove the algebraic characterization used throughout these notes: a regularly
varying function is a power times a slowly varying function.  The proof is only
an unwinding of the ratio definition. The [Karamata note](./incerto-karamata.md)
develops the related integral asymptotics and their moment consequences.

If $f(x)=x^\rho L(x)$ with $L$ slowly varying, then for fixed $t>0$,

$$
\frac{f(tx)}{f(x)}
=
t^\rho\frac{L(tx)}{L(x)}
\to t^\rho.
$$

Conversely, if $f\in RV_\rho$, define $L(x)=x^{-\rho}f(x)$.  Then

$$
\frac{L(tx)}{L(x)}
=
t^{-\rho}\frac{f(tx)}{f(x)}
\to 1,
$$

so $L$ is slowly varying and $f(x)=x^\rho L(x)$.

For survival tails, set $\rho=-\alpha$ in the same equivalence:
$\bar F(x)=x^{-\alpha}L(x)$, with $L(x)=x^\alpha\bar F(x)$ slowly varying.
No separate survival-tail argument is needed.

For the [Pareto distribution](./incerto-pareto.md) with lower cutoff
$x_m$,

$$
\bar F(x)=x_m^\alpha x^{-\alpha},
$$

so $L(x)=x_m^\alpha$ is constant and therefore slowly varying.

(regular-variation-representation)=
## What the representation theorem adds

The elementary equivalence above defines $L$; it does not prove Karamata's
representation theorem. For positive measurable slowly varying $L$, that
theorem gives, eventually,

$$
L(x)=c(x)\exp\left\{\int_a^x\frac{\varepsilon(v)}v\,dv\right\},
\qquad c(x)\to c\in(0,\infty),\quad \varepsilon(x)\to0,
$$

with measurable $c$ and locally integrable $\varepsilon(v)/v$.
We cite the existence of this representation: @bingham1987regular,
Theorem 1.3.1, p. 12. This is a different result from the
[integral theorem](incerto-karamata.md).

Its sufficiency follows directly. For fixed $t>0$, the logarithm of
$L(tx)/L(x)$ is $\log[c(tx)/c(x)]+\int_x^{tx}\varepsilon(v)\,dv/v$.
The integral's absolute value is at most $|\log t|$ times the supremum
of $|\varepsilon|$ between $x$ and $tx$, which tends to zero.
Also $\log L(x)/\log x\to0$: split the integral at a fixed point beyond
which $|\varepsilon|<\delta$, divide by $\log x$, and then let
$\delta\downarrow0$. Hence a regularly varying survival satisfies
$\log\bar F(x)/\log x\to-\alpha$. This is a ratio of logarithms;
convergence of a derivative of a log-log plot needs extra smoothness.

(regular-variation-powers)=
## Positive powers of a tail variable

We prove the transformation rule used in SCoFT v4 §5.2.3, p. 100
[@taleb2025scoftv4], without differentiating a density. Let $X\ge0$ have
$\bar F(x)=x^{-\alpha}L(x)$ eventually and let $r>0$. Then

$$
P(X^r>y)=\bar F(y^{1/r})=y^{-\alpha/r}L(y^{1/r}).
$$

For fixed $t>0$, the ratio of the last slowly varying factors has arguments
$t^{1/r}y^{1/r}$ and $y^{1/r}$, so it tends to one. Thus $X^r$ has
survival exponent $\alpha/r$. Units may be fixed first by applying this
argument to $X/x_m$. Negative powers concern the behavior near zero and
are outside this right-tail rule. The exact Pareto payoff case is already
derived in [asymmetric payoffs](incerto-asymmetric-payoffs.md#payoffs-tail-transform).

## Ratio diagnostics

The fixed-multiplier ratio can be compared algebraically for an exact Pareto
tail and a power law with a slowly varying logarithmic correction.

(regular-variation-ratio-diagnostic)=
For an exact Pareto tail and a logarithmically corrected tail, respectively,

$$
\frac{\bar F(2x)}{\bar F(x)}=2^{-\alpha},
\qquad
\frac{(2x)^{-\alpha}(\log(2x))^\beta}
     {x^{-\alpha}(\log x)^\beta}
=2^{-\alpha}\left(1+\frac{\log 2}{\log x}\right)^\beta.
$$

For $\alpha=1.5$ and $\beta=1$, the correction multiplier is $2$ at $x=2$,
$3/2$ at $x=4$, and $5/4$ at $x=16$. The second expression is used only
sufficiently far into the tail, where it is decreasing and at most one
before taking the ratio; a full distribution also needs a body below that range.

The exact Pareto ratio is constant at $2^{-\alpha}$; the log-corrected
ratio approaches the same target slowly. A finite diagnostic curve can
suggest regular variation, but the definition is about the asymptotic limit.

```{figure} ../assets/figures/incerto/glossary-slow-variation.svg
:name: regular-variation-slow-ratio
:alt: The doubling ratio of a logarithmically corrected power tail decreases slowly toward the constant one-quarter ratio of an exact Pareto tail on a logarithmic threshold axis.

This reused chapter-2 figure uses $\bar F(u)=u^{-2}(1+\log u)$ for
$u\ge1$, extended by one below $1$. It has exponent $2$ and slowly
varying factor $1+\log u$, differing from the preceding exponent-$1.5$
example with factor $\log u$. Its doubling ratio is
$\tfrac14[1+\log 2/(1+\log u)]$, which approaches $1/4$.
All values are population evaluations; the threshold axis is logarithmic.
```

Use [`regular_variation_ratio`](incerto-api-reference.md#api-regular-variation-ratio)
to evaluate the law used in the figure:

```python
import numpy as np
from xmath.incerto.tail_diagnostics import regular_variation_ratio

survival = lambda u: u**-2 * (1 + np.log(u))
result = regular_variation_ratio(survival, [1, 10, 100, 10000])
for u, ratio in zip(result["threshold"], result["ratio"]):
    print(f"u={u:.4g}: ratio={ratio:.4g}, limit=0.25")
```

**Output**

```text
u=1: ratio=0.4233, limit=0.25
u=10: ratio=0.3025, limit=0.25
u=100: ratio=0.2809, limit=0.25
u=1e+04: ratio=0.267, limit=0.25
```

The existing producer is `uv run python demos/incerto/glossary_worked_cases.py`.

The [chapter 5 demo](incerto-scoft-ch5.md#scoft-ch5-demo) compares
multiplier ratios for Student and shifted double-Pareto tails. Their finite
threshold differences are explained by the
[matching calculation](incerto-double-pareto.md#double-pareto-student-matching).

## Caveats

- Regular variation is an asymptotic property.  It does not say that every
  moderate observation follows a power law.
- Estimating $\alpha$ from finite samples is threshold-sensitive; the
  [Hill estimator](./incerto-hill-estimator.md) and log-log plots are
  diagnostics, not certificates.
- A nonnegative law with a regularly varying right tail and $\alpha>0$
  is subexponential (Foss, Korshunov, and Zachary, 2nd ed., §3.5,
  Theorem 3.29 and following paragraph, p. 57); see
  [Subexponentiality](./incerto-subexponentiality.md) for the one-big-jump
  principle.
- Moment implications require assumptions on the full tail and should point to
  [Karamata's theorem](./incerto-karamata.md) or to a Pareto-specific proof.

## References

- Foss, Korshunov, and Zachary, *An Introduction to Heavy-Tailed and
  Subexponential Distributions*, @foss2013heavy,
  §3.5, Theorem 3.29, p. 57: cited subexponential classification.
- @bingham1987regular, *Regular Variation*, Theorem 1.3.1, p. 12:
  the cited representation theorem; the elementary power-times-$L$
  characterization is proved on this page.
- @taleb2025scoftv4, §5.1.1, pp. 95–97, and §5.2.3, p. 100:
  scaling and transformation context.
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  @embrechts1997modelling.
- Resnick, *Heavy-Tail Phenomena* @resnick2007heavy.

## Backlinks

- Used by: [Pareto Distribution](./incerto-pareto.md),
  [Double Pareto Distribution](./incerto-double-pareto.md),
  [Subexponentiality](./incerto-subexponentiality.md),
  [Tail Class Catalog](./incerto-tail-class-catalog.md),
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md),
  [Generalized Central Limit Theorem](./incerto-generalized-central-limit-theorem.md),
  and [Hill Estimator](./incerto-hill-estimator.md).
