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
an unwinding of the ratio definition.  Deeper representation theorems for
slowly varying functions are cited through [Karamata's theorem](./incerto-karamata.md).

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

## Caveats

- Regular variation is an asymptotic property.  It does not say that every
  moderate observation follows a power law.
- Estimating $\alpha$ from finite samples is threshold-sensitive; the
  [Hill estimator](./incerto-hill-estimator.md) and log-log plots are
  diagnostics, not certificates.
- A regularly varying right tail with $\alpha>0$ is subexponential under
  standard conditions; see Subexponentiality (planned) for the
  one-big-jump principle.
- Moment implications require assumptions on the full tail and should point to
  [Karamata's theorem](./incerto-karamata.md) or to a Pareto-specific proof.

## References

- Bingham, Goldie, and Teugels, *Regular Variation* [1987](https://doi.org/10.1017/CBO9780511721434).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).

## Backlinks

- Used by: [Pareto Distribution](./incerto-pareto.md),
  Double Pareto Distribution (planned),
  Subexponentiality (planned),
  Tail Class Catalog (planned),
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md),
  Generalized Central Limit Theorem (planned),
  and [Hill Estimator](./incerto-hill-estimator.md).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/theorems/regular-variation.md`, revision `9717c9c`
(2026-09-13 import). Copyright (c) 2023 xshi19. Licensed under MIT.
Links and notation were adapted for this site; executable figures and simulations
were replaced with static calculations. No upstream execution or formal-proof
verification is claimed for this adaptation.

:::{dropdown} MIT permission notice

MIT License

Copyright (c) 2023 xshi19

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
:::
