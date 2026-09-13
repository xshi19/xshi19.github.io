---
title: Pareto Distribution
options:
  concept:
    id: pareto
    type: distribution
    prerequisites: []
    special_case_of: [regular-variation]
    applications: [pareto-moment-existence, hill-estimator]
    tags:
      - fat-tails
      - power-law
---

## Overview

How quickly does the chance of a large observation fall as we raise the
threshold? The Pareto law models a positive quantity above a lower cutoff.
Its tail exponent $\alpha$ controls that rate: smaller exponents give more
probability to very large observations. Doubling any threshold in its support
multiplies the survival probability by $2^{-\alpha}$.

Basic probability and integration are enough for this page. A survival
probability means $\mathbb P(X>x)$. The formulas below define the CDF $F$,
density $f$, and quantile $Q$; the survival table illustrates the exponent.

(pareto-definition)=
## Definition and formula summary

Let $X$ be a positive random variable with lower cutoff $x_m>0$ and tail
exponent $\alpha>0$.  The Pareto Type I distribution is defined by

$$
\bar F(x)=\mathbb P(X>x)=
\begin{cases}
1, & x < x_m,\\
\left(\frac{x_m}{x}\right)^\alpha, & x\ge x_m.
\end{cases}
$$

For $x\ge x_m$, the CDF, density, and quantile function are

$$
F(x)=1-\left(\frac{x_m}{x}\right)^\alpha,
\qquad
f(x)=\alpha x_m^\alpha x^{-(\alpha+1)},
\qquad
Q(q)=x_m(1-q)^{-1/\alpha},\quad 0<q<1.
$$

The exact Pareto tail is [regularly varying](./incerto-regular-variation.md)
with index $-\alpha$.  Its raw moment of order $p>0$ exists exactly when
$p<\alpha$:

$$
\mathbb E[X^p]=\frac{\alpha x_m^p}{\alpha-p},\qquad p<\alpha.
$$

In particular,

$$
\mathbb E[X]=\frac{\alpha x_m}{\alpha-1},\qquad \alpha>1,
$$

and

$$
\operatorname{Var}(X)
=\frac{\alpha x_m^2}{(\alpha-1)^2(\alpha-2)},\qquad \alpha>2.
$$

Thus the mean exists only for $\alpha>1$, and a finite variance exists only
for $\alpha>2$. Here $p$ is the moment order and $q$ is the quantile level.

## Shape and tail intuition

(pareto-survival-scale)=
Above its lower cutoff, the Pareto law has no characteristic upper scale.  Once
both thresholds lie in the Pareto tail, multiplying the threshold by a fixed factor changes the exceedance
probability by a fixed power:

$$
\frac{\bar F(tx)}{\bar F(x)}=t^{-\alpha},\qquad t>0,\; x\ge x_m,\; tx\ge x_m.
$$

This is why the exponent $\alpha$ is the central knob.  Smaller $\alpha$ means
that threshold doublings are punished less severely, so rare observations remain
large enough to dominate sums, moments, and empirical estimates.

The density also shows how the distribution piles mass near the lower cutoff
while leaving a long right tail. Its values and scaling make this explicit.

(pareto-density-plot)=
At the cutoff, $f(x_m)=\alpha/x_m$. For $x\ge x_m$,

$$
\frac{f(2x)}{f(x)}=2^{-(\alpha+1)}.
$$

This ratio describes the decay of the density without a numerical plot.

Larger $\alpha$ concentrates more mass near $x_m$ and makes
the density decay faster.  Smaller $\alpha$ keeps more visible mass far from the
cutoff.

## Power-law survival

Survival probabilities are usually more revealing than densities for fat-tail
work.  For an exact Pareto law,

$$
\bar F(x)=x_m^\alpha x^{-\alpha},\qquad x\ge x_m,
$$

so a log-log survival plot is a straight line with slope $-\alpha$:

$$
\log \bar F(x)=\alpha\log x_m-\alpha\log x.
$$

(pareto-survival-plot)=
For $x_m=1$, the survival values are powers of the threshold:

| Tail exponent $\alpha$ | $\bar F(1)$ | $\bar F(2)$ | $\bar F(4)$ |
| --- | --- | --- | --- |
| $0.8$ | $1$ | $2^{-0.8}$ | $2^{-1.6}$ |
| $1.5$ | $1$ | $2^{-1.5}$ | $2^{-3}$ |
| $3$ | $1$ | $2^{-3}$ | $2^{-6}$ |

These survival functions trace straight lines on log-log axes because each is
an exact power. The heaviest tail has the shallowest line.

The same scaling appears numerically. For a fixed multiplier $t>0$, the
ratio $\bar F(tx)/\bar F(x)$ depends only on $t$, provided both
$x\ge x_m$ and $tx\ge x_m$. The identity follows by canceling $(x_m/x)^\alpha$. A Lean formalization
is planned; this page supplies an ordinary algebraic argument.

(pareto-survival-ratio-check)=
For example, with $\alpha=1.16$, doubling any threshold in the support
multiplies survival by $2^{-1.16}\approx0.4475$. This is a direct evaluation
of the scaling identity, not an empirical estimate.

## Moment thresholds

For a positive moment order $p$, the exact Pareto formula is

$$
\mathbb E[X^p]=\frac{\alpha x_m^p}{\alpha-p}\quad (p<\alpha),
$$

and the moment is infinite for $p\ge\alpha$. The
[Pareto moment theorem](./incerto-pareto-moment-existence.md) gives the direct
integration proof and the truncated-moment formulas and table. Truncation
makes each truncated population moment finite: raising the upper cutoff reveals
whether it approaches a limit, grows logarithmically at $p=\alpha$, or grows
as a power for $p>\alpha$.

For a general [regularly varying tail](./incerto-regular-variation.md),
$\bar F(x)=x^{-\alpha}L(x)$, the boundary $p=\alpha$ depends on $L$.
[Karamata's theorem](./incerto-karamata.md) explains this distinction; the
exact Pareto boundary always diverges.

The moment-existence thresholds give a concrete example:

(pareto-stat-check)=
With $x_m=1$ and $\alpha=1.16$, the mean is
$1.16/(1.16-1)=7.25$ and the second raw moment is infinite; the variance
is therefore infinite.

For simulation, inverse transform sampling gives

$$
X=x_m U^{-1/\alpha},\qquad U\sim\operatorname{Uniform}(0,1).
$$

When $x_m$ is known and the observations are assumed to be exact Pareto draws,
the maximum-likelihood estimator is

$$
\widehat\alpha=\frac{n}{\sum_{i=1}^n\log(X_i/x_m)}.
$$

This estimator is not a license to fit an exact Pareto from the sample minimum:
empirical tail thresholds are rarely known and must be diagnosed separately.

A separate discussion of the two-sided double Pareto model is planned.

## Caveats

- Empirical data rarely follows an exact Pareto law from its minimum value.  A
  tail model needs a threshold choice, diagnostic plots, and sensitivity checks.
- A finite theoretical mean can still be practically hard to estimate when
  $\alpha$ is close to 1.  The issue is pre-asymptotic behavior, not merely the
  formal existence of $\mathbb E[X]$.
- The exact failure mode depends on the moment order.  At $p=\alpha$, the
  cutoff moment grows like $\log b$; for $p>\alpha$, it grows like
  $b^{p-\alpha}$.
- Do not estimate high moments of heavy-tailed samples without checking whether
  those moments are implied by the fitted tail exponent.

## References

- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).

## Backlinks

- Generalizes to: [Regular Variation](./incerto-regular-variation.md); moment consequences are covered by
  [Karamata's theorem](./incerto-karamata.md).
- Used by: Double Pareto Distribution (planned),
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md),
  Pre-Asymptotic LLN Behavior (planned),
  Max-to-Sum Ratio (planned),
  and [Hill Estimator](./incerto-hill-estimator.md).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/distributions/pareto.md`, revision `9717c9c`
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
