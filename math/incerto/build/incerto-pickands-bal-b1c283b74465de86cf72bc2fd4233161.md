---
title: Pickands-Balkema-de Haan Theorem
options:
  concept:
    id: pickands-balkema-de-haan
    type: theorem
    depends_on:
      - regular-variation
      - extreme-value-index
      - generalized-pareto
      - generalized-extreme-value
    tags:
      - extreme-value-theory
      - peaks-over-threshold
---

## Statement

Let $F$ be a distribution function with finite or infinite right endpoint
$x_F=\sup\{x:F(x)<1\}$.  For a high threshold $u<x_F$, define the conditional
excess distribution

$$
F_u(y)=\mathbb P(X-u\le y\mid X>u)
=\frac{F(u+y)-F(u)}{1-F(u)},
\qquad 0\le y<x_F-u.
$$

If $F$ is in the maximum domain of attraction of an extreme-value law with
extreme-value index $\xi$, then there is a positive scale function $\beta(u)$
such that

$$
\sup_{0\le y<x_F-u}
\left|F_u(y)-G_{\xi,\beta(u)}(y)\right|
\to 0,
\qquad u\uparrow x_F.
$$

The comparison is over the excess support, using the full GPD CDF: for
$\xi<0$, set $G_{\xi,\beta}(y)=1$ at and beyond $-\beta/\xi$.  The generalized Pareto term
has scale $\beta>0$ and is

$$
G_{\xi,\beta}(y)
=
1-\left(1+\frac{\xi y}{\beta}\right)^{-1/\xi},
\qquad \xi\ne0,
$$

on the support where $1+\xi y/\beta>0$, and

$$
G_{0,\beta}(y)=1-e^{-y/\beta}
$$

for $\xi=0$.  The theorem is the asymptotic justification for the
peaks-over-threshold model: sufficiently high threshold exceedances are modeled
by a [generalized Pareto distribution](./incerto-generalized-pareto.md).

$X$ denotes a draw from $F$, $u$ the threshold, $x_F$ its right endpoint,
$\xi$ the extreme-value index, and $\beta(u)>0$ the excess scale.
The maximum-domain-of-attraction assumption means that for iid draws,
some $a_n>0$ and $b_n$ give $(M_n-b_n)/a_n\Rightarrow H_{\xi,0,1}$.
Here $M_n$ is the sample maximum and $\Rightarrow$ is convergence in
distribution. The GEV page defines $H$. A shared notation page is planned.

## Threshold intuition

Block-maxima theory says that properly normalized maxima have only a small
number of possible limiting shapes, collected in the
[Generalized Extreme-Value Distribution](./incerto-generalized-extreme-value.md).
The Pickands-Balkema-de Haan theorem says the matching threshold view has the
same discipline: once we condition on being far enough into the tail, the
remaining excess has an approximately
[Generalized Pareto Distribution](./incerto-generalized-pareto.md).

For fat-tail work, the case $\xi>0$ is the main bridge.  It corresponds to a
[regularly varying](./incerto-regular-variation.md) right tail with exponent
$\alpha=1/\xi$.  The theorem does not say every high observation is generated
by a clean [Pareto law](./incerto-pareto.md).  It says the excess
distribution over moving high thresholds has a universal limit shape under the
same domain-of-attraction assumptions used in extreme-value theory.

## Exact Pareto excess special case

The full Pickands-Balkema-de Haan theorem is cited here, not proved.  The
calculation below proves the exact Pareto special case: once the threshold is
above the lower cutoff, the excess distribution is already generalized Pareto,
not merely asymptotically close to it.

The full theorem is a classical result of Balkema and de Haan and,
independently, Pickands.  Its proof uses the equivalence between convergence of
normalized maxima and convergence of normalized threshold excesses.  We
record the statement and use exact Pareto algebra as a checkable special
case.

If $X$ is Pareto Type I with lower cutoff $x_m$ and tail exponent $\alpha$,
then for $u\ge x_m$,

$$
\mathbb P(X-u>y\mid X>u)
=
\frac{\mathbb P(X>u+y)}{\mathbb P(X>u)}
=
\left(1+\frac{y}{u}\right)^{-\alpha}.
$$

This is exactly a generalized Pareto survival function with

$$
\xi=\frac1\alpha,
\qquad
\beta(u)=\frac{u}{\alpha}.
$$

Thus exact Pareto tails do not merely approach the generalized Pareto form;
they have it at every threshold above $x_m$.

## Exact threshold calibration

(pbdh-threshold-excess-fit)=
For $\alpha=1.7=17/10$, the population excess parameters are
$\xi=10/17$ and $\beta(u)=10u/17$. The scaled excess
$Z=(X-u)/u$, conditional on $X>u$, has survival $(1+z)^{-1.7}$.

| Threshold $u\ge x_m$ | Shape $\xi$ | Scale $\beta(u)$ | $\mathbb P(Z>1\mid X>u)$ |
| --- | --- | --- | --- |
| $x_m$ | $10/17$ | $10x_m/17$ | $2^{-1.7}$ |
| $2x_m$ | $10/17$ | $20x_m/17$ | $2^{-1.7}$ |
| $4x_m$ | $10/17$ | $40x_m/17$ | $2^{-1.7}$ |

These are exact population identities, not fitted estimates. In finite data,
estimated shape and scale fluctuate, and real tails may approach the GPD
family slowly. [Threshold selection](./incerto-tail-threshold-selection.md)
examines that modeling choice; the theorem supplies no finite-sample cutoff.

## Caveats

- The theorem is asymptotic in the threshold.  It does not choose the threshold
  for a finite dataset.
- Dependence, seasonality, volatility clustering, rounding, truncation, and
  mixtures can all distort threshold exceedances.
- A generalized Pareto fit is a tail model, not a proof that the full
  distribution is Pareto.
- When $\xi\ge1$, the fitted generalized Pareto model has no finite mean.  When
  $\xi\ge1/2$, it has no finite variance.  These moment boundaries are often
  the practical reason the shape estimate matters.
- Raising the threshold can reduce tail-approximation bias but leaves fewer
  exceedances; it does not guarantee a monotone improvement in a fitted model.

## References

- Balkema and de Haan, "Residual Life Time at Great Age"
  [1974](https://doi.org/10.1214/aop/1176996548).
- Pickands, "Statistical Inference Using Extreme Order Statistics"
  [1975](https://doi.org/10.1214/aos/1176343003).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [2001](https://doi.org/10.1007/978-1-4471-3675-0).

## Backlinks

- Depends on: [Regular Variation](./incerto-regular-variation.md),
  [Extreme-Value Index](./incerto-extreme-value-index.md),
  [Generalized Pareto Distribution](./incerto-generalized-pareto.md),
  [Generalized Extreme-Value Distribution](./incerto-generalized-extreme-value.md),
  and a shared Notation page (planned).
- Used by: [Mean Excess Function](./incerto-mean-excess-function.md) and
  S&P 500 Tail Diagnostics (planned).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/theorems/pickands-balkema-de-haan.md`, revision `9717c9c`
(2026-09-13 Batch 2 import). Copyright (c) 2023 xshi19. Licensed under MIT.
Links, notation, and qualifications were adapted for this site; executable
figures and simulations were replaced with static calculations. No upstream
execution or formal-proof verification is claimed for this adaptation.

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
