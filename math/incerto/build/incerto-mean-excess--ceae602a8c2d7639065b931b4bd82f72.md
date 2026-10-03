---
title: Mean Excess Function
options:
  concept:
    id: mean-excess-function
    type: theorem
    depends_on:
      - generalized-pareto
      - pareto
    tags:
      - peaks-over-threshold
      - diagnostics
---

## Statement

For a random variable $X$ and a threshold $u$ with $\mathbb P(X>u)>0$ and finite
conditional expectation above $u$, the mean excess function is

$$
e(u)=\mathbb E[X-u\mid X>u].
$$

It measures the expected overshoot beyond $u$ after the threshold has already
been crossed.  For a [Pareto Type I](./incerto-pareto.md) tail with
exponent $\alpha>1$ and $u\ge x_m$,

$$
e(u)=\frac{u}{\alpha-1}.
$$

For a [generalized Pareto variable](./incerto-generalized-pareto.md)
with location zero, shape $\xi$, scale $\beta>0$, and
threshold $u$ in its support with $\beta+\xi u>0$, the mean excess function is
linear when $\xi<1$:

$$
e(u)=\frac{\beta+\xi u}{1-\xi}.
$$

The exponential distribution is the boundary case $\xi=0$, where $e(u)$ is
constant. Here $u$ is the threshold, $\xi$ the GPD shape, and $\beta$
its scale at threshold zero.

## Diagnostic intuition

The mean excess function asks what remains after an event is already large.
Thin-tail intuition often expects the remaining excess to be tame once a high
threshold has been crossed.  A Pareto tail says the opposite: the expected
additional excess grows in proportion to the threshold itself.

This makes the mean excess plot a practical threshold diagnostic.  A roughly
flat plot suggests exponential-like exceedances.  A roughly increasing linear
plot suggests heavy-tail generalized Pareto behavior.  A downward line suggests
a finite endpoint.  Strong curvature usually says the chosen threshold range is
mixing body and tail behavior, or that the model class is too simple.

## Pareto derivation and GPD sketch

We derive the Pareto mean-excess formula from the conditional survival
function.  The generalized Pareto linear formula follows from standard GPD
[threshold stability](./incerto-generalized-pareto.md#threshold-excess-stability-and-pareto-special-case)
and the GPD mean formula. The GPD page derives the stability identity; its
moment formulas are cited rather than reproved here.

For Pareto Type I,

$$
\mathbb P(X>x)=\left(\frac{x_m}{x}\right)^\alpha,
\qquad x\ge x_m.
$$

For $u\ge x_m$ and $\alpha>1$,

$$
e(u)
=
\mathbb E[X-u\mid X>u]
=
\int_0^\infty \mathbb P(X-u>y\mid X>u)\,dy.
$$

Taking the ratio $\bar F(u+y)/\bar F(u)$ for $y\ge0$ gives

$$
\mathbb P(X-u>y\mid X>u)
=
\left(1+\frac{y}{u}\right)^{-\alpha}.
$$

Therefore

$$
e(u)
=
\int_0^\infty \left(1+\frac{y}{u}\right)^{-\alpha}\,dy
=
u\int_1^\infty z^{-\alpha}\,dz
=
\frac{u}{\alpha-1}.
$$

For a generalized Pareto distribution, the threshold-stability property gives
another generalized Pareto distribution above threshold $u$ with updated scale
$\beta+\xi u$, provided $u$ is in the support and $\beta+\xi u>0$.  Its mean
exists only for $\xi<1$, and equals $(\beta+\xi u)/(1-\xi)$.

## Mean-excess diagnostic

Exact theoretical shapes provide a baseline for the empirical diagnostic.
Always report how many observations exceed each threshold.

(mean-excess-diagnostic-plot)=
The theoretical shapes can be compared directly:

| Distribution | Threshold range | Mean excess $e(u)$ |
| --- | --- | --- |
| Pareto, $x_m=1$, $\alpha=1.6$ | $u\ge1$ | $u/0.6$ |
| Exponential with mean $1$ | $u\ge0$ | $1$ |
| Uniform on $[0,5]$ | $0\le u<5$ | $(5-u)/2$ |

For observations $x_1,\ldots,x_n$, let $k(u)=\sum_i\mathbf1_{\{x_i>u\}}$.
The empirical diagnostic is

$$
\widehat e_n(u)=\frac{\sum_{i:x_i>u}(x_i-u)}{k(u)},\qquad k(u)>0.
$$

For the fixed observations $(1,2,2,4)$ at $u=2$, only $4$ exceeds the
threshold, so $k(2)=1$ and $\widehat e_n(2)=2$. At $u=4$ the mean excess
estimate is undefined because the exceedance count is zero. The
[counting example](./incerto-counting-exceedances.md) uses the same strict
inequality convention.

**What to notice.** Pareto mean excess rises with the threshold, exponential
mean excess is flat, and bounded-tail mean excess slopes downward toward the
endpoint.  Empirical values are diagnostics; high thresholds aim to reduce tail-model
bias but leave fewer exceedances and more noise.

## Caveats

- The mean excess function is itself a mean.  If the fitted tail has
  $\xi\ge1$, the theoretical mean excess is infinite.
- Empirical mean excess plots are unstable at high thresholds.  Always inspect
  exceedance counts.
- Linear-looking behavior is suggestive, not decisive.  Mixtures and finite
  upper truncation can create misleading curvature.
- For two-sided returns, apply the function to a one-sided loss variable or to
  absolute returns after stating the modeling choice.

## References

- Davison and Smith, "Models for Exceedances over High Thresholds"
  [1990](https://doi.org/10.1111/j.2517-6161.1990.tb01796.x).
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [2001](https://doi.org/10.1007/978-1-4471-3675-0).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).

## Backlinks

- Depends on: Pickands-Balkema-de Haan Theorem (planned),
  [Generalized Pareto Distribution](./incerto-generalized-pareto.md),
  [Pareto Distribution](./incerto-pareto.md), and threshold notation in
  Notation (planned).
- Used by: S&P 500 Tail Diagnostics (planned) and
  Tail Threshold Selection (planned).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/theorems/mean-excess-function.md`, revision `9717c9c`
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
