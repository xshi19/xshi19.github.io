---
title: Hill Estimator
options:
  concept:
    id: hill-estimator
    type: method
    prerequisites:
      - pareto
      - regular-variation
      - extreme-value-index
    tags:
      - tail-index
      - estimation
---

## Overview

How heavy is the upper tail, and how sensitive is the answer to where we
start fitting it? Hill estimates the exponent of a positive
[Pareto-type tail](./incerto-pareto.md) from the largest observations.
Choosing how many observations to retain is part of the estimate.

Read the Pareto and [regular variation](./incerto-regular-variation.md)
pages first. The formula uses logarithms and sorted observations; the
[extreme-value index](./incerto-extreme-value-index.md) $\xi$ is the reciprocal of the
Pareto exponent $\alpha$. For data analysis, inspect several thresholds and check dependence and
measurement limits before interpreting a stable-looking curve. A dedicated
threshold-selection page is planned.

(hill-estimator-definition)=
## Estimator

Let $x_1,\dots,x_n$ be observed right-tail data with $x_i>0$ for every $i$,
and write the ascending realized order statistics as

$$
x_{1:n}\le \cdots \le x_{n:n}.
$$

For $1\le k<n$, the Hill estimate of the extreme-value index is

$$
\widehat\xi_{k,n}
=
\frac1k\sum_{j=1}^{k}
\log\left(\frac{x_{n-j+1:n}}{x_{n-k:n}}\right).
$$

For a [Pareto-type](./incerto-pareto.md) right tail with exponent
$\alpha$, the corresponding tail exponent estimate is

$$
\widehat\alpha_{k,n}=\frac{1}{\widehat\xi_{k,n}}.
$$

The tuning parameter $k$ is the number of upper order statistics used.  A Hill
stability plot graphs $\widehat\xi_{k,n}$ or $\widehat\alpha_{k,n}$ over a
range of $k$ values.  On this page, we name the plotted coordinate explicitly:
$\xi$ is the extreme-value index, while $\alpha=1/\xi$ is the Pareto tail
exponent.

Here $x_{i:n}$ is the $i$th ascending observation, and $k$ selects the top
$k$ values above the order-statistic threshold. If all selected values equal
that threshold, $\widehat\xi=0$ and there is no finite reciprocal estimate.
Ties can also make the strict exceedance count smaller than $k$.

## When to use it

Hill is a right-tail estimator for strictly positive observations in a
Pareto-type regime, meaning the survival tail is expected to be
[regularly varying](./incerto-regular-variation.md).  It is appropriate only
after deciding what data transformation makes the tail positive and
one-sided.

Small $k$ means the threshold is very high, so the estimate uses only the most
extreme observations and has high variance.  Large $k$ lowers the threshold,
which adds data but can mix non-tail observations into the calculation.  A
stable region is a practical compromise between those two failures.

Report a range of $k$ values, the corresponding thresholds, and the sensitivity
of the estimated exponent. A plateau can guide further checks, but it does not
establish regular variation or independence.

(hill-pareto-calibration)=
## Exact Pareto calibration

For a Pareto Type I variable,

$$
\mathbb P(X>x)=\left(\frac{x_m}{x}\right)^\alpha,\qquad x\ge x_m.
$$

Define $Y=\log(X/x_m)$. For $y\ge0$,

$$
\mathbb P(Y>y)
=
\mathbb P(X>x_m e^y)
=e^{-\alpha y},
$$

so $Y$ is exponential with mean $1/\alpha$.  Equivalently,

$$
\mathbb E\left[\log\left(\frac{X}{u}\right)\mid X>u\right]
=\frac1\alpha
$$

for any Pareto threshold $u\ge x_m$.  The Hill estimator replaces this
conditional expectation by the empirical average of log-excesses above the
random threshold $X_{n-k:n}$, or by $x_{n-k:n}$ after the sample is realized.

For exact Pareto samples this explains why $\widehat\xi_{k,n}$ targets
$1/\alpha$.  For a positive iid sample whose right tail is regularly varying
with extreme-value index $\xi>0$, the standard consistency result uses an
intermediate sequence $k=k_n$ with

$$
k_n\to\infty,\qquad \frac{k_n}{n}\to0.
$$

Under these assumptions, $\widehat\xi_{k_n,n}$ converges in probability to
$\xi$.  The proof of that full theorem belongs to extreme-value asymptotics;
we use the exact Pareto calibration and cite the general result in Resnick
[2007](https://doi.org/10.1007/978-0-387-45024-7). Hill's original estimator is
described in his [1975 paper](https://doi.org/10.1214/aos/1176343247),
pp. 1163–1174.

## Stability diagnostics

A stability diagnostic repeats the same calculation over $k$ and reports both
the threshold and the chosen tail coordinate. The arithmetic is visible even
in a fixed four-observation example.

(hill-estimator-stability-diagnostic)=
Take the fixed ordered observations $(1,2,4,8)$. The Hill calculation gives:

| $k$ | Threshold $x_{n-k:n}$ | $\widehat\xi_{k,n}$ | $\widehat\alpha_{k,n}$ |
| --- | --- | --- | --- |
| $1$ | $4$ | $\log 2$ | $1/\log 2$ |
| $2$ | $2$ | $3\log 2/2$ | $2/(3\log 2)$ |
| $3$ | $1$ | $2\log 2$ | $1/(2\log 2)$ |

This small deterministic example shows how the selected threshold changes
the estimate. It is an illustration of the arithmetic, with no claim that the
four values are a random Pareto sample.

In a distribution with a bounded body and a Pareto tail, increasing $k$ far
enough pulls body observations into the calculation. For empirical data,
a plateau is evidence to inspect, not a certificate.

## Failure modes

- Hill is a right-tail estimator for strictly positive data: every realized
  input must satisfy $x_i>0$.  Transform or split two-sided data before using
  it.
- The estimator is threshold-sensitive.  A reported $\widehat\alpha$ should
  include the selected $k$, the threshold, and a stability plot.
- A plateau is suggestive, not a proof of a Pareto tail.  Dependence,
  mixtures, truncation, and measurement limits can all manufacture or destroy
  apparent stability.
- Estimating $\alpha$ and plugging it into moments is dangerous near moment
  boundaries.  Small estimation error around $\alpha=1$ or $\alpha=2$ can
  change whether a mean or variance is treated as finite.
- The reciprocal $\widehat\alpha=1/\widehat\xi$ is unstable when
  $\widehat\xi$ is close to zero.  Thin-tail regimes need different tools.

## References

- Hill, "A Simple General Approach to Inference About the Tail of a
  Distribution" [1975](https://doi.org/10.1214/aos/1176343247).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).

## Backlinks

- Depends on: [Pareto Distribution](./incerto-pareto.md),
  [Regular Variation](./incerto-regular-variation.md), and the canonical
  order statistic notation in Notation (planned).
- Used by: [Extreme Value Index Estimation](./incerto-extreme-value-index.md),
  Tail Threshold Selection (planned), and
  S&P 500 Tail Diagnostics (planned).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/methods/hill-estimator.md`, revision `9717c9c`
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
