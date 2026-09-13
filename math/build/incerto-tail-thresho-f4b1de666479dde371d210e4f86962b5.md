---
title: Tail Threshold Selection
options:
  concept:
    id: tail-threshold-selection
    type: method
    depends_on:
      - extreme-value-index
      - hill-estimator
      - mean-excess-function
      - generalized-pareto
    tags:
      - estimation
      - diagnostics
      - tail-risk
---

## Statement

Tail threshold selection is the modeling step that chooses where the body of a
sample ends and where a tail estimator or peaks-over-threshold model begins.
For a threshold $u$, the exceedance sample is

$$
\{x_i-u:x_i>u\}.
$$

For a [Hill estimator](./incerto-hill-estimator.md), the equivalent tuning parameter is
the number $k$ of upper order statistics, with empirical threshold
$x_{n-k:n}$.  Hill is designed for positive, regularly varying right tails
with extreme-value index $\xi>0$; for a Pareto-type tail,
$\alpha=1/\xi$.  A threshold selection report should include at least:

- the candidate thresholds or $k$ values;
- the number of exceedances retained;
- Hill or EVI stability over the candidate range;
- mean-excess or GPD-shape stability where relevant;
- the reason the selected range was kept or rejected.

The goal is not to find a universally correct threshold.  It is to make the
bias-variance tradeoff visible.

Here $x_{1:n}\le\cdots\le x_{n:n}$ are the sorted observations,
$k$ counts the upper values used by Hill, and $\xi$ is the
extreme-value index. Ties at the threshold can make the strict exceedance
count differ from $k$ and should be reported explicitly.

## Bias-variance intuition

Every tail model asks the same awkward question: how far out is "tail"?  Set
the threshold too low and the estimator is biased by body observations.  Set
it too high and the estimator is dominated by too few extremes.  A good
threshold analysis therefore looks for a region where several diagnostics stop
moving violently while enough exceedances remain to estimate anything at all.

This page is distinct from the
[body-shoulder-tail diagnostic](./incerto-body-shoulder-tail.md).  Body/shoulder geometry
describes how a distribution's density responds to variance mixing.  Threshold
selection is an empirical modeling decision for tail estimation.

## Diagnostics

For a one-sided positive sample:

1. Choose a grid of candidate thresholds, often empirical quantiles.
2. Record exceedance counts at each threshold.
3. Plot Hill estimates over $k$ or threshold values.
4. Plot the [mean-excess function](./incerto-mean-excess-function.md) for the
   same threshold range.
5. If using a GPD model, fit shape and scale over multiple thresholds.
6. Prefer a range where estimates are reasonably stable and exceedance counts
   are not too small.

The accepted range should be reported, not hidden.  Downstream quantities such
as [moment existence](./incerto-pareto-moment-existence.md), return levels, or
expected shortfall can be highly sensitive to $\xi$. Under an exact GPD model,
finite variance requires $\xi<1/2$, while finite mean and expected shortfall
require $\xi<1$. For a general regularly varying tail, moments at the boundary
also depend on the slowly varying factor. High return-level estimates are
sensitive to $\xi$ but do not have mathematical singularities specifically
at those two values.

## Static body-plus-tail example

(tail-threshold-selection-diagnostics)=
Consider the population model from the source example: with probability
$0.82$, $X$ is uniform on $[1,3]$; with probability $0.18$, it is Pareto
with cutoff $3$ and exponent $\alpha=1.7=17/10$. Thus

$$
\bar F(u)=
\begin{cases}
1, & u<1,\\
0.82(3-u)/2+0.18, & 1\le u<3,\\
0.18(3/u)^{1.7}, & u\ge3.
\end{cases}
$$

Above $3$, the conditional tail is exactly Pareto. Its shape is $\xi=10/17$,
its excess scale is $\beta(u)=10u/17$, and its mean excess is
$e(u)=u/(\alpha-1)=10u/7$. For an iid sample of size $n=30{,}000$,
the expected number of exceedances is $n\bar F(u)$.

| Threshold $u$ | Expected exceedances (rounded) | Exact tail mean excess | Modified scale $\beta(u)-\xi u$ |
| --- | --- | --- | --- |
| $2$ | $17{,}700$ | Body and tail both contribute | GPD tail formula does not apply |
| $3$ | $5{,}400$ | $30/7$ | $0$ |
| $6$ | $1{,}662$ | $60/7$ | $0$ |
| $12$ | $512$ | $120/7$ | $0$ |

These counts are expectations under a specified population, not realized sample
counts or fitted estimates. At $u=2$, the exact mean excess instead is

$$
e(2)=\frac{0.205+0.18(51/7-2)}{0.59}
\approx1.960,
$$

where $0.205$ is the body contribution to $\mathbb E[(X-2)_+]$ and $51/7$
is the conditional tail mean. Extending the Pareto line below its cutoff
would incorrectly give $20/7$.

At thresholds at least $3$, raising $u$ introduces no tail-model bias in this
exact construction but still loses observations. In empirical data, the cutoff
is unknown and there may be no exact Pareto region at any finite threshold.
The table therefore does not select an empirical threshold or report an
uncertainty interval.

For a general GPD tail, threshold stability is
$\beta(u')=\beta(u)+\xi(u'-u)$. Hence shape and modified scale
$\beta(u)-\xi u$ should be inspected together. The simpler relation
$\beta(u)/u=\xi$ is specific to the exact Pareto tail. Selected-threshold
uncertainty, dependence, and tail approximation error are additional to the
sampling uncertainty of a fit at a fixed threshold.

## Caveats

- Threshold selection is a modeling judgment, not a theorem.  Different
  diagnostics can disagree.
- The Hill estimator is a right-tail regular-variation tool for $\xi>0$, not a
  general estimator for Gumbel-type or bounded-tail domains.
- The ratio $\hat\beta(u)/u$ is an exact-Pareto diagnostic.  For a general GPD
  threshold model, threshold stability is
  $\beta(u')=\beta(u)+\xi(u'-u)$, so $\beta(u)-\xi u$ is the stable quantity.
- Very high thresholds may reduce approximation bias but can leave too few
  exceedances for stable estimation.
- Dependence, volatility clustering, rounding, reporting limits, and
  truncation can all make threshold diagnostics misleading.
- A selected threshold for one sample window or horizon should not be reused
  mechanically for a different dataset.
- Thresholds for right-tail positive data do not automatically apply to
  two-sided returns or losses without an explicit transformation.

## References

- Hill, "A Simple General Approach to Inference About the Tail of a
  Distribution" [1975](https://doi.org/10.1214/aos/1176343247).
- Davison and Smith, "Models for Exceedances over High Thresholds"
  [1990](https://doi.org/10.1111/j.2517-6161.1990.tb01796.x).
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [2001](https://doi.org/10.1007/978-1-4471-3675-0).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).

## Backlinks

- Depends on: [Extreme-Value Index Estimation](./incerto-extreme-value-index.md),
  [Hill Estimator](./incerto-hill-estimator.md),
  [Mean Excess Function](./incerto-mean-excess-function.md), and
  [Generalized Pareto Distribution](./incerto-generalized-pareto.md).
- Related: [Body, Shoulders, and Tails](./incerto-body-shoulder-tail.md) is a density
  geometry diagnostic, not a threshold-selection rule.
- Used by: S&P 500 Tail Diagnostics (planned).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/methods/tail-threshold-selection.md`, revision `9717c9c`
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
