---
options:
  concept:
    id: lln-preasymptotic
    type: empirical-example
    depends_on:
    - pareto
    - lln-failure
    tags:
    - finite-sample
    - simulation
title: Pre-Asymptotic LLN Behavior
---

## Setup

This synthetic example plots running means

$$
\bar X_n=\frac{S_n}{n}
$$

for iid [Pareto Type I](incerto-pareto.md) samples with $x_m=1$.  The
target contrast is:

- if $\alpha\le1$, [LLN Failure Under Infinite Mean](incerto-lln-failure.md)
  says $\bar X_n\to\infty$ almost surely;
- if $\alpha>1$, the finite-mean law of large numbers says
  $\bar X_n\to\alpha/(\alpha-1)$, but convergence can still be slow when
  $\alpha$ is close to 1.

The symbols $\bar X_n$, $S_n$, $\alpha$, and $x_m$ follow the shared
[notation table](https://xshi19.github.io/math/notation/).

## What the simulation isolates

The simulation uses exact iid Pareto draws.  It does not include dependence,
threshold fitting, censoring, mixtures, or measurement error.  That narrow
setup is deliberate: it isolates why the theorem can be practically easy to
underestimate even before empirical complications enter.

The inverse-transform representation is

$$
X=(1-U)^{-1/\alpha},\qquad U\sim \operatorname{Uniform}(0,1).
$$

The distribution helper in `xmath.incerto.distributions` implements the same Pareto
Type I law.

## Running-mean paths

The left panel shows single running-mean paths at log-spaced checkpoints.  The
right panel summarizes repeated final sample means at several sample sizes.

Run `python demos/incerto/lln_preasymptotic.py` after installing `xshi-math[incerto]`
to reproduce these static figures. The script records the sample sizes and
random seeds where simulation is used.

![lln preasymptotic diagnostic 1](../assets/incerto/figures/lln-preasymptotic-1.svg)


For $\alpha\le1$, no finite theoretical mean exists.  For
$\alpha=1.3$, the finite mean exists, but large observations still create
visible pre-asymptotic jumps and wide repeated-simulation bands.

## Interpretation and caveats

- One simulated path is not a proof and not an estimator of $\alpha$.
- The plot uses log axes.  This makes early and late behavior comparable, but
  it also visually compresses large jumps.
- A path with $\alpha>1$ can look worse than a path with $\alpha\le1$ over a
  finite window if it happens to receive an unusually large early draw.
- Real data adds threshold choice, dependence, measurement limits, and model
  uncertainty.  The example isolates only the iid Pareto mechanism.

## References

- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [@feller1971introduction].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].

## Backlinks

- Depends on: [Pareto Distribution](incerto-pareto.md) and
  [LLN Failure Under Infinite Mean](incerto-lln-failure.md).
- Used by: [Max-to-Sum Ratio](incerto-max-to-sum-ratio.md) and
  [Generalized Central Limit Theorem](incerto-generalized-central-limit-theorem.md).

## Recorded calculation

The committed demo produced the following on 2026-10-17. These are sample
calculations, with seeds and sample sizes specified in the script.

```text
Pareto alpha=0.8:
  final running mean after 300,000 draws: 454.983
  largest observation's share of total: 0.600
  finite theoretical mean: none
Pareto alpha=1.0:
  final running mean after 300,000 draws: 12.020
  largest observation's share of total: 0.070
  finite theoretical mean: none
Pareto alpha=1.3:
  final running mean after 300,000 draws: 4.100
  largest observation's share of total: 0.010
  finite theoretical mean: 4.333
```

## Source and license

Adapted from `xshi19/incerto-wiki`, `content/concepts/examples/lln-preasymptotic.md`,
at revision `9717c9cf83aee18b47f423a14bce2f0d14bb8bcb`.
The original author-controlled material is used under MIT; the cited books
and external datasets retain their own rights. Calculations are static and
are not executed by the site build.

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
