---
title: LLN Failure Under Infinite Mean
options:
  concept:
    id: lln-failure
    type: theorem
    depends_on:
      - pareto
      - pareto-moment-existence
    tags:
      - law-of-large-numbers
      - infinite-mean
---

## Statement

Let $X_1,X_2,\dots$ be independent copies of a nonnegative random variable
$X$, and let

$$
S_n=\sum_{i=1}^n X_i.
$$

If $\mathbb E[X]=\infty$, then

$$
\frac{S_n}{n}\xrightarrow{\text{a.s.}}\infty.
$$

For a [Pareto Type I variable](./incerto-pareto.md) with lower cutoff
$x_m>0$ and tail exponent $\alpha$, this applies exactly when $\alpha\le1$.
In that regime the sample mean is not estimating a hidden finite number; the
ordinary law of large numbers normalization has failed.

Here $\mathbb E$ is expectation and a.s. means almost surely. The Pareto
parameters $x_m$ and $\alpha$ are the lower cutoff and positive tail exponent.
Shared notation is planned.

## Intuition

The finite-mean law of large numbers says that many small and moderate
observations eventually average out the noise.  With a nonnegative infinite
mean, every finite truncation has an ordinary average, but those truncation
levels can be pushed higher without bound.  The untruncated average must
eventually dominate each one.

For [Pareto tails](./incerto-pareto.md) with $\alpha\le1$, rare
observations are large enough that no finite long-run mean exists.  A running
average may drift downward between records, but the theorem says there is no
stable finite level waiting in the limit.

## Proof sketch

We prove the idea by truncating the variables, applying the strong law to each
bounded truncation, and then letting the cutoff rise.  The Pareto threshold
condition is not reproved here; it is supplied by
[Pareto Moment Existence](./incerto-pareto-moment-existence.md).

For a cutoff $c>0$, define the truncated variable

$$
X_i^{(c)}=\min(X_i,c).
$$

Each $X_i^{(c)}$ is bounded, so the strong law of large numbers gives

$$
\frac1n\sum_{i=1}^n X_i^{(c)}
\xrightarrow{\text{a.s.}}
\mathbb E[\min(X,c)].
$$

Since $X_i\ge X_i^{(c)}$,

$$
\liminf_{n\to\infty}\frac{S_n}{n}
\ge
\mathbb E[\min(X,c)]
$$

almost surely for each fixed $c$.  Apply this on the countable sequence
$c=1,2,3,\dots$.  By monotone convergence,

$$
\mathbb E[\min(X,c)]\uparrow \mathbb E[X]=\infty.
$$

Therefore the liminf of $S_n/n$ is larger than every finite number, almost
surely, which proves $S_n/n\to\infty$ almost surely.

For a Pareto Type I variable,
[Pareto Moment Existence](./incerto-pareto-moment-existence.md) proves that
$\mathbb E[X]$ is infinite exactly when $\alpha\le1$.

## Sample-path behavior

(lln-failure-sample-paths)=
The running average obeys the exact update identity

$$
\frac{S_{n+1}}{n+1}-\frac{S_n}{n}
=\frac{X_{n+1}-S_n/n}{n+1}.
$$

It falls whenever the next observation is below the current average and
rises whenever it is above. Divergence to infinity does not imply a monotone
sample path. Long downward stretches between records are compatible with the
theorem. The [max-to-sum ratio](./incerto-max-to-sum-ratio.md) measures a
different quantity: the largest observation's share of the realized total.
In particular, infinite mean alone does not imply that this share tends to one.

(lln-failure-truncated-mean-check)=
## Capped means for exact Pareto tails

The truncation proof can be inspected directly. For Pareto Type I with $x_m=1$
and $c\ge1$,

$$
\mathbb E[\min(X,c)]
=1+\int_1^c x^{-\alpha}\,dx
=\begin{cases}
1+\dfrac{c^{1-\alpha}-1}{1-\alpha}, & \alpha\ne1,\\
1+\log c, & \alpha=1.
\end{cases}
$$

The following values are evaluations of this identity, rounded to three decimals.
They are capped population means, not sample averages or
$\mathbb E[X\mathbf 1_{\{X\le c\}}]$.

| Exponent $\alpha$ | $c=10$ | $c=1000$ | $c=10^7$ | Limit as $c\to\infty$ |
| --- | --- | --- | --- | --- |
| $0.8$ | $3.924$ | $15.905$ | $121.594$ | $\infty$ |
| $1$ | $3.303$ | $7.908$ | $17.118$ | $\infty$ |
| $1.3$ | $2.663$ | $3.914$ | $4.307$ | $13/3$ |

For a general regularly varying tail at exponent $\alpha=1$, the slowly varying
factor can make the mean finite. The theorem requires infinite mean; the exact
Pareto cutoff $\alpha\le1$ is not a universal boundary rule at equality.
See [Karamata's theorem](./incerto-karamata.md) for the tail-integral criterion.

## Counterexamples and caveats

- We state a one-sided nonnegative result.  If a distribution has large
  positive and negative tails, failure of the ordinary LLN can mean
  non-convergence rather than divergence to $+\infty$.
- The theorem is asymptotic.  A finite sample from an infinite-mean law can
  still show long quiet stretches, especially before the next record-sized
  observation arrives.
- A finite mean does not guarantee a comfortable sample size.  When
  $1<\alpha<2$, the mean exists but the variance is infinite, so convergence of
  the sample average can still be painfully slow.
- Dependence, truncation, censoring, and changing thresholds can alter what a
  real data set shows.  This result is the iid baseline.

## References

- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [1971, 2nd ed., Wiley](https://www.wiley-vch.de/de/fachgebiete/mathematik-und-statistik/an-introduction-to-probability-theory-and-its-applications-volume-2-978-0-471-25709-7).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).

## Backlinks

- Depends on: [Pareto Distribution](./incerto-pareto.md) and the
  shared Notation page (planned).
- Used by: Pre-Asymptotic LLN Behavior (planned) and
  [Max-to-Sum Ratio](./incerto-max-to-sum-ratio.md).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/theorems/lln-failure.md`, revision `9717c9c`
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
