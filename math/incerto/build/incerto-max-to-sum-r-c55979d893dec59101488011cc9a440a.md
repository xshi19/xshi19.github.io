---
title: Max-to-Sum Ratio
options:
  concept:
    id: max-to-sum-ratio
    type: method
    depends_on:
      - pareto
    tags:
      - finite-sample
      - tail-risk
---

## Statement

For realized nonnegative observations $x_1,\dots,x_n$ with $s_n>0$, define

$$
s_n=\sum_{i=1}^n x_i,\qquad
m_n=\max_{1\le i\le n}x_i,\qquad
r_n=\frac{m_n}{s_n}.
$$

The realized max-to-sum ratio $r_n$ is the share of the total produced by the
largest observed value.  Its random-sample counterpart is
$R_n=M_n/S_n$, where $S_n=\sum_{i=1}^n X_i$ and
$M_n=\max_{1\le i\le n}X_i$. When $X_1,X_2,\dots$ are iid nonnegative
variables with finite positive mean $\mu=\mathbb E[X]$, then

$$
R_n\to0\qquad\text{almost surely}.
$$

Thus a persistent large realized value $r_n$ is a warning sign that the sample
sum is being governed by extremes rather than by aggregation.  For
[Pareto Type I](./incerto-pareto.md) samples with $\alpha\le1$, the
ordinary mean is infinite, so the finite-mean argument below does not apply.

Lowercase symbols describe the observed list; uppercase symbols describe
random samples. For a nonnegative list with positive total,
$1/n\le r_n\le1$. If the total is zero, the ratio is undefined.
For the almost-sure limit, $S_n/n\to\mu>0$ ensures that $S_n>0$
eventually almost surely; any convention on earlier zero totals gives the
same limit.

## Extreme-dominance intuition

In a thin-tailed or comfortable finite-mean sample, the largest observation may
be memorable, but it should eventually become a negligible fraction of the
whole.  The total grows like $n\mu$, while the largest draw grows more slowly
than $n$ on the scale needed to dominate the sum.

Fat-tailed samples can look different.  One observation can represent a large
fraction of all observed mass, especially near or below the Pareto mean
boundary $\alpha=1$.  This is the same mechanism behind
[LLN Failure Under Infinite Mean](./incerto-lln-failure.md), but $R_n$ makes
the dominance visible without plotting the entire running average.  It
complements the path view in
[Pre-Asymptotic LLN Behavior](./incerto-lln-preasymptotic.md).

## Finite-mean proof

We prove the finite-mean theorem: if the iid nonnegative observations have a
positive finite mean, then the maximum becomes a negligible share of the sum.
The proof uses the strong law and Borel--Cantelli.  The infinite-mean Pareto
regimes are cited or described heuristically, because their exact limits need
heavier regular-variation machinery.

Assume $X_i\ge0$ are iid and $0<\mu=\mathbb E[X]<\infty$.  The strong law
gives $S_n/n\to\mu$ almost surely.  It remains to show that $M_n/n\to0$
almost surely.

For any $\varepsilon>0$, integrability implies

$$
\sum_{n=1}^{\infty}\mathbb P(X_n>\varepsilon n)<\infty,
$$

because the sum is bounded by a constant multiple of
$\int_0^\infty\mathbb P(X>t)\,dt=\mathbb E[X]$.  By Borel--Cantelli,
$X_n/n\to0$ almost surely.  Therefore $M_n/n\to0$ almost surely: after a
random finite index all new observations are at most $\varepsilon n$, while
the finitely many old observations divided by $n$ vanish.  Hence

$$
\frac{M_n}{S_n}=\frac{M_n/n}{S_n/n}\to0
$$

almost surely.

For a Pareto Type I tail,
[Pareto Moment Existence](./incerto-pareto-moment-existence.md) shows that
the mean exists exactly when $\alpha>1$.  The regimes are qualitatively
different:

- $\alpha>1$: $M_n/S_n\to0$ almost surely by the finite-mean theorem.
- Exact Pareto $\alpha=1$: $M_n/S_n\to0$ in probability, but slowly;
  $M_n/(x_m n)$ is bounded in probability, while
  $S_n/(x_m n\log n)\to1$ in probability (a cited limit).
- $0<\alpha<1$: under standard regular-variation assumptions, $M_n/S_n$ has
  a nondegenerate limiting distribution.  Infinite mean does not imply that
  one observation asymptotically equals the whole sum; several of the largest
  observations may keep material shares.

## An exact running-list diagnostic

(max-to-sum-ratio-simulation)=
Consider the illustrative list $1,1,1,1,16,1,1,1,1$. No distribution is fitted
to these numbers; they isolate the effect of a record observation.

| Prefix length $n$ | Total $s_n$ | Maximum $m_n$ | Largest share $r_n$ |
| --- | --- | --- | --- |
| $4$ | $4$ | $1$ | $1/4$ |
| $5$ | $20$ | $16$ | $4/5$ |
| $9$ | $24$ | $16$ | $2/3$ |

Adding the record changes the largest share abruptly. Later small observations
increase the denominator while leaving the maximum fixed. Reordering the list
changes this path but preserves the final ratio.

For an exact Pareto law with $\alpha>1$, the Frechet maximum scale is
$x_m n^{1/\alpha}$, whereas the sum is asymptotic to $n\mu$. Thus a typical
ratio has scale $n^{1/\alpha-1}$: its exponent is $-3/13$ at $\alpha=1.3$
and $-2/3$ at $\alpha=3$. This scale comparison supplies no universal finite-$n$
quantile or confidence bound. The infinite-mean limits cited above require
separate asymptotic results; no Monte Carlo quantiles are reported here.

## Reference distributions for the diagnostic

Use model-based simulation as the primary reference: specify or fit an iid
parent distribution on nonnegative values, simulate independent samples of
the observed size $n$, and compute $R_n$ for each sample. Compare the observed
$r_n$ with that simulated distribution. Report the parent model, fitted
parameters (including $\alpha$ for a Pareto model), sample size, simulation
count, and seed. The reference assumes the stated parent law is appropriate;
a fitted-model comparison is conditional on the fit and needs a separate
assessment of parameter uncertainty before it is used as a calibrated test.
A tail-only fit also needs a model for the body when $S_n$ includes all
observations. A tail exponent alone does not determine a finite-$n$ reference.
If the model permits zero totals, state whether the reference is conditional
on $S_n>0$.

As a complement, use $m$-out-of-$n$ subsampling **without replacement**, with
$m\to\infty$ and $m/n\to0$: compute the ratio on all size-$m$ subsets,
or approximate that distribution using many uniformly sampled subsets.
Validity requires a limiting law for the statistic with the normalization
appropriate to the parent tail. For example,
for a stated exact Pareto exponent, the preceding maximum and sum limits give
the following local normalizing factors $c_k$ for $c_kR_k$:

| Parent regime | $c_k$ |
| --- | --- |
| $\alpha>1$ | $k^{1-1/\alpha}$ |
| $\alpha=1$ | $\log k$ |
| $0<\alpha<1$ | $1$ |

The [Pareto note](./incerto-pareto.md#sums-and-moment-orders) proves the boundary
normalizations and cites the nondegenerate infinite-mean limit. Compare
$c_nr_n$ with the subsample values $c_mr_m$, using the same model parameters.
Raw $m$-sample quantiles are not $n$-sample quantiles when $c_m\ne c_n$.
For a general regularly varying parent, slowly varying factors can change
the normalization. If its parameters are estimated, justify that their
substitution leaves the normalized limit unchanged; mere consistency of an
exponent estimate need not control a power of $n$.

For iid data, a nondegenerate weak limit of the normalized statistic and
$m/n\to0$ justify convergence of the subsampling distribution at continuity
points of the limiting CDF. Estimated centering or normalization must have
negligible effect at the subsample scale. We cite this result from
@politis1994subsampling;
[Bickel, Götze, and van Zwet (1997), Theorem 1, p. 9](https://www3.stat.sinica.edu.tw/statistica/oldpdf/a7n11.pdf#page=9)
gives a proof and an explicit plug-in condition. These are asymptotic
conditions; report the chosen $m$ and sensitivity to it.

Ordinary $n$-out-of-$n$ empirical bootstrap validity is not assumed here.
Its maximum cannot exceed the observed maximum, and with a unique sample
maximum it equals that maximum with conditional probability
$1-(1-1/n)^n\to1-e^{-1}$. This can obstruct a continuous extreme-value
limit. Heavy-tailed sums can also have non-Gaussian stable limits for which
ordinary bootstrap approximations fail; see
[Bickel, Götze, and van Zwet (1997), §2.3, Example 4, p. 5](https://www3.stat.sinica.edu.tw/statistica/oldpdf/a7n11.pdf#page=5).
The finite-mean theorem for $R_n$ proves no resampling validity statement.

## Caveats

- $R_n$ is a dominance diagnostic, not an estimator of the tail exponent.
- A small $R_n$ does not prove thin tails.  A sample may simply not have seen a
  record-sized observation yet.
- A large $R_n$ can also come from data errors, censoring, mixtures, dependence,
  or nonstationarity.  Inspect the observation before interpreting the ratio as
  a tail fact.
- $R_n$ has no universal "large" threshold.  Compare an observed value with a
  reference distribution under the model and calibration assumptions above.
- For signed returns, $M_n/S_n$ may be unstable or meaningless because of
  cancellation.  Apply it to nonnegative losses, severities, exposures, or
  possibly $|X_i|$, explicitly changing the interpretation.
- A running max-to-sum curve is order-dependent, while the final $R_n$ is not.
- For $\alpha<1$, exact limiting behavior of $M_n/S_n$ belongs to stable and
  Poisson point process asymptotics. Here those limits are cited from Resnick
  (2007); only the finite-mean statement is proved.

## References

- Politis and Romano, "Large Sample Confidence Regions Based on Subsamples
  under Minimal Assumptions," *The Annals of Statistics* **22** (1994),
  2031–2050 [DOI](https://doi.org/10.1214/aos/1176325770).
- Bickel, Götze, and van Zwet, "Resampling Fewer than $n$ Observations:
  Gains, Losses, and Remedies for Losses," *Statistica Sinica* **7** (1997),
  1–31; §2.3, Example 4, and §3, Theorem 1
  [PDF](https://www3.stat.sinica.edu.tw/statistica/oldpdf/a7n11.pdf).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).

## Backlinks

- Depends on: [Pareto Distribution](./incerto-pareto.md),
  [LLN Failure Under Infinite Mean](./incerto-lln-failure.md),
  [Pre-Asymptotic LLN Behavior](./incerto-lln-preasymptotic.md), and the
  shared sum notation in [Shared notation](https://xshi19.github.io/math/notation/).
- Related: [Subexponentiality](./incerto-subexponentiality.md).
- Used by: finite-sample tail-risk examples (planned).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/methods/max-to-sum-ratio.md`, revision `9717c9c`
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
