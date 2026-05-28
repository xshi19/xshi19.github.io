---
kernelspec:
  name: python3
  display_name: Python 3
---

# Max-to-Sum Ratio

## Statement

Let $X_1,\dots,X_n$ be nonnegative observations, and define

$$
S_n=\sum_{i=1}^n X_i,\qquad
M_n=\max_{1\le i\le n}X_i,\qquad
R_n=\frac{M_n}{S_n}.
$$

The max-to-sum ratio $R_n$ is the share of the total produced by the largest
single observation.  When $X_1,X_2,\dots$ are iid with finite positive mean
$\mu=\mathbb E[X]$, then

$$
R_n\xrightarrow{\mathbb P}0.
$$

Thus a persistent large value of $R_n$ is a warning sign that the sample sum is
being governed by extremes rather than by aggregation.  For Pareto Type I
samples with $\alpha\le1$, the ordinary mean is infinite, so the finite-mean
argument below does not apply.

## Intuition

In a thin-tailed or comfortable finite-mean sample, the largest observation may
be memorable, but it should eventually become a negligible fraction of the
whole.  The total grows like $n\mu$, while the largest draw grows more slowly
than $n$ on the scale needed to dominate the sum.

Fat-tailed samples can look different.  One observation can represent a large
fraction of all observed mass, especially near or below the Pareto mean
boundary $\alpha=1$.  This is the same mechanism behind
[LLN Failure Under Infinite Mean](../theorems/lln-failure.md), but $R_n$ makes
the dominance visible without plotting the entire running average.  It
complements the path view in
[Pre-Asymptotic LLN Behavior](../examples/lln-preasymptotic.md).

## Proof

Assume $X_i\ge0$ are iid and $0<\mu=\mathbb E[X]<\infty$.  Fix
$\eta>0$.  On the event $S_n/n\ge\mu/2$, the inequality $R_n>\eta$ implies

$$
\frac{M_n}{n}>\frac{\eta\mu}{2}.
$$

Therefore

$$
\mathbb P(R_n>\eta)
\le
\mathbb P\left(\frac{S_n}{n}<\frac{\mu}{2}\right)
+
\mathbb P\left(\frac{M_n}{n}>\frac{\eta\mu}{2}\right).
$$

The first term tends to zero by the weak law of large numbers.  For the second
term, the union bound gives

$$
\mathbb P\left(M_n>\frac{\eta\mu n}{2}\right)
\le
n\,\mathbb P\left(X>\frac{\eta\mu n}{2}\right).
$$

If $X$ is nonnegative and integrable, then $x\,\mathbb P(X>x)\to0$.  One quick
way to see this is to use monotonicity of the survival function:

$$
\frac{x}{2}\mathbb P(X>x)
\le
\int_{x/2}^{x}\mathbb P(X>t)\,dt
\le
\int_{x/2}^{\infty}\mathbb P(X>t)\,dt,
$$

and the final tail integral tends to zero because
$\mathbb E[X]=\int_0^\infty\mathbb P(X>t)\,dt<\infty$.  With
$x=\eta\mu n/2$, this proves

$$
n\,\mathbb P\left(X>\frac{\eta\mu n}{2}\right)\to0.
$$

Both upper-bound terms vanish, so $R_n\to0$ in probability.

For a Pareto Type I tail, [Pareto Distribution](../distributions/pareto.md)
shows that the mean exists exactly when $\alpha>1$.  The proof above is
therefore unavailable when $\alpha\le1$.

## Python

The following simulation estimates the median and $90$th percentile of $R_n$
over repeated Pareto samples.  It uses NumPy's `pareto(alpha) + 1`, which is a
Pareto Type I sample with $x_m=1$.

```{code-cell} python
:label: max-to-sum-ratio-python-check

import numpy as np

from incerto.distributions import pareto_type1
from incerto.estimators import max_to_sum_ratio


def max_share_quantiles(alpha, n, reps=200, batch=25, seed=20260523):
    rng = np.random.default_rng(seed)
    shares = np.empty(reps)

    for start in range(0, reps, batch):
        stop = min(start + batch, reps)
        sample = pareto_type1.rvs(
            alpha, size=(stop - start, n), random_state=rng
        )
        shares[start:stop] = max_to_sum_ratio(sample, axis=1)

    return np.quantile(shares, [0.5, 0.9])


for alpha in (0.8, 1.3, 3.0):
    print(f"Tail exponent alpha={alpha}: median and 90th percentile of R_n")
    print("      n    median      q90")
    for n in (1_000, 10_000, 100_000):
        seed = 20260523 + int(10 * alpha) + n
        q50, q90 = max_share_quantiles(alpha, n, seed=seed)
        print(f"{n:7d}  {q50:8.3f}  {q90:7.3f}")
```

Typical output has a stable contrast: for $\alpha=0.8$, the largest observation
often remains a large share of the sample sum even at $n=100{,}000$; for
$\alpha=3$, the largest share is already tiny at the same sizes.  The
intermediate case $\alpha=1.3$ has a finite mean, but the diagnostic still
declines slowly.

## Caveats

- $R_n$ is a dominance diagnostic, not an estimator of the tail exponent.
- A small $R_n$ does not prove thin tails.  A sample may simply not have seen a
  record-sized observation yet.
- A large $R_n$ can also come from data errors, censoring, mixtures, dependence,
  or nonstationarity.  Inspect the observation before interpreting the ratio as
  a tail fact.
- The finite-mean proof gives convergence in probability.  Stronger almost
  sure versions need additional standard arguments about the growth of
  $M_n$.
- For $\alpha<1$, exact limiting behavior of $M_n/S_n$ belongs to stable and
  Poisson point process asymptotics.  This page only uses simulation and the
  finite-mean contrast.

## References

- Taleb, *Statistical Consequences of Fat Tails*, Chapter 4
  [@taleb2020scoft].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].

## Backlinks

- Depends on: [Pareto Distribution](../distributions/pareto.md),
  [LLN Failure Under Infinite Mean](../theorems/lln-failure.md),
  [Pre-Asymptotic LLN Behavior](../examples/lln-preasymptotic.md), and the
  canonical max and sum notation in [Notation](../../notation/index.md).
- Used by: [Chapter 3 reading guide](../../reading-guides/taleb-scoft/ch3.md)
  and future finite-sample tail-risk examples.
