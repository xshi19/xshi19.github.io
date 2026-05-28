---
kernelspec:
  name: python3
  display_name: Python 3
---

# LLN Failure Under Infinite Mean

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

For a Pareto Type I variable with lower cutoff $x_m>0$ and tail exponent
$\alpha$, this applies exactly when $\alpha\le1$.  In that regime the sample
mean is not estimating a hidden finite number; the ordinary law of large
numbers normalization has failed.

## Intuition

The finite-mean law of large numbers says that many small and moderate
observations eventually average out the noise.  With a nonnegative infinite
mean, every finite truncation has an ordinary average, but those truncation
levels can be pushed higher without bound.  The untruncated average must
eventually dominate each one.

For Pareto tails with $\alpha\le1$, rare observations are large enough that no
finite long-run mean exists.  A running average may drift downward between
records, but the theorem says there is no stable finite level waiting in the
limit.

## Proof

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

For a Pareto Type I variable, the page on
[Pareto Distribution](../distributions/pareto.md) proves that
$\mathbb E[X]$ is infinite exactly when $\alpha\le1$.

## Python

The truncation proof can be inspected without simulation.  For a Pareto Type I
tail with $x_m=1$,

$$
\mathbb E[\min(X,c)]
=1+\int_1^c x^{-\alpha}\,dx.
$$

```{code-cell} python
:label: lln-failure-python-check

import numpy as np

cutoffs = np.logspace(1, 7, 7)
print("Cutoffs:", np.array2string(cutoffs, formatter={"float_kind": lambda x: f"{x:.0e}"}))

for alpha in (0.8, 1.0, 1.3):
    if alpha == 1.0:
        truncated_mean = 1.0 + np.log(cutoffs)
        target = np.inf
    else:
        truncated_mean = 1.0 + (cutoffs ** (1 - alpha) - 1.0) / (1 - alpha)
        target = alpha / (alpha - 1.0) if alpha > 1.0 else np.inf

    target_text = "infinite" if np.isinf(target) else f"{target:.3f}"
    print(f"alpha={alpha}:")
    print(f"  truncated means: {np.array2string(truncated_mean, precision=3)}")
    print(f"  limiting mean: {target_text}")
```

For $\alpha=0.8$ and $\alpha=1$, the truncated means keep growing with the
cutoff.  For $\alpha=1.3$, they approach the finite Pareto mean
$\alpha/(\alpha-1)$.

## Caveats

- This page states a one-sided nonnegative result.  If a distribution has large
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
  [@feller1971introduction].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].
- Taleb, *Statistical Consequences of Fat Tails*, Chapter 4
  [@taleb2020scoft].

## Backlinks

- Depends on: [Pareto Distribution](../distributions/pareto.md) and the
  canonical sum notation in [Notation](../../notation/index.md).
- Used by: [Pre-Asymptotic LLN Behavior](../examples/lln-preasymptotic.md) and
  [Max-to-Sum Ratio](../methods/max-to-sum-ratio.md).

<!-- incerto-provenance:start -->
:::{div} incerto-provenance
**Provenance.** Source: `content/concepts/theorems/lln-failure.md`. Last verified: 2026-05-24. Checked against cited sources, page proof or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
