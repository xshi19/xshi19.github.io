---
kernelspec:
  name: python3
  display_name: Python 3
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

For a [Pareto Type I variable](../distributions/pareto.md) with lower cutoff
$x_m>0$ and tail exponent $\alpha$, this applies exactly when $\alpha\le1$.
In that regime the sample mean is not estimating a hidden finite number; the
ordinary law of large numbers normalization has failed.

The symbols $X_i$, $S_n$, $\mathbb E$, $\alpha$, and $x_m$ follow the shared
[notation table](../../notation/index.md).

## Intuition

The finite-mean law of large numbers says that many small and moderate
observations eventually average out the noise.  With a nonnegative infinite
mean, every finite truncation has an ordinary average, but those truncation
levels can be pushed higher without bound.  The untruncated average must
eventually dominate each one.

For [Pareto tails](../distributions/pareto.md) with $\alpha\le1$, rare
observations are large enough that no finite long-run mean exists.  A running
average may drift downward between records, but the theorem says there is no
stable finite level waiting in the limit.

## Proof sketch

We prove the idea by truncating the variables, applying the strong law to each
bounded truncation, and then letting the cutoff rise.  The Pareto threshold
condition is not reproved here; it is supplied by
[Pareto Moment Existence](pareto-moment-existence.md).

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
[Pareto Moment Existence](pareto-moment-existence.md) proves that
$\mathbb E[X]$ is infinite exactly when $\alpha\le1$.

## Sample-path behavior

The theorem is asymptotic, but a simulation makes the mechanism visible.  The
left panel plots running means under infinite-mean and finite-mean Pareto
regimes.  The right panel plots the running maximum's share of the running sum.

```{code-cell} python
:label: lln-failure-sample-paths
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

from incerto.distributions import pareto_type1
from incerto.figures import FIGURE_SIZES, set_theme, style_axes

set_theme()

rng = np.random.default_rng(20260617)
sample_size = 30_000
n = np.arange(1, sample_size + 1)
alphas = (0.8, 1.0, 1.3)

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"])

for alpha in alphas:
    sample = pareto_type1.rvs(alpha, size=sample_size, random_state=rng)
    cumulative_sum = np.cumsum(sample)
    running_mean = cumulative_sum / n
    max_share = np.maximum.accumulate(sample) / cumulative_sum

    axes[0].plot(n, running_mean, label=fr"$\alpha={alpha}$")
    axes[1].plot(n, max_share, label=fr"$\alpha={alpha}$")

finite_mean = 1.3 / (1.3 - 1.0)
axes[0].axhline(
    finite_mean,
    color="black",
    linestyle="--",
    linewidth=1.1,
    label=r"mean for $\alpha=1.3$",
)

axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("sample size n")
axes[0].set_ylabel(r"running mean $S_n/n$")
axes[0].set_title("Running means")
axes[0].legend()

axes[1].set_xscale("log")
axes[1].set_xlabel("sample size n")
axes[1].set_ylabel(r"running max share $M_n/S_n$")
axes[1].set_title("Extreme contribution")
axes[1].legend()

style_axes(axes, grid_axis="both")
plt.show()
```

**What to notice.** The infinite-mean paths do not settle around a finite
target, and new records can move a large share of the cumulative sum.  The
$\alpha=1.3$ path has a finite mean, but it still converges slowly because the
variance is infinite.

The truncation proof can also be inspected without simulation.  For a Pareto
Type I tail with $x_m=1$,

$$
\mathbb E[\min(X,c)]
=1+\int_1^c x^{-\alpha}\,dx.
$$

```{code-cell} python
:label: lln-failure-truncated-mean-check
:tags: [hide-input]

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
  [@feller1971introduction].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].

## Backlinks

- Depends on: [Pareto Distribution](../distributions/pareto.md) and the
  canonical sum notation in [Notation](../../notation/index.md).
- Used by: [Pre-Asymptotic LLN Behavior](../examples/lln-preasymptotic.md) and
  [Max-to-Sum Ratio](../methods/max-to-sum-ratio.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/theorems/lln-failure.md`. Last verified: 2026-06-25. Checked against cited sources, scoped proof sketch, and executable sample-path and truncation diagnostics.
:::
<!-- incerto-provenance:end -->
