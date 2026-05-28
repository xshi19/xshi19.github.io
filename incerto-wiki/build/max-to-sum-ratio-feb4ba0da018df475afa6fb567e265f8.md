# Max-to-Sum Ratio

## Statement

For nonnegative observations $X_1,\dots,X_n$, define

$$
M_n=\max_{1\le i\le n}X_i,
\qquad
S_n=\sum_{i=1}^n X_i,
\qquad
R_n=\frac{M_n}{S_n}.
$$

The ratio $R_n$ measures how much of the total is carried by the single largest
observation.

If the $X_i$ are iid nonnegative with $0<\mathbb E[X_1]<\infty$, then

$$
R_n\to0
\qquad\text{almost surely.}
$$

For regularly varying tails with index $-\alpha$ and $0<\alpha<1$, the ratio
does not collapse to 0. The maximum and the sum live on the same asymptotic
scale, so extremes keep a material share of the aggregate.

## Intuition

In a thin-tailed sum, the largest term eventually becomes small relative to the
total. The average is made by many observations.

In a very heavy-tailed sum, especially when $\alpha<1$, the total is built out
of a few order statistics. The largest observation is not a rounding error; it
is part of the structure of the sum. This is the finite-sample version of the
"tail wags the dog" idea.

## Proof

Assume first that $0<\mathbb E[X_1]<\infty$. The strong law gives

$$
\frac{S_n}{n}\to \mathbb E[X_1]
\qquad\text{almost surely.}
$$

Finite mean also implies $M_n/n\to0$ almost surely. One way to see this is to
use the tail-sum criterion and Borel-Cantelli:

$$
\sum_{n=1}^\infty \mathbb P(X_n>\varepsilon n)<\infty
\qquad(\varepsilon>0).
$$

Thus eventually $X_n\le\varepsilon n$, and the finite set of earlier
observations is negligible when divided by $n$. Hence $M_n/n\to0$. Combining
the two limits,

$$
R_n
=
\frac{M_n/n}{S_n/n}
\to0.
$$

For $0<\alpha<1$ and $\bar F\in RV_{-\alpha}$, the finite-mean proof is
unavailable. Standard extreme-value theory shows that the normalized upper
order statistics converge to the points of an $\alpha$-stable Poisson point
process, and the normalized sum converges to the corresponding stable total.
Consequently $M_n/S_n$ has a non-degenerate limiting behavior rather than a
zero limit. This page cites that theorem instead of re-proving the point
process result.

## Python

```python
import numpy as np
from scipy.stats import pareto
from incerto.estimators import max_to_sum_ratio

rng = np.random.default_rng(20260523)
n = 10_000
reps = 500

for alpha in [0.8, 1.2, 2.5]:
    samples = pareto.rvs(b=alpha, size=(reps, n), random_state=rng)
    ratios = max_to_sum_ratio(samples, axis=1)
    print(alpha, np.median(ratios), np.quantile(ratios, [0.1, 0.9]))
```

The ratio is a diagnostic, not a tail-index estimator. It is useful because it
shows whether a reported total is broad-based or mostly one observation.

## Caveats

- $R_n$ ignores the second, third, and later extremes. For $\alpha<1$, several
  large order statistics may jointly dominate the sum.
- At the boundary $\alpha=1$, the ratio can drift slowly, and finite samples
  can be misleading.
- For signed data, use absolute losses, positive parts, or a domain-specific
  exposure before forming the ratio. Cancellation can hide tail dominance.
- A large ratio in one finite sample is not proof of a power law; it is a cue
  to inspect tail behavior and data-generating assumptions.

## References

- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Taleb, *Statistical Consequences of Fat Tails*, Chapters 3 and 5
  [@taleb2020scoft].

## Backlinks

- Depends on: [Pareto Distribution](../distributions/pareto.md),
  [Regular Variation](../theorems/regular-variation.md), and
  [LLN Failure Under Infinite Mean](../theorems/lln-failure.md).
- Used by: [Pre-Asymptotic LLN Behavior](../examples/lln-preasymptotic.md)
  and the [Chapter 5 reading guide](../../reading-guides/taleb-scoft/ch5.md).
