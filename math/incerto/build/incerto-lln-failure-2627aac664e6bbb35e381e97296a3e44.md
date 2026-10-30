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

(lln-modes)=
## Weak and strong laws

Write $\bar X_n=n^{-1}\sum_{i=1}^nX_i$. Convergence in probability to
$\mu$ means $P(|\bar X_n-\mu|>\varepsilon)\to0$ for every
$\varepsilon>0$. Almost-sure convergence means
$P(\lim_n\bar X_n=\mu)=1$. The second implies the first, by bounded
convergence applied to the deviation indicators. Convergence in law to a
constant also implies convergence in probability to that constant.

For iid $X_i$ with $\mathbb E|X_i|<\infty$, the strong law gives
$\bar X_n\to\mathbb EX_1$ almost surely. We cite the strong law from
@durrett2019probability, Theorem 2.4.1, p. 76. Variance need not exist.
We prove the weak law under the same iid integrability assumption here.

Let $X_i^{(c)}=\max(-c,\min(X_i,c))$. For fixed $c$, independence and
Chebyshev give

$$
\begin{gathered}
P\!\left(\left|\overline{X^{(c)}}_n-\mathbb EX^{(c)}\right|>
\varepsilon\right)\\
\le\frac{c^2}{n\varepsilon^2}.
\end{gathered}
$$

Also $\mathbb E|X-X^{(c)}|\to0$ by dominated convergence, and
$|\mathbb EX-\mathbb EX^{(c)}|\le\mathbb E|X-X^{(c)}|$.
Markov bounds the probability that the average of the discarded absolute
remainders exceeds $\varepsilon$ by
$\mathbb E|X-X^{(c)}|/\varepsilon$, uniformly in $n$. The triangle
inequality, first choosing $c$ large and then $n$ large, proves the weak
law. No variance of $X$ was used.

For a concrete distinction between convergence modes, independent events
$B_n$ with $P(B_n)=1/n$ give $\mathbf1_{B_n}\to0$ in probability but
not almost surely: the second Borel–Cantelli lemma gives infinitely many
occurrences (@durrett2019probability, Theorem 2.3.7, p. 70). This example concerns a sequence of indicators, not iid sample
means. For dependent centered observations, a sufficient weak-law condition
is $\operatorname{Var}(S_n)=o(n^2)$, directly by Chebyshev; finite marginal
variances alone do not ensure it.

## Infinite-mean statement

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
See [Shared notation](https://xshi19.github.io/math/notation/).

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

## Proof of nonnegative divergence

We prove the statement by truncating the variables, applying the strong law to each
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
```{figure} ../assets/figures/incerto/lln-failure-paths.svg
:alt: Full Pareto running-mean paths and repeated-sample quantiles on logarithmic axes contrast exponents 0.8, 1 and 1.3; records cause jumps and the infinite-mean ensembles remain broad.

Exact Pareto samples with $x_m=1$ and $\alpha=0.8,1,1.3$. Top: one
10,000-observation path per law, all prefixes retained (PCG64 seeds
20261080–20261082). Bottom: 600 independent samples at each of
$n=30,100,300,1000,3000$ (seeds 20261180–20261182); bands are pointwise
10th/90th sampling percentiles and lines are medians. The dashed reference
$13/3$ exists only for $\alpha=1.3$. Axes are logarithmic; nothing is capped,
trimmed or discarded. The finite experiment illustrates, rather than proves,
the different LLN regimes.
```

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
\begin{gathered}
\mathbb E[\min(X,c)]\\
=1+\int_1^c x^{-\alpha}\,dx\\
=\begin{cases}
1+\dfrac{c^{1-\alpha}-1}{1-\alpha}, & \alpha\ne1,\\
1+\log c, & \alpha=1.
\end{cases}
\end{gathered}
$$

The following values are evaluations of this identity, rounded to three decimals.
They are capped population means, not sample averages or
$\mathbb E[X\mathbf 1_{\{X\le c\}}]$.

| Exponent $\alpha$ | $c=10$ | $c=1000$ | $c=10^7$ | Limit as $c\to\infty$ |
| --- | --- | --- | --- | --- |
| $0.8$ | $3.924$ | $15.905$ | $121.594$ | $\infty$ |
| $1$ | $3.303$ | $7.908$ | $17.118$ | $\infty$ |
| $1.3$ | $2.663$ | $3.914$ | $4.307$ | $13/3$ |

```{figure} ../assets/figures/incerto/lln-capped-means.svg
:alt: Population capped Pareto means grow as a power for exponent 0.8 and logarithmically for exponent 1, while exponent 1.3 approaches the finite mean 13/3.

The exact capped-mean formula on $1\le c\le10^7$, using log axes and
the same three exponents as the sampling figure. These are population
expectations of $\min(X,c)$; the cap is not applied to the preceding samples.
```

[`pareto_capped_mean`](incerto-api-reference.md#api-pareto-capped-mean)
also handles caps below the lower support boundary.

```python
from xmath.incerto.limits import pareto_capped_mean

for alpha in (.8, 1, 1.3):
    values = pareto_capped_mean([.5, 10, 1000], alpha)
    print(f"alpha={alpha:g}: " + ", ".join(f"{v:.4g}" for v in values))
```

**Output**

```text
alpha=0.8: 0.5, 3.924, 15.91
alpha=1: 0.5, 3.303, 7.908
alpha=1.3: 0.5, 2.663, 3.914
```

Regenerate both figures with `uv run python demos/incerto/limit_distributions.py`.

For a general regularly varying tail at exponent $\alpha=1$, the slowly varying
factor can make the mean finite. The theorem requires infinite mean; the exact
Pareto cutoff $\alpha\le1$ is not a universal boundary rule at equality.
See [Karamata's theorem](./incerto-karamata.md) for the tail-integral criterion.

(lln-higher-moments)=
## The LLN for higher absolute moments

For iid $X_i$ and a fixed $p>0$, apply the same theorems to
$Z_i=|X_i|^p$. If $\mathbb E|X|^p<\infty$, the empirical absolute
moment converges almost surely to that expectation. If it is infinite,
the nonnegative divergence proof above gives
$n^{-1}\sum_i|X_i|^p\to\infty$ almost surely. This proves both claims
without requiring a finite variance of $Z_i$.

For exact Pareto, the [moment gate](incerto-pareto-moment-existence.md)
is $p<\alpha$. A finite variance of its empirical $p$th moment instead
requires $2p<\alpha$; [moment estimation](incerto-moment-estimation.md#moment-estimation-gates)
derives the variance. Signed odd powers can have an undefined expectation,
so they cannot replace $|X|^p$ in the nonnegative argument. Empirical
central moments about an estimated center are different statistics.
The [Cauchy example](incerto-cauchy-running.md) makes the distinction concrete.

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

- @durrett2019probability, Theorem 2.4.1, p. 76: strong-law input.
- @taleb2025scoftv4, §7.1, pp. 137–139, and §7.6, pp. 148–150: context.

- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [1971, 2nd ed., Wiley](https://www.wiley-vch.de/de/fachgebiete/mathematik-und-statistik/an-introduction-to-probability-theory-and-its-applications-volume-2-978-0-471-25709-7).
- Resnick, *Heavy-Tail Phenomena* @resnick2007heavy.

## Backlinks

- Depends on: [Pareto Distribution](./incerto-pareto.md) and the
  [Shared notation](https://xshi19.github.io/math/notation/).
- Used by: [Pre-Asymptotic LLN Behavior](./incerto-lln-preasymptotic.md) and
  [Max-to-Sum Ratio](./incerto-max-to-sum-ratio.md).
