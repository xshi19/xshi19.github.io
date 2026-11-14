---
title: Cauchy running statistics
options:
  concept:
    id: cauchy-running
    type: empirical-example
    depends_on: [generalized-central-limit-theorem, lln-failure]
    tags: [stable-laws, simulation, law-of-large-numbers]
---

## A location without a mean

Let $X_1,X_2,\ldots$ be iid Cauchy variables with location $\mu$ and
scale $\sigma>0$. Their density and characteristic function are

$$
\begin{gathered}
f(x)=\frac{\sigma}{\pi((x-\mu)^2+\sigma^2)},\\
\varphi(t)=\exp(i\mu t-\sigma|t|).
\end{gathered}
$$

The characteristic function is a cited identity, not proved here
(@durrett2019probability, §3.3.1, Example 3.3.16, p. 131).
Tail integration establishes the moment boundary.

:::{prf:proposition} Cauchy absolute moments and undefined mean
:label: prop-cauchy-running-moments

Let $X$ have the Cauchy density
$f(x)=\sigma/[\pi((x-\mu)^2+\sigma^2)]$, where
$\mu\in\mathbb R$ and $\sigma>0$.
For every $p>0$, $\mathbb E|X|^p<\infty$ if and only if $p<1$.
Writing $X^+=\max(X,0)$ and $X^-=\max(-X,0)$, we have
$\mathbb EX^+=\mathbb EX^-=\infty$, so the ordinary mean is undefined.
:::

:::{prf:proof}
On either tail, $f(x)\sim\sigma/(\pi x^2)$, so the moment integral
has the same convergence boundary as $\int_1^\infty x^{p-2}\,dx$.
On bounded intervals, $|x|^pf(x)$ is integrable. At $p=1$, each tail
integral diverges logarithmically, proving both assertions about the parts.
:::

Thus $\mu$ is a location and median, not an expectation.
See the [stable-law convention](incerto-generalized-central-limit-theorem.md#stable-exact-sums)
for the shared scale parameterization.

(cauchy-average-law)=
## The average has exactly the same law

We prove this finite-sample identity from the stated characteristic function.

:::{prf:proposition} Cauchy averages retain their law and spread
:label: prop-cauchy-running-average-law

Let $X_1,X_2,\ldots$ be iid Cauchy variables with location
$\mu\in\mathbb R$ and scale $\sigma>0$. For every integer $n\ge1$,
$\bar X_n=n^{-1}\sum_{i=1}^nX_i$ has the same law as $X_1$, and
for every $\varepsilon>0$,

$$
P(|\bar X_n-\mu|>\varepsilon)
=1-\frac2\pi\arctan(\varepsilon/\sigma)>0.
$$

The sequence $\bar X_n$ cannot converge in probability to any finite
constant. Its $p$-quantile is $\mu+\sigma\tan(\pi(p-1/2))$ for
$0<p<1$, so its interquartile range is $2\sigma$ for every $n$.
:::

:::{prf:proof}
Independence gives, for every integer $n\ge1$,

$$
\begin{aligned}
\mathbb E e^{it\bar X_n}&=\varphi(t/n)^n\\
&=\exp(i\mu t-\sigma|t|).
\end{aligned}
$$

Uniqueness of characteristic functions therefore gives
$\bar X_n\overset d=X_1$. In particular,

$$
\begin{gathered}
P(|\bar X_n-\mu|>\varepsilon)\\
=1-\frac2\pi\arctan(\varepsilon/\sigma)>0\\
(\varepsilon>0).
\end{gathered}
$$

Integrating the density gives
$F(x)=1/2+\pi^{-1}\arctan((x-\mu)/\sigma)$, which yields the displayed
deviation probability. For any finite constant $c$ and $\varepsilon>0$,
$P(|\bar X_n-c|>\varepsilon)=P(|X_1-c|>\varepsilon)>0$ is independent
of $n$, ruling out convergence in probability to $c$.
Inverting $F$ gives the stated quantiles; subtracting the quartiles gives
$2\sigma$.
:::

**Remarks.** This argument alone does not classify every possible pathwise
limit. Successive running averages share observations and are not asserted
to be independent.

By contrast, the indicator strong law establishes consistency of medians.

:::{prf:proposition} Strong consistency of Cauchy sample medians
:label: prop-cauchy-running-median-consistency

Let $X_1,X_2,\ldots$ be iid Cauchy variables with location
$\mu\in\mathbb R$ and scale $\sigma>0$. For each $n$, let $m_n$ be
any sample median: at least half of $X_1,\ldots,X_n$ are at or below
$m_n$ and at least half are at or above it. Then $m_n\to\mu$ almost surely.
:::

:::{prf:proof}
For fixed
$\varepsilon>0$, the strong law applied to the indicators
$\mathbf1_{\{X_i\le\mu-\varepsilon\}}$ and
$\mathbf1_{\{X_i\le\mu+\varepsilon\}}$ eventually puts fewer than half the observations at or below the lower
endpoint and more than half at or below the upper endpoint. Every sample
median is then inside that interval, since
$F(\mu-\varepsilon)<1/2<F(\mu+\varepsilon)$.
Intersect over rational positive $\varepsilon$ to obtain almost-sure
convergence.
:::

**Remark.** Median consistency does not restore the missing mean.

(cauchy-running-figures)=
## Paths and repeated samples

```{figure} ../assets/figures/incerto/cauchy-running.svg
:alt: Three full Cauchy running-mean paths jump on a signed logarithmic axis; a second panel compares persistent mean spread with shrinking sample-median spread.

Standard Cauchy draws, $\mu=0$, $\sigma=1$. The first panel retains all
10,000 prefixes of three paths (PCG64 seed 20261071); the signed logarithmic
vertical axis has a linear region between −2 and 2. The second shows
pointwise 10th/50th/90th sampling percentiles over 600 independent samples
at each of five sizes (seed 20261072). Dashed horizontal lines are the
exact 10th/90th quantiles of the mean. Bands are sampling quantiles, not
confidence intervals or simultaneous path bands. No draws are capped or
removed; the computation, not the distribution, is bounded.
```

Use [`running_mean`](incerto-api-reference.md#api-running-mean) for one
path and [`repeated_centered_sums`](incerto-api-reference.md#api-repeated-centered-sums)
for independent repetitions. Here centering at zero names the Cauchy
location; it does not assert a population mean.

```python
import numpy as np
from scipy.stats import cauchy
from xmath.incerto.limits import repeated_centered_sums

n = np.array([10, 100, 1000])
sums = repeated_centered_sums(cauchy(), n, repetitions=400, random_state=71)
for size, row in zip(n, sums / n[:, None]):
    q = np.quantile(row, [.1, .5, .9])
    print(f"n={size}: " + ", ".join(f"{v:.4g}" for v in q))
```

**Output**

```text
n=10: -2.694, 0.01121, 2.404
n=100: -3.183, -0.05087, 3.134
n=1000: -3.107, 0.04288, 2.812
```

Regenerate the figure with `uv run python demos/incerto/limit_distributions.py`.
Finite floating-point observations do not establish finite population moments.
The runner raises on nonfinite output rather than dropping observations.

## Higher moments and ergodicity

The moment boundary gives a direct application of the nonnegative
[infinite-mean theorem](incerto-lln-failure.md), which is proved in that note.

:::{prf:corollary} Divergence of averages of Cauchy absolute powers
:label: cor-cauchy-running-absolute-power-averages

If $X_1,X_2,\ldots$ are iid Cauchy variables with location
$\mu\in\mathbb R$ and scale $\sigma>0$, then for each fixed $p\ge1$,

$$
\frac1n\sum_{i=1}^n|X_i|^p\longrightarrow+\infty
\qquad\text{almost surely}.
$$
:::

:::{prf:proof}
By [](#prop-cauchy-running-moments), $\mathbb E|X_i|^p=\infty$.
The variables $|X_i|^p$ are iid and nonnegative, so the cited
infinite-mean theorem applies.
:::

**Remark.** That theorem does not apply
to the signed Cauchy observations themselves. A principal-value cancellation
is not a Lebesgue expectation.

An iid Cauchy sequence is ergodic under the shift. The integrability
hypothesis of the [ergodic averaging theorem](incerto-ergodicity.md)
fails for the identity observable. Calling the process nonergodic because
its sample mean fails would confuse two different conditions.

## Sources

@taleb2025scoftv4, §§7.2.1–7.2.2, pp. 139–141, and §7.6, pp. 148–150,
motivate the experiment. The finite-sample average identity, quantile
calculation, and median argument above are derived here.
