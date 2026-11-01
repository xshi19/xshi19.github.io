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

The characteristic function is a cited identity
(@durrett2019probability, §3.3.1, Example 3.3.16, p. 131). The positive and negative
parts both have infinite expectation: integrating $|x|f(x)$ on either
tail gives logarithmic divergence. Thus $\mu$ is a location and median,
not an expectation. See the [stable-law convention](incerto-generalized-central-limit-theorem.md#stable-exact-sums)
for the shared scale parameterization.

(cauchy-average-law)=
## The average has exactly the same law

We prove this finite-sample identity from the stated characteristic function.
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

The average does not converge in probability to $\mu$, or to any finite
constant. This argument alone does not classify every possible pathwise
limit. It also does not say that successive running averages are independent:
they share observations.

The population quantile is
$Q(p)=\mu+\sigma\tan(\pi(p-1/2))$, $0<p<1$.
Consequently the interquartile range of every $\bar X_n$ is $2\sigma$.
By contrast, sample medians consistently estimate $\mu$. For fixed
$\varepsilon>0$, the strong law applied to the indicators
$\mathbf1_{\{X_i\le\mu-\varepsilon\}}$ and
$\mathbf1_{\{X_i\le\mu+\varepsilon\}}$ eventually puts fewer than half the observations at or below the lower
endpoint and more than half at or below the upper endpoint. Every sample
median is then inside that interval. Intersect over rational positive $\varepsilon$ to
obtain almost-sure convergence. This is a proof about the median, not a
restoration of the missing mean.

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

For $p>0$, direct tail integration gives $\mathbb E|X|^p<\infty$ exactly
when $p<1$. At $p\ge1$, the nonnegative
[infinite-mean theorem](incerto-lln-failure.md) applies to $|X_i|^p$:
their sample average tends to $+\infty$ almost surely. It does not apply
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
