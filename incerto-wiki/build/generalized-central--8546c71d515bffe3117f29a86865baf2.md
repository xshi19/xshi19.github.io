---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: generalized-central-limit-theorem
    type: theorem
    depends_on:
      - regular-variation
      - pareto-moment-existence
    proof_depends_on: [pareto]
    tags:
      - stable-laws
      - sums
      - infinite-variance
---

# Generalized Central Limit Theorem

(generalized-clt-statement)=
## Statement

The ordinary central limit theorem uses finite variance and normalizes sums by
$\sqrt{n}$.  The substantive generalized central limit theorem is a
classification theorem: a nondegenerate distribution can occur as a weak limit
of centered and normalized iid sums if and only if it is stable.  In other
words, if there are constants $a_n>0$ and $b_n$ such that

$$
\frac{X_1+\cdots+X_n-b_n}{a_n}
\Rightarrow Z,
$$

for a nondegenerate limit $Z$, then $Z$ must be stable; conversely, stable laws
arise as such limits for suitable iid summands.  The Gaussian law is the stable
case $\alpha=2$.  The heavy-tailed stable cases have index $0<\alpha<2$.

For a two-sided regularly varying sufficient condition, assume

$$
\mathbb P(|X|>x)=x^{-\alpha}L(x),\qquad 0<\alpha<2,
$$

with tail balance

$$
\frac{\mathbb P(X>x)}{\mathbb P(|X|>x)}\to p,\qquad
\frac{\mathbb P(X<-x)}{\mathbb P(|X|>x)}\to q,\qquad p+q=1.
$$

Choose $a_n$ so that $n\mathbb P(|X|>a_n)\to1$.  Then, with suitable
centering constants $b_n$,

$$
\frac{X_1+\cdots+X_n-b_n}{a_n}
\Rightarrow Z_\alpha,
$$

where $Z_\alpha$ is an $\alpha$-stable law whose skewness is determined by
$p-q$.  For nonnegative Pareto-type examples this reduces to a strongly
right-skewed stable limit, and the scaling is of order $n^{1/\alpha}$ up to a
slowly varying factor.  When $1<\alpha<2$, the mean exists but the variance is
infinite, so a common centering choice is $b_n=n\mathbb E[X]$.

A common centering summary is

$$
b_n=
\begin{cases}
0, & 0<\alpha<1\quad\text{in the usual nonnegative case},\\
n\mathbb E[X\mathbf 1_{\{|X|\le a_n\}}], & \alpha=1\quad\text{as a standard choice},\\
n\mathbb E[X], & 1<\alpha<2.
\end{cases}
$$

The symbols $X_i$, $\alpha$, $S_n$, and $\mathbb E$ follow the shared
[notation table](../../notation/index.md); $a_n$, $b_n$, $p$, $q$, and
$Z_\alpha$ are local to this statement.

We record the stable-limit contrast needed by the wiki's heavy-tail
pages.  We do not classify every possible domain of attraction or every
parameterization of stable laws.

The classification and sufficient tail condition above are cited from
Feller [@feller1971introduction] and Resnick [@resnick2007heavy]; the local
argument below addresses only exact Pareto maximum scaling.

## Why $\sqrt{n}$ fails

The normal law is not the only possible attractor for sums.  It is the
finite-variance attractor.  When the tail is heavy enough that variance does
not exist, the largest observations remain visible at the same scale as the
sum, and the $\sqrt{n}$ normalization is no longer the right one.

For a [Pareto-type tail](../distributions/pareto.md) with exponent
$\alpha<2$, the natural scale of the maximum is about $n^{1/\alpha}$.  Stable
normalization puts the sum on that same order.  This is why infinite-variance
sums can keep producing large jumps instead of smoothing into Gaussian-looking
noise.

## Scaling argument

The full generalized central limit theorem is a stable-law classification
result, so we cite it rather than reproduce the proof.  What we prove
directly is the exact Pareto maximum scaling calculation, which shows why the
usual $\sqrt n$ scale is too small when $\alpha<2$.

For a [Pareto Type I](../distributions/pareto.md) variable with survival
$\bar F(x)=(x_m/x)^\alpha$, choose $a_n=x_m n^{1/\alpha}$.  Then for $y>0$ and sufficiently large $n$ such that $a_n y\ge x_m$,

$$
\mathbb P(M_n/a_n\le y)
=
\left(1-\frac{1}{ny^\alpha}\right)^n
\to
e^{-y^{-\alpha}}.
$$

The maximum remains of order $a_n$.  Since $a_n$ grows faster than
$\sqrt{n}$ when $\alpha<2$, the Gaussian finite-variance scaling is not the
right asymptotic scale for such tails.  This maximum calculation identifies the
right order of extreme observations; it is a scaling heuristic, not a proof of
stable convergence for sums.

## Pareto simulation

The simulation below uses Pareto samples with $\alpha=1.5$.  The mean exists,
but the variance does not.  After subtracting $n\mathbb E[X]$, the
$n^{1/\alpha}$-scaled sums have a more stable central spread than the same sums
scaled by $\sqrt{n}$.

```{code-cell} python
:label: generalized-clt-scaling-simulation
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

from incerto.distributions import pareto_type1
from incerto.figures import FIGURE_SIZES, set_theme, style_axes

set_theme()

alpha = 1.5
mean = alpha / (alpha - 1.0)
rng = np.random.default_rng(20260617)
repetitions = 2_000
n_values = np.array([200, 500, 1_000, 3_000])

stable_iqr = []
normal_iqr = []
last_stable = None
last_normal = None

for n_obs in n_values:
    samples = pareto_type1.rvs(alpha, size=(repetitions, n_obs), random_state=rng)
    centered = samples.sum(axis=1) - n_obs * mean
    stable_scaled = centered / (n_obs ** (1 / alpha))
    normal_scaled = centered / np.sqrt(n_obs)
    stable_iqr.append(np.subtract(*np.quantile(stable_scaled, [0.75, 0.25])))
    normal_iqr.append(np.subtract(*np.quantile(normal_scaled, [0.75, 0.25])))
    last_stable = stable_scaled
    last_normal = normal_scaled

fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.6), constrained_layout=True)

axes[0].plot(n_values, stable_iqr, marker="o", label=fr"$n^{{1/\alpha}}$ scaling")
axes[0].plot(n_values, normal_iqr, marker="o", label=r"$\sqrt{n}$ scaling")
reference_iqr = normal_iqr[0] * (n_values / n_values[0]) ** (1 / 6)
axes[0].plot(n_values, reference_iqr, linestyle="--", color="gray",
             label=r"reference slope $n^{1/6}$")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("sample size n")
axes[0].set_ylabel("interquartile range")
axes[0].set_title("Spread under two normalizations")
axes[0].legend()

combined = np.concatenate([last_stable, last_normal])
lo, hi = np.quantile(combined, [0.02, 0.98])
bins = np.linspace(lo, hi, 45)
axes[1].hist(last_stable, bins=bins, alpha=0.65, weights=np.full(last_stable.size, 1 / (last_stable.size * (bins[1]-bins[0]))), label=fr"$n^{{1/\alpha}}$")
axes[1].hist(last_normal, bins=bins, alpha=0.45, weights=np.full(last_normal.size, 1 / (last_normal.size * (bins[1]-bins[0]))), label=r"$\sqrt{n}$")
axes[1].set_xlabel("centered and scaled sum")
axes[1].set_ylabel("density")
axes[1].set_title(f"Scaled sums at n={n_values[-1]} (pooled 2–98% range)")
axes[1].legend()

style_axes(axes, grid_axis="both")
plt.show()
```

The stable scaling keeps the central spread roughly in the
same range as $n$ grows.  The $\sqrt{n}$ scaling spreads out because it is too
small for an infinite-variance Pareto tail; for $\alpha=1.5$ its spread grows
roughly like $n^{1/\alpha-1/2}=n^{1/6}$.  This is a simulation of the scaling
contrast, not a finite-sample proof of the full stable limit. The dashed line
shows slope $n^{1/6}$ anchored at the first simulated spread. The histograms
show only the pooled 2nd–98th percentile range; their heights retain full-sample
normalization, so cropped probability is not redistributed into the visible bins.

## Caveats

- Stable-law parameterizations vary across books and software.  We use
  only the stable index $\alpha$ and avoid detailed skew/location notation.
- Regular variation with tail balance is a clean sufficient setting, not the
  only possible stable domain-of-attraction condition.
- Right-tail regular variation alone is not enough for a general two-sided
  variable; the left tail can change the scale or skewness of the limit.
- Some infinite-variance distributions can still be in the Gaussian domain of
  attraction when their truncated second moment is slowly varying.
- For $0<\alpha<1$, the absolute first moment is infinite. At $\alpha=1$,
  its finiteness depends on the slowly varying factor: the tail-integral
  criterion is $\int^\infty L(x)/x\,dx<\infty$. For example, a nonnegative
  variable with survival $1/[x(\log x)^2]$ for $x\ge e$ has a finite mean,
  since that tail integral equals $1$. Exact Pareto tails have infinite mean
  at $\alpha=1$. See the boundary discussion in [Karamata's theorem](karamata.md).
  The truncated centering above remains a standard choice at this boundary;
  finite mean alone does not justify replacing it by $n\mathbb E[X]$ without
  checking the resulting location shift on the $a_n$ scale.
- Finite samples can look calmer than the asymptotic theory suggests until a
  large observation arrives.

## References

- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [@feller1971introduction].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].
- Taleb, *Statistical Consequences of Fat Tails* [@taleb2020scoft].

## Backlinks

- Depends on: [Regular Variation](regular-variation.md) and
  [Pareto Moment Existence](pareto-moment-existence.md).
- Related to: [LLN Failure Under Infinite Mean](lln-failure.md).
- Used by: [Pre-Asymptotic LLN Behavior](../examples/lln-preasymptotic.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/theorems/generalized-central-limit-theorem.md`. Last verified: 2026-06-25. Checked against cited sources, scoped maximum-scaling argument, and executable stable-scaling simulation.
:::
<!-- incerto-provenance:end -->
