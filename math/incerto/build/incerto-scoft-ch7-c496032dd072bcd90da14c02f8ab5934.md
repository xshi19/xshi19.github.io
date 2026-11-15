---
title: Chapter 7 — Averages and limit laws
---

An average can converge while its fluctuations remain far from Gaussian.
It can also fail to estimate any finite expectation. This chapter separates
the target, the mode of convergence, and the scale of the error. Start with
[weak and strong laws](incerto-lln-failure.md#lln-modes), then compare
[normalized sums](incerto-generalized-central-limit-theorem.md) with the
[Cauchy average](incerto-cauchy-running.md).

This guide uses @taleb2025scoftv4, third edition (2025),
arXiv:2001.10488v4, chapter 7, printed pp. 137–152, with attached
appendices C (pp. 183–186) and D (pp. 187–188). The concept pages contain
original proofs at the stated scope and identify cited general theorems.

## Reading map

| V4 locator, printed pages | Question | Canonical treatment |
| --- | --- | --- |
| §7.1, pp. 137–139 | What do the weak and strong LLNs assert? | [Integrable weak-law proof; cited strong law](incerto-lln-failure.md#lln-modes), [nonnegative infinite mean](incerto-lln-failure.md) |
| §§7.2.1–7.2.2, pp. 139–141 | What stays stable under sums, and when do averages concentrate? | [Exact symmetric stable sums](incerto-generalized-central-limit-theorem.md#stable-exact-sums), [Cauchy counterexample](incerto-cauchy-running.md#cauchy-average-law) |
| §§7.3.1–7.3.4, pp. 141–145 | Which comparisons of convergence speed are justified? | [Moment conditions and CDF bounds](incerto-generalized-central-limit-theorem.md#clt-cumulants), [Pareto normalization experiment](incerto-generalized-central-limit-theorem.md#generalized-clt-scaling-simulation) |
| §7.4, pp. 145–147 | What do cumulants of standardized iid sums tell us? | [Cumulant scaling and a non-Gaussian zero-excess example](incerto-generalized-central-limit-theorem.md#clt-cumulants) |
| §7.5, pp. 147–148 | What hypotheses belong to CLT variants? | [Finite variance versus stable attraction](incerto-generalized-central-limit-theorem.md); general array theorems are cited context, not proved here |
| §§7.6–7.6.1, pp. 148–150 | Which higher moments can be estimated by averaging? | [Absolute-moment LLN](incerto-lln-failure.md#lln-higher-moments), [estimator variance](incerto-moment-estimation.md#moment-estimation-gates), [max-to-sum diagnostic](incerto-max-to-sum-ratio.md) |
| §7.7, pp. 151–152 | How does stable mean absolute deviation scale? | [Symmetric-case integral proof](incerto-generalized-central-limit-theorem.md#stable-mean-deviation) |
| §C.1, pp. 183–185 | Can thin-tailed rates create power-tailed outcomes? | [Exponential-to-Pareto transform](incerto-ergodicity.md#ergodicity-growth-transform), [finite rate uncertainty](incerto-mixture-uncertainty.md#mixture-growth-rate) |
| §C.2, pp. 185–186 | Does a transformation establish an empirical tail model? | [Assumption boundary](incerto-ergodicity.md#ergodicity-growth-transform); historical pandemic fits are not reproduced or certified |
| Appendix D, pp. 187–188 | What does an exponential-moment bound actually guarantee? | [Optimized Bernoulli Chernoff bound](incerto-cramer-condition.md#cramer-bernoulli) |

## Three boundaries to keep distinct

For exact Pareto with exponent $\alpha=1.5$, the mean is $3x_m$ and
the strong law holds. Variance is infinite; the centered sum has scale
$x_m n^{2/3}$, so the mean error has scale

$$
\bar X_n-3x_m=O_p\!\left(x_m n^{-1/3}\right).
$$

For exact Pareto at $\alpha=1$, the mean is infinite even though
[the largest share tends to zero in probability](incerto-pareto.md#sums-and-moment-orders).
For a Cauchy law, both signed expectation parts diverge and every sample
average has the same distribution as one observation.

The source writes the centering and scaling constants in an order different
from this site's $(S_n-b_n)/a_n$, $a_n>0$. We retain the site's convention.
$\alpha$ denotes a survival exponent or the explicitly named stable index;
the Gaussian stable index is two despite the absence of a power tail.
$\sigma$ in the symmetric stable characteristic function is a scale,
with Gaussian variance $2\sigma^2$. See [shared notation](https://xshi19.github.io/math/notation/).

## Qualifications to carry into the source

- A CLT requires a parent law in a domain of attraction. There is no
  unconditional claim that all iid sums admit a nondegenerate normalized limit.
- The exponent-two Pareto example has infinite variance. A Gaussian basin
  does not always imply square-root normalization. A central bell shape,
  a fourth-moment plot, and uniform CDF error are different diagnostics.
- Cumulants are evaluated at the origin and standardized by the matching
  power of variance. Skewness scales as $n^{-1/2}$ and excess kurtosis as
  $n^{-1}$ when those moments exist. Neither a zero fourth cumulant nor
  a finite empirical value identifies the parent law.
- Our §7.7 calculation covers symmetric stable laws and checks the
  Gaussian endpoint explicitly. We do not adopt the source's general
  skew-stable expression or its Fourier-transform constants without proof.
- Convergence of a sample mean and ergodicity are distinct questions.
  A [multiplicative growth example](incerto-ergodicity.md#ergodicity-growth)
  compares different observables; it does not invalidate the ergodic theorem.
- A right-tail specification cannot determine an entire two-sided rate
  density. Appendix C's transformation is proved under an exact model;
  it supplies no independent empirical evidence about epidemics.
- Appendix D's exponential expression is a bound or a logarithmic-rate
  statement, depending on hypotheses. It is not generally an exact
  probability. Independence of summands supplies MGF factorization.

## Worked values and figures

Use [`pareto_capped_mean`](incerto-api-reference.md#api-pareto-capped-mean)
and [`symmetric_stable_mean_abs`](incerto-api-reference.md#api-symmetric-stable-mean-abs).

```python
from xmath.incerto.limits import pareto_capped_mean, symmetric_stable_mean_abs

print(f"Capped Pareto(1) at 100: {pareto_capped_mean(100, 1):.4g}")
for n in (1, 100, 10000):
    print(f"Stable alpha=1.5, n={n}: {symmetric_stable_mean_abs(1.5, n=n):.4g}")
```

**Output**

```text
Capped Pareto(1) at 100: 5.605
Stable alpha=1.5, n=1: 1.705
Stable alpha=1.5, n=100: 0.3674
Stable alpha=1.5, n=10000: 0.07916
```

The capped Pareto calculation changes the observable; the stable calculation
uses an untruncated law. Bounded simulations and analytic curves are generated
by `uv run python demos/incerto/limit_distributions.py`:

- [Cauchy paths and repeated means/medians](incerto-cauchy-running.md#cauchy-running-figures).
- [Pareto running means, sampling spread and cap growth](incerto-lln-failure.md#lln-failure-sample-paths).
- [Correct versus square-root normalization](incerto-generalized-central-limit-theorem.md#generalized-clt-scaling-simulation).
- [Multiplicative paths and two population targets](incerto-ergodicity.md#ergodicity-growth).

The [max-to-sum experiment](incerto-max-to-sum-ratio.md#max-to-sum-paths)
already supplies repeated paths across the mean boundary. Reuse its specified
model reference; a bootstrap cannot manufacture an unobserved tail.
Return to the [SCoFT overview](incerto-scoft-intro.md) for the chapter route.
