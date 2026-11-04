---
title: Chapter 8 — How much averaging helps
---

An asymptotic limit does not tell us the error at a particular sample size.
This chapter compares that error through
[kappa](incerto-kappa.md#kappa-definition), the growth of mean absolute
deviation under summation. The
[sample-sufficiency calculation](incerto-moment-estimation.md#moment-estimation-sufficiency)
then asks how many observations match a stated Gaussian reference.

This guide uses @taleb2025scoftv4, third edition (2025),
arXiv:2001.10488v4, chapter 8, printed pp. 153–169. Section 8.6 is the
chapter's derivation appendix. It is distinct from the book's lettered
appendices. The attached earlier paper is @taleb2018kappa,
arXiv:1802.05495v3; book locators below always refer to v4.

## Reading map

| V4 locator, printed pages | Question | Canonical treatment |
| --- | --- | --- |
| §8.1, pp. 154–156 | What does finite-sample concentration measure? | [Population absolute error and its domain](incerto-kappa.md#kappa-definition) |
| §8.2, pp. 156–158 | How are two aggregation sizes compared? | [Two-index kappa and telescoping growth](incerto-kappa.md#kappa-definition) |
| §8.3, pp. 158–160 | What is the stable benchmark? | [Exact stable scaling and qualified attraction](incerto-kappa.md#kappa-stable); reuse the [stable-law note](incerto-generalized-central-limit-theorem.md) |
| §8.3.2, pp. 159–160 | How many observations match a Gaussian reference? | [Equal-single-MAD comparison and integer search](incerto-moment-estimation.md#moment-estimation-sufficiency) |
| §8.4.1, p. 161 | Does a shared exponent imply equal finite-sample behavior? | [Stable skewness versus distribution-body effects](incerto-kappa.md#kappa-stable), [Pareto/Student numerical comparison](incerto-kappa.md#kappa-shape-figure) |
| §8.4.2, p. 161 | What does the Student logarithmic rate establish? | [Anchored kappa versus distributional convergence](incerto-kappa.md#kappa-finite-variance) |
| §8.4.3, pp. 161–162 | How does lognormal scale affect concentration? | [Fixed-size and fixed-law limits](incerto-kappa.md#kappa-lognormal) |
| §8.4.4, p. 162 | Can kappa be negative? | [Unimodal counterexample and normal mixture](incerto-kappa.md#kappa-negative) |
| §§8.5.1–8.5.3, pp. 162–164 | What transfers to diversification and inference? | [Independent equal-weight exposures and dependence boundary](incerto-moment-estimation.md#moment-estimation-portfolio); [multivariate limitations](incerto-multivariate-tails.md) |
| §8.6.1, pp. 164–166 | Can cubic Student sums be evaluated exactly? | [Characteristic-function integral, finite sum, and corrected constant](incerto-kappa.md#kappa-student) |
| §8.6.2, pp. 166–168 | What can be proved for lognormal sums? | [Three limits and bounded conditional Monte Carlo](incerto-kappa.md#kappa-lognormal); moment matching remains an approximation |
| §8.6.3, p. 168 | What is the exponential/Gamma benchmark? | [Gamma integration-by-parts formula](incerto-kappa.md#kappa-gamma) |
| §8.6.4, p. 169 | How do separated means produce negative kappa? | [Exact binomial mixture of absolute-normal moments](incerto-kappa.md#kappa-negative) |

## Qualifications and notation

We write $M(n)=\mathbb E|S_n-n\mu|$, require $\mathbb E|X|<\infty$,
and retain both indices in

$$
\kappa_{n_0,n}=2-\frac{\log(n/n_0)}{\log(M(n)/M(n_0))}.
$$

The source's adjacent-size
shorthand $\kappa_1$ is $\kappa_{1,2}$. Student degrees of freedom are
$\nu$; $\alpha$ names the stable index or a stated power-tail exponent.
The [shared notation](https://xshi19.github.io/math/notation/) fixes sums
and means. These distinctions prevent several incorrect inferences:

- Stable sum growth $n^{1/\alpha}$ gives $\kappa=2-\alpha$.
  The finite-first-moment endpoint excludes index one.
- Kappa need not lie in $[0,1]$, even for a uniquely unimodal density.
  The triangular example proves this without a simulation.
- For cubic Student sums, our derivation gives
  $\log(n)\kappa_{1,n}\to2\log(\pi/2)$, correcting the printed
  constant in §8.6.1, p. 165. This is not a CLT error bound.
- Matching absolute mean error does not match variance or confidence
  coverage. At Gaussian reference size 30, the exact cubic Student
  calculation gives size 44 under equal single-observation MADs.
  The value 120 in §8.3.2, p. 160 is not the minimum for this criterion.
- Replacing a lognormal sum by a moment-matched lognormal is a heuristic.
  Our limits use direct expectation arguments; numerical estimates use
  the true sum model.
- The portfolio calculation assumes independent, identically distributed,
  equally weighted exposures. It supplies no empirical stock kappas,
  optimal-allocation theorem, or general confidence-interval cutoff.

## A worked comparison

Use [`gaussian_equivalent_n`](incerto-api-reference.md#api-gaussian-equivalent-n),
[`student_sum_mad`](incerto-api-reference.md#api-student-sum-mad), and
[`gamma_sum_mad`](incerto-api-reference.md#api-gamma-sum-mad).

```python
from xmath.incerto.concentration import gaussian_equivalent_n, student_sum_mad, gamma_sum_mad

for name, mad in (("Student 3", student_sum_mad), ("Exponential", gamma_sum_mad)):
    n = gaussian_equivalent_n(mad, 30)
    print(f"{name}: n={n}, relative absolute error={mad(n)/(n*mad(1)):.4f}")
```

**Output**

```text
Student 3: n=44, relative absolute error=0.1817
Exponential: n=36, relative absolute error=0.1803
```

The Gaussian target is $1/\sqrt{30}$. The notes prove monotonicity of
average MAD, which makes a deterministic integer search possible.
See the [aggregation curves](incerto-kappa.md#kappa-aggregation-figure),
[lognormal calculation](incerto-kappa.md#kappa-lognormal-figure),
[shape comparisons](incerto-kappa.md#kappa-shape-figure), and
[required sample sizes](incerto-moment-estimation.md#moment-estimation-sufficiency).
Regenerate all four with `uv run python demos/incerto/concentration.py`.
Return to the [SCoFT overview](incerto-scoft-intro.md) for the chapter route.
