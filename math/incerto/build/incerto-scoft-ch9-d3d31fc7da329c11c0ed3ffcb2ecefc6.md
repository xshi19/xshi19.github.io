---
title: Chapter 9 — Maxima and hidden tails
---

A maximum records the largest observed value. A tail model describes what
may exceed it and how much those outcomes contribute to a payoff. This
chapter connects the [GEV family](incerto-generalized-extreme-value.md),
[threshold excesses](incerto-pickands-balkema-de-haan.md), and
[hidden moments](incerto-hidden-tail.md). Appendix E supplies the
[exact Pareto exponent calibration](incerto-hill-estimator.md#hill-sampling-law).

The source is SCoFT, third edition (2025), arXiv:2001.10488v4,
chapter 9, printed pp. 171–181, and Appendix E, pp. 189–191
[@taleb2025scoftv4]. All locators below use that edition. The concept
notes give original derivations and distinguish exact-model identities,
cited general theorems, and numerical comparisons.

## Reading map

| V4 locator, printed pages | Question | Treatment |
| --- | --- | --- |
| §9.1, pp. 171–173 | What is normalized, and which limit is assumed? | [GEV definition](incerto-generalized-extreme-value.md) and [EVI convention](incerto-extreme-value-index.md#evi-chapter-nine); endpoint approach alone is insufficient |
| §9.1.1, pp. 173–174 | Why does a power tail give Fréchet maxima? | [Regular-variation argument](incerto-frechet.md#frechet-regular-variation); reuse the [exact Pareto maximum](incerto-frechet.md#frechet-pareto-maxima-check) |
| §9.1.2, pp. 174–176 | How do normal maxima approach Gumbel? | [Exact CDF, normalizing constants, and proof](incerto-generalized-extreme-value.md#gev-gaussian-maxima) |
| §9.1.3, pp. 176–177 | What justifies a high-threshold GPD? | [PBDH assumptions and normal special case](incerto-pickands-balkema-de-haan.md#pbdh-chapter-nine), [GPD moment integrals](incerto-generalized-pareto.md#gpd-moment-integrals) |
| §9.2, pp. 177–179 | How much of a moment lies beyond the sample maximum? | [Exact missing mass](incerto-hidden-tail.md#hidden-tail-missing-mass) and [hidden Pareto moments](incerto-hidden-tail.md#hidden-tail-pareto) |
| §9.2.1, pp. 179–180 | How does the normal parent compare? | [Normal hidden-mean integral](incerto-hidden-tail.md#hidden-tail-gaussian), expressed in standard-deviation units |
| §9.3, pp. 180–181 | Does accurate tail probability imply accurate payoff? | [Empirical CDF and contamination counterexample](incerto-hidden-tail.md#hidden-tail-empirical); [mean excess versus unconditional contribution](incerto-mean-excess-function.md#mean-excess-hidden-tail) |
| Appendix E, pp. 189–190 | What assumptions make parameter-based extrapolation possible? | [Plug-in body/tail decomposition](incerto-plug-in-tail-estimation.md#plugin-tail-body) and [threshold selection](incerto-tail-threshold-selection.md#threshold-chapter-nine) |
| E.1, p. 191 | What is the sample-exponent law? | [Gamma-to-inverse-gamma proof](incerto-hill-estimator.md#hill-sampling-law), [debiasing and truncation](incerto-plug-in-tail-estimation.md#plugin-exponent-truncation) |

## Notation and qualifications

Use $x_m$ for the Pareto minimum, $\alpha$ for its positive exponent,
$\xi$ for EVI, and $M_n$ for the iid sample maximum. The source's minimum
$L$ becomes $x_m$; $L(x)$ retains its slowly varying meaning. The
[shared canon](https://xshi19.github.io/math/notation/) distinguishes
$\bar F=1-F$, threshold $u$, and mean excess $e(u)$.

A compact comparison summarizes the exact Pareto calculation. For
$\alpha>1$, let $r=1-1/\alpha$, $\mu=\mathbb E[X]$, and
$T_1(t)=\mathbb E[X\mathbf1_{\{X>t\}}]$. Then

$$
\begin{gathered}
\mathbb E[\bar F(M_n)]=\frac1{n+1},\\
\frac{\mathbb E[T_1(M_n)]}{\mu}\\
=\frac{\Gamma(n+1)\Gamma(1+r)}{\Gamma(n+1+r)}.
\end{gathered}
$$

The first quantity decreases at order $1/n$; the second at order
$n^{-r}$. Both are population calculations averaged over repeated samples.
Neither is the bias of the empirical sample mean.

Several distinctions matter when reading the source:

- A slowly varying factor need not settle to a constant. Fréchet attraction
  uses tail ratios and a suitable normalization.
- The Gumbel CDF for maxima is $\exp(-e^{-z})$; no minima convention is
  silently substituted. Finite-block accuracy is checked against $\Phi^n$.
- PBDH needs the domain-of-attraction assumption. Its uniform CDF limit
  does not by itself imply convergence of means.
- A hidden $p$th Pareto moment is finite only for $p<\alpha$.
  The exact missing probability is Beta(1,n); the exponential law is the
  limit after multiplying that probability by $n$.
- An empirical distribution is a legitimate probability measure. CDF
  accuracy alone does not control unbounded-payoff accuracy. A continuous
  population is not absolutely continuous with respect to the finite
  atomic empirical measure, so a population/empirical density ratio
  cannot reconstruct the missing tail.
- The inverse-gamma law needs exact Pareto sampling and a specified fixed
  cutoff or fixed retained count. Debiasing keeps the shape parameter;
  conditioning or clipping changes the sampling experiment.

## A finite-sample comparison

Use [`expected_pareto_hidden_fraction`](incerto-api-reference.md#api-expected-pareto-hidden-fraction).

```python
from xmath.incerto.extremes import expected_pareto_hidden_fraction

for n in (100, 1000, 10000):
    mass = expected_pareto_hidden_fraction(n, 1.5, p=0)
    mean = expected_pareto_hidden_fraction(n, 1.5)
    print(f"n={n}: missing probability={mass:.4g}, hidden mean fraction={mean:.4g}")
```

**Output**

```text
n=100: missing probability=0.009901, hidden mean fraction=0.192
n=1000: missing probability=0.000999, hidden mean fraction=0.08928
n=10000: missing probability=9.999e-05, hidden mean fraction=0.04145
```

See the [hidden-tail curves](incerto-hidden-tail.md#hidden-tail-figure),
[Gaussian maxima](incerto-generalized-extreme-value.md#gev-gaussian-maxima),
and [Hill sampling comparison](incerto-hill-estimator.md#hill-sampling-figure).
Regenerate with `uv run python demos/incerto/extreme_values.py`.
Return to the [SCoFT overview](incerto-scoft-intro.md) for the reading route.
