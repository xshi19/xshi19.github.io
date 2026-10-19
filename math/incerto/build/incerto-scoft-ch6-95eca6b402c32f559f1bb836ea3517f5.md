---
title: Chapter 6 — Thick tails in higher dimensions
---

A tail exponent describes one coordinate. It does not tell us whether large
coordinates arrive together, whether a covariance exists, or what a matrix
of observations can estimate. Chapter 6 adds those questions. Begin with
[density geometry](incerto-iso-density-tail-geometry.md), then use the
[multivariate Student construction](incerto-multivariate-tails.md) to separate
shared scale, independence, and correlation.

This guide uses @taleb2025scoftv4, third edition (2025),
arXiv:2001.10488v4, chapter 6, printed pp. 107–120, and attached
appendix A, pp. 121–125. The substantive numbered sections end on
p. 119; p. 120 is the chapter transition. The notes supply original
arguments and examples, with proved and cited results identified locally.

## Reading map

| V4 locator, printed pages | Question | Concept page |
| --- | --- | --- |
| §6.1, p. 108 | What changes when a vector shares a random variance? | [Elliptical variance mixture](incerto-multivariate-tails.md#multivariate-elliptical); reuses the [chapter-4 mixing mechanism](incerto-mixture-uncertainty.md) |
| §6.2, pp. 109–112 | Which linear maps preserve ellipticality, and what can break it? | [Linear-map proof](incerto-multivariate-tails.md#multivariate-elliptical), [switching correlation](incerto-iso-density-tail-geometry.md#tail-geometry-switching) |
| §6.3 and §6.3.1, pp. 113–114 | What is the normalized Student density, and why are its coordinates dependent? | [Student derivation](incerto-multivariate-tails.md#multivariate-student), [same Cauchy marginals](incerto-iso-density-tail-geometry.md#tail-geometry-elliptical) |
| §6.4, pp. 114–115 | Can information detect dependence when correlation vanishes or is undefined? | [Entropy and mutual information](incerto-multivariate-tails.md#multivariate-information) |
| §6.5, pp. 115–116 | Which random matrices have a semicircle limit? | [Ensemble, normalization, and finite spectral bounds](incerto-multivariate-tails.md#multivariate-spectra) |
| §6.6, pp. 116–117 | What does a bounded sample coefficient mean without a population variance? | [Sample correlation and a fixed-sample contamination limit](incerto-multivariate-tails.md#multivariate-sample-correlation) |
| §6.7, pp. 117–119 | What population quantity can fitted regression explain? | [Finite-variance regression target and consistency](incerto-multivariate-tails.md#multivariate-regression) |
| §A.1, pp. 122–124 | How can separated regimes have kurtosis below three? | [Normal location-mixture proof](incerto-tail-class-catalog.md#tail-catalog-regimes), [rare Pareto regime](incerto-pareto.md#pareto-regime-mixture) |
| §A.2, pp. 124–125 | When is an absorbing state eventually reached? | [Two-state transition calculation and decreasing-hazard counterexample](incerto-tail-class-catalog.md#tail-catalog-absorbing) |

## Three meanings of zero

For a bivariate Student with degrees of freedom $\nu>0$ and scatter
$\Sigma=\left(\begin{smallmatrix}1&\rho\\\rho&1\end{smallmatrix}\right)$,
setting $\rho=0$ means the scatter axes are orthogonal. If $\nu>2$,
Pearson correlation also equals zero. Independence still fails for every
finite $\nu$, because the two coordinates share the scale.

At $\nu=1$ this is the spherical Cauchy. Its Pearson coefficient is
undefined but its mutual information is finite:

$$
I(X_1;X_2)=\log(8\pi)-3\approx0.224171\quad(\rho=0).
$$

The [entropy calculation](incerto-multivariate-tails.md#multivariate-information)
proves the value. Independent Cauchy coordinates instead have mutual
information zero and nonelliptical contours. Thus a marginal family and
one matrix parameter do not specify the dependence model.

## Qualifications to carry into the source

- A Student scatter matrix is $\Sigma$, while covariance, when finite,
  is $\nu\Sigma/(\nu-2)$. The density needs its normalizing constant.
  The [mixture proof](incerto-multivariate-tails.md#multivariate-student)
  keeps both distinctions explicit.
- Stable vectors need not be elliptical. The
  [independent Cauchy counterexample](incerto-iso-density-tail-geometry.md#tail-geometry-switching)
  has stable linear projections and nonelliptical level sets. A singular
  law need not have a density, but stability alone does not imply its absence.
- A semicircle statement needs a matrix ensemble and scaling. A finite
  fourth moment is not necessary for the cited empirical spectral limit;
  extreme eigenvalues and sample covariance spectra are separate questions.
- The uncentered cosine in §6.6 differs from centered sample Pearson
  correlation. Neither finite statistic creates a missing population moment.
- For $V=b+aX+\varepsilon$ with independent, centered finite-variance
  error and standard normal $X$, the population target is
  $a^2/(a^2+\operatorname{Var}\varepsilon)$. The
  [proof](incerto-multivariate-tails.md#multivariate-regression)
  gives consistency but no universal upward-bias or $O(n^{-2})$ rate.
  An undefined infinite-variance target must not be assigned zero by a ratio
  of infinities. The source's historical regressions are not reproduced.
- An absorbing state is eventually reached in our homogeneous model only
  when its conditional failure probability is positive. Irreversibility
  without persistent risk is insufficient, as the appendix example shows.

These qualifications bound the teaching claims. They do not assess portfolio
performance, historical data quality, or the source's broad prescriptions.
Notation follows the [shared canon](https://xshi19.github.io/math/notation/):
$\alpha$ denotes a survival exponent, while $\nu$ denotes Student degrees
of freedom (equal to each marginal's exponent here). $Y$ is mixing variance,
$G=1/Y$ precision, $V$ a regression response, and $T$ a transition matrix.
The appendix's state probabilities must not be read as point densities.

(scoft-ch6-demo)=
## Worked cases

Run `uv run python demos/incerto/multivariate_tails.py`. Five single-column
figures use exact density/entropy formulas, checked one-dimensional
quadrature, and bounded seeded simulations:

| Calculation | Figure and scope |
| --- | --- |
| Independent versus spherical Cauchy; switching normal correlation | [Density contours](incerto-iso-density-tail-geometry.md#tail-geometry-switching), 241-by-241 grid on $[-4,4]^2$ |
| Student information and joint absolute exceedances | [Information](incerto-multivariate-tails.md#multivariate-information), $\nu\in[0.75,30]$, and gamma-precision quadrature at $\nu=3$ |
| Symmetric matrix spectra | [Spectra](incerto-multivariate-tails.md#multivariate-spectra), twelve matrices of size 80 per law |
| Sample correlation and fitted regression $R^2$ | [Sampling](incerto-multivariate-tails.md#multivariate-regression), 400 replications at four sizes through 1000 |
| Separated regimes and conditional failure | [Appendix A examples](incerto-tail-class-catalog.md#tail-catalog-regimes), exact normal-mixture formulas and 100 transition steps |

The sampling experiments use PCG64 seed 20261018, without clipping or
discarding draws. Printed quantiles describe the simulated sampling laws;
they do not certify an asymptotic theorem. The runner also checks a Cauchy
sector probability, a finite spectral identity, and the rare-Pareto example.
There is no external dataset or fitted tail model.

Continue to [moment estimation](incerto-moment-estimation.md) for the
distinction between an existing moment and a stable estimator, or return to
the [SCoFT overview](incerto-scoft-intro.md).
