---
title: Chapter 4 — Finite-moment fat tails
---

Chapter 4 asks how a distribution can depart sharply from a Gaussian model
while retaining finite moments. Its first mechanism is to randomize a
Gaussian's variance. Its second question is statistical: which moments
must exist for a chosen estimate to have a finite error variance?

We use SCoFT's 2025 third edition, arXiv:2001.10488v4, ch. 4,
printed pp. 69–89. The chapter's broad use of “fat tails” includes comparisons
with a Gaussian at fixed variance. That usage does not assert
[regular variation](incerto-regular-variation.md) or
[subexponentiality](incerto-subexponentiality.md).

## A reading path through the chapter

| V4 locator, printed pages | Mathematical question | Our treatment |
| --- | --- | --- |
| §§4.1–4.1.2, pp. 69–74 | Can variance stay fixed while higher moments increase? | [Variance mixtures](incerto-mixture-uncertainty.md): conditioning, finite-state examples, kurtosis, and Gaussian-relative tails. |
| §4.2, pp. 74–75 | Does mixing variance or its reciprocal give the same tail? | [Variance and precision](incerto-mixture-uncertainty.md#mixture-variance-precision): gamma, lognormal, and Student models with proved moment conditions. |
| §§4.3–4.3.1, pp. 75–78 | Where does a perturbation add or remove density? | [Body, shoulders, and tails](incerto-body-shoulder-tail.md): local density curvature, survival curvature, and exact finite-mixture crossings. |
| §§4.4–4.4.3, pp. 79–84 | What differs between quadratic and absolute deviations? | [Moment estimation](incerto-moment-estimation.md) and [dispersion ratios](incerto-dispersion-ratio.md): estimands, estimator moments, Gaussian calibration, and repeated samples. |
| §4.4.4, pp. 84–86 | Why do higher powers emphasize large observations? | [Power means](incerto-moment-estimation.md#moment-estimation-power-means): Jensen proof, equality, and the maximum limit. |
| §4.4.5, pp. 87–88 | Does a dispersion comparison justify a general recommendation? | [Different estimands](incerto-moment-estimation.md#mean-absolute-deviation-is-a-different-estimand): state the target and loss before comparing methods. |
| §4.5, pp. 88–89 | Why do geometric unit balls expand with norm order? | [Norm geometry](incerto-moment-estimation.md#moment-estimation-power-means): distinguish sums of powers from averages of powers. |

## Start with a conditional model

Let $X=\sqrt YZ$, with $Z\sim N(0,1)$ independent of positive variance $Y$.
The shared mixing symbol is $Y$; see the
[notation canon](https://xshi19.github.io/math/notation/) and
[Normix foundation](https://xshi19.github.io/math/normix-theory/normix-normal-mixtures/).
With $\mathbb EY=1$ and finite $\mathbb EY^2$,

$$
\mathbb EX=0,\qquad \operatorname{Var}(X)=1,\qquad
\mathbb EX^4=3\mathbb EY^2.
$$

The [mixture note](incerto-mixture-uncertainty.md#mixture-moment-transfer)
proves these identities. Equal probabilities on variances 0.4 and 1.6 give
fourth moment 4.08; probabilities 0.99 and 0.01 on variances 0.5 and 50.5
give 77.25. Both mixtures have a finite MGF everywhere. A sharper peak and
larger far-tail probabilities relative to a unit normal do not make either
mixture a power law.

## Specify what is perturbed

The source's Gaussian boundaries in §4.3.1 concern differentiation with
respect to **scale**. Our body/shoulder note also computes **variance**
curvature. For a unit normal the positive density boundaries are

$$
\begin{aligned}
\text{variance perturbation:}&\quad \sqrt{3\pm\sqrt6},\\
\text{scale perturbation:}&\quad \sqrt{(5\pm\sqrt{17})/2}.
\end{aligned}
$$

Neither pair is a threshold for fitting a Pareto tail. A small-perturbation
expansion is pointwise, and its zero-curvature points need not equal the
crossings of a finite mixture. Tail probabilities have their own curvature:
convexity in variance holds only above a threshold relative to that variance,
not for every tail event and every mixing interval.

## Separate a moment from its estimator

For iid Pareto observations, the [moment note](incerto-moment-estimation.md#moment-estimation-gates)
proves

$$
\mathbb E X^q<\infty\iff q<\alpha,\qquad
\operatorname{Var}(\widehat m_{q,n})<\infty\iff 2q<\alpha
\quad\text{when }q<\alpha.
$$

At exponent 3 the second moment exists, its sample average is unbiased and
consistent, and that average has infinite variance. Its **square root** has
a finite second moment. This last distinction matters when reading the
standard-deviation comparison around §4.4.3, p. 83: conclusions about an
average of squares do not transfer unchanged to its square root.

Even with every moment finite, a rare high-variance component can be absent
from most short samples. The repeated-sample demo shows both mechanisms.
Neither mechanism implies that every sample mean is downward biased.
The [plug-in singularity](incerto-moment-estimation.md#moment-estimation-plugin)
adds a third issue: fitting a model may create divergent estimator moments
near a parameter boundary.

## Notation and source qualifications

| Source usage | Convention used here |
| --- | --- |
| Normal scale parameter | $N(\mu,v)$ uses variance as its second argument; write $v=\sigma^2$ explicitly. |
| Stochastic variance or reciprocal variance | $Y$ is positive variance; local $Q=1/Y$ is precision. |
| Student tail parameter | $\nu$ is degrees of freedom; the power-tail exponent is $\alpha=\nu$. Unit Student scale is not unit standard deviation. |
| Mean deviation / MAD | State the center and whether the deviation is a mean or a median; our dispersion demo applies Gaussian calibration. |
| Rising norms | $A_p$ averages powers; geometric $\ell_p$ norms sum them. Their monotonicities differ. |

Zero mean does not identify an MGF with a characteristic function; the
[finite-mixture formulas](incerto-mixture-uncertainty.md#mixture-finite-states)
show their opposite signs in the exponent. Missing a power moment also does
not by itself establish a power law. The explicit mixture and Pareto
calculations provide narrower, checkable statements than either shortcut.

The chapter's efficiency discussion is conditional on a model and an
estimand. A Gaussian calibration does not make mean absolute deviation an
unbiased estimator of standard deviation under every other law. We do not
adopt a universal preference for one dispersion statistic. The bounded
Cauchy experiment is reserved for chapter 7; recursive mixing for chapter 20.

## Worked cases

Use [`normal_mixture_absolute_moment`](incerto-api-reference.md#api-normal-mixture-absolute-moment), [`power_mean`](incerto-api-reference.md#api-power-mean).

```python
from xmath.incerto.finite_moments import normal_mixture_absolute_moment, power_mean

variances, weights = [0.5, 50.5], [0.99, 0.01]
for order in (2, 4):
    print(f"{order:d} {normal_mixture_absolute_moment(order, variances, weights):.4g}")
means = [power_mean([0, 0, 0, 4], p) for p in (1, 2, 4)]
print("Power means: [" + ", ".join(f"{v:.4g}" for v in means) + "]")
```

**Output**

```text
2 1
4 77.25
Power means: [1, 2, 2.828]
```

The finite mixture and the fixed vector answer different moment and norm questions.

Figure regeneration: `uv run python demos/incerto/finite_moments.py` for the deterministic
mixture, curvature and norm figures and the seeded moment experiment.
Figure regeneration: `uv run python demos/incerto/dispersion_ratio.py` for the existing
Gaussian, Student, and Pareto comparison. Model curves and simulations are
identified separately in their concept-page captions; no external data are
needed. The scalar variance–mean gamma derivation is in the [mixture note](incerto-mixture-uncertainty.md#mixture-variance-gamma).

## References

- @taleb2025scoftv4, *Statistical Consequences of Fat Tails*, third edition,
  2025, arXiv:2001.10488v4, ch. 4, §§4.1–4.5, printed pp. 69–89.
  The table gives the section locators used. Our examples, proofs and figures
  are developed independently; the concept pages identify cited inputs.
