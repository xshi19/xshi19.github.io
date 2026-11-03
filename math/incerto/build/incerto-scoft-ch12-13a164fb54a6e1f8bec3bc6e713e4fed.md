---
title: Chapter 12 — Point forecasts and tail risk
---

A forecast of an amount needs a target, a predictive distribution, and a
decision context. A single mean or median cannot supply all three. The
[point-forecast note](incerto-point-forecast.md) develops this distinction
through stable scaling, repeated evaluation of forecast losses, quantiles,
and synthetic bounded losses.

The source is SCoFT, third edition (2025), arXiv:2001.10488v4,
chapter 12, printed pp. 241–246 [@taleb2025scoftv4]. The chapter is
coauthored with Yaneer Bar-Yam and Pasquale Cirillo. The attached paper is
arXiv:2007.16096v1 [@taleb2020pointforecasts]; its main statements and
commentary occupy pp. 1–3, technical discussion pp. 3–5, and further
pandemic discussion pp. 5–8. These are separate locators, not an assertion
that the paper and book contain identical claims.

## Reading map

| V4 locator, printed pages | Question | Treatment |
| --- | --- | --- |
| §12.1, p. 241 | What does a single point omit? | [Target and loss](incerto-point-forecast.md#point-forecast-targets), [predictive quantiles and severity](incerto-point-forecast.md#point-forecast-quantiles) |
| §12.2, pp. 241–243 | Why can averages be poor descriptions of a skewed outcome? | [Estimation versus prediction](incerto-point-forecast.md#point-forecast-scaling); [historical fatalities and data limits](incerto-point-forecast.md#point-forecast-pandemics) |
| §12.3, pp. 243–245 | How do tail exponents affect averaging and evidence? | [Exact stable scales and cited Pareto limits](incerto-point-forecast.md#point-forecast-scaling); [score moments and paired comparisons](incerto-point-forecast.md#point-forecast-evaluation) |
| §12.3, p. 244 | What information does a survival probability carry? | [Survival integral versus one threshold](incerto-point-forecast.md#point-forecast-quantiles), [event/payoff distinction](incerto-calibration-payoff.md) |
| §12.3, p. 245; §12.4, pp. 245–246 | How do actions change the forecast being tested? | [Conditional action regimes and an explicit precaution model](incerto-point-forecast.md#point-forecast-pandemics) |

The [population cap and dual transform](incerto-point-forecast.md#point-forecast-bounds)
clarify the bounded-variable issue behind the pandemic application. This
section uses the underlying study [@cirillo2020pandemics, §II, p. 4]; it is
not attributed to a formula in chapter 12. All examples here are synthetic.
The concept note identifies the third-party historical compilation without
copying its data or treating its fitted tail as a universal law.

## A rate with a precise meaning

For iid symmetric stable errors with index $0<\alpha\le2$ and
unit characteristic-function scale, the sample average has scale

$$
s_n=n^{1/\alpha-1}.
$$

This is an exact distributional identity, proved in the note. It describes
an interquantile-width multiplier even when a mean absolute error does not
exist. For positive Pareto samples with $1<\alpha<2$, the analogous
mean-error power is asymptotic; a general regularly varying law may add
a slowly varying factor. A future single outcome retains its own dispersion.

Use [`stable_average_scale`](incerto-api-reference.md#api-stable-average-scale).
The calculation compares averaging **within** each law, not risk across
laws with a common variance, which need not exist.

```python
from xmath.incerto.forecasting import stable_average_scale

for alpha in (.8, 1, 1.5, 2):
    ratio = stable_average_scale(alpha, 100) / stable_average_scale(alpha, 10)
    print(f"alpha={alpha:g}: tenfold-sample scale ratio={ratio:.3f}")
```

**Output**

```text
alpha=0.8: tenfold-sample scale ratio=1.778
alpha=1: tenfold-sample scale ratio=1.000
alpha=1.5: tenfold-sample scale ratio=0.464
alpha=2: tenfold-sample scale ratio=0.316
```

## Qualifications to carry through the chapter

The chapter's warning about an isolated number does not imply that every
point forecast violates probability theory. A median, quantile, or bounded
event probability is a defined object. Its usefulness depends on the loss
and decision. A strong law is also a convergence statement, not a universal
finite-sample convergence rate.

The note supplies precise replacements for several broad source claims:

- Exact Pareto means diverge at $\alpha\le1$; a fixed bounded fatality
  variable has all positive moments. Moment boundaries for general regular
  variation depend on the slowly varying factor.
- A fixed-threshold survival fraction has MSE $p(1-p)/n$. The complete
  survival curve determines expected excess; one probability does not.
- Infinite variance of an amount does not make every score or paired
  comparison unstable. Bounded scores and bounded paired differences
  retain finite-variance iid guarantees.
- Multiplication or exponential growth alone proves neither a power law
  nor chaos. Tail extrapolation, threshold choice, and dependence remain
  assumptions to assess; more information need not preserve a tail exponent.
- Precaution needs an exposure, action costs, and effectiveness assumptions.
  A changed action regime requires a conditional forecast and a causal
  comparison, not an unconditional accuracy verdict.

The three figures show [averaging scales](incerto-point-forecast.md#point-forecast-scaling-figure),
[score sampling variation](incerto-point-forecast.md#point-forecast-evaluation-figure),
and [endpoint/cap sensitivity](incerto-point-forecast.md#point-forecast-bounds-figure).
Generate them with `uv run python demos/incerto/point_forecasts.py`.
The [chapter 11 guide](incerto-scoft-ch11.md) supplies the calibration
background; the [SCoFT overview](incerto-scoft-intro.md) gives the chapter route.
