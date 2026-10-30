---
title: Chapter 10 — Diagnosing returns
---

Chapter 10 brings the earlier moment and tail tools to a return series.
Our [S&P diagnostics page](incerto-sp500.md) separates aggregation,
drawdowns, moment concentration, records, and gain/loss comparisons.
The [statistical-pipeline page](incerto-ml-econometrics.md) connects those
questions to risk estimation and function extrapolation.

The source is SCoFT, third edition (2025), arXiv:2001.10488v4:
chapter 10, printed pp. 193–205; Appendix F, pp. 207–212; and
Appendix G, pp. 213–216 [@taleb2025scoftv4]. Locators below refer to
that version. The notes supply original arguments and code. They do not
reproduce the source's historical S&P study.

## Reading map

| V4 locator, printed pages | Question | Treatment |
| --- | --- | --- |
| §10.1, pp. 193–195 | Can unstable moments identify a power tail? | [MS plot and an oscillating-tail counterexample](incerto-sp500.md#sp500-ms); finite data cannot settle moment existence |
| §10.2.1, p. 195 | How should kurtosis change under aggregation? | [Exact iid fourth-moment identity](incerto-sp500.md#sp500-aggregation), sample block counts and overlap sensitivity |
| §10.2.2, pp. 197–198 | What does a drawdown measure? | [Ordered prices and fractional decline](incerto-sp500.md#sp500-drawdown), with an explicit sign convention |
| §10.2.3, pp. 198–199 | What does empirical kappa compare? | [Sample aggregation](incerto-sp500.md#sp500-aggregation) and the canonical [kappa definition](incerto-kappa.md) |
| §10.2.4, p. 200 | Does a conditional tail mean stay proportional to its threshold? | [Correctly normalized ratios](incerto-sp500.md#sp500-drawdown); separate positive and negative return tails |
| §§10.2.5–10.2.6, pp. 200–201 | How much does one observation contribute to a high moment? | [Raw-power concentration](incerto-sp500.md#sp500-ms), distinguished from central kurtosis |
| §10.2.7, pp. 201–204 | What is the iid record-count benchmark? | [Insertion-rank proof](incerto-sp500.md#sp500-records); continuity, ties, dependence, and the limits of a nonrejection |
| §10.2.8 and §10.3, p. 205 | How should tail asymmetry and the overall evidence be interpreted? | [Two-sided diagnostics and dated local runs](incerto-sp500.md#sp500-local-run), without a universal tail-law verdict |
| Appendix F opening and §F.1, pp. 207–210 | Which moments control a risk estimator's variation? | [Raw second-moment variance](incerto-ml-econometrics.md#pipeline-moments) and [contamination figure](incerto-ml-econometrics.md#pipeline-contamination-figure) |
| §F.2, pp. 210–212 | What can an empirical fit learn about unobserved losses? | [CDF-versus-mean counterexample](incerto-ml-econometrics.md#pipeline-unseen) and [frozen holdout comparison](incerto-ml-econometrics.md#pipeline-holdout) |
| Appendix G, pp. 213–215 | How do bounded bases and hinges extrapolate? | [Finite-basis bounds and exact piecewise-linear representation](incerto-ml-econometrics.md#pipeline-hinges) |
| §G.0.1, pp. 215–216 | How do angles determine hinge weights? | [Tangent slopes, slope differences, and axis units](incerto-ml-econometrics.md#pipeline-angles) |

## Quantities and source qualifications

We keep $\alpha$ for the positive tail exponent, $u$ for a threshold,
and $e(u)$ for mean excess. The source's threshold $K$ in chapter 10
becomes $u$; $K_j$ remains a local hinge knot in the Appendix G treatment.
For $u>0$ and a finite conditional mean,

$$
\begin{gathered}
\phi(u)=\frac{\mathbb E[X\mid X>u]}u,\\
\phi(u)=1+\frac{e(u)}u.
\end{gathered}
$$

For an exact Pareto tail with $\alpha>1$, this is
$\alpha/(\alpha-1)$; for a unit exponential it tends to one.
The ratio is dimensionless. It cannot have limit zero or require a
second division by $u$.

The source's empirical interpretations require care. Moment instability
does not prove an infinite moment, and an infinite moment does not imply
regular variation. The kurtosis aggregation identity needs iid observations
and a finite fourth moment. The harmonic record benchmark needs iid
continuous sampling; agreement with it proves neither independence nor
extreme-value attraction. A finite sum of hinges represents a continuous
piecewise-linear function; arbitrary nonlinear functions need a separate
approximation statement. Hinge weights are slope differences, with slopes
given by tangents of angles under specified axis units.

## A small diagnostic

Use [`moment_share_path`](incerto-api-reference.md#api-moment-share-path).

```python
from xmath.incerto.diagnostics import moment_share_path

for sample in ([1, 1, 1, 1], [1, 1, 1, 5]):
    print(f"Largest fourth-power share: {moment_share_path(sample)[-1]:.4f}")
```

**Output**

```text
Largest fourth-power share: 0.2500
Largest fourth-power share: 0.9952
```

The larger share concerns four invented observations. Both lists could
come from bounded distributions with every moment finite. The numerical
contrast is not a test of infinite moments.

See [aggregation sampling ranges](incerto-sp500.md#sp500-aggregation-figure)
and the [risk-estimator stress experiment](incerto-ml-econometrics.md#pipeline-contamination-figure).
Regenerate with `uv run python demos/incerto/sp500_diagnostics.py`.
The empirical branch reuses the [existing local S&P cache](incerto-sp500-tail.md)
without fetching data. Return to the [SCoFT overview](incerto-scoft-intro.md)
for the reading route.
