---
title: Chapter 11 — Calibration and payoffs
---

Chapter 11 compares forecasts of events with forecasts of their consequences.
The [calibration and payoff page](incerto-calibration-payoff.md) develops
that distinction through exact models, conditional calibration, and score
sampling laws. The [point-forecast page](incerto-point-forecast.md) explains
which loss functions preserve or change an amount's tail behavior.

The source is SCoFT, third edition (2025), arXiv:2001.10488v4,
chapter 11, printed pp. 219–240 [@taleb2025scoftv4]. All locators below
refer to that version. The notes use original derivations, examples,
code, and figures. They do not reproduce the book's empirical or
psychological claims as established facts.

## Reading map

| V4 locator, printed pages | Question | Treatment |
| --- | --- | --- |
| §11.1 and §11.1.1, pp. 220–224 | What separates binary and continuous payoffs? | [Strict event, tail amount, and vanilla excess](incerto-calibration-payoff.md#calibration-contract); static replication scope |
| §11.1.2, pp. 224–225 | What does a verbal event leave undefined? | [Threshold, horizon, units, and severity](incerto-calibration-payoff.md#calibration-contract); fat tails do not invalidate specified events |
| §§11.2–11.2.2, pp. 225–226 | When does replacing severity by the threshold fail? | [Exact Pareto and exponential means](incerto-calibration-payoff.md#calibration-tail-models) and the existing [mean-excess corollary](incerto-pickands-balkema-de-haan.md#pbdh-mean-excess-corollary) |
| §11.2.3, pp. 227–230 | Can probability and payoff move oppositely? | [Fixed-mean lognormal calculation](incerto-calibration-payoff.md#calibration-lognormal); payoff-equivalent units are not event probabilities |
| §11.2.4, p. 230 | What does parameter uncertainty change? | [Specified Pareto mixture and convexity proof](incerto-calibration-payoff.md#calibration-uncertainty); conditional weights and integrability |
| §11.3, pp. 230–231 | What does calibration mean? | [Conditional-frequency definition and Brier decomposition](incerto-calibration-payoff.md#calibration-definition), with synthetic calibration curves |
| §11.4, pp. 231–233 | Which performance metric matches the target? | [Amount loss functions](incerto-point-forecast.md#point-forecast-targets), [MASE and sMAPE](incerto-point-forecast.md#point-forecast-metrics); [ruin](incerto-ruin.md) for absorbing paths |
| §11.4.1, pp. 233–235 | Which score distributions and limits follow? | [Binomial pushforward](incerto-calibration-payoff.md#calibration-tally) and [two beta joint laws](incerto-calibration-payoff.md#calibration-beta) |
| §11.5, pp. 235–238 | How do nonlinear exposures enter an ML model? | [Conditional payoff expectation](incerto-calibration-payoff.md#calibration-nonlinear); reuse [hinges](incerto-ml-econometrics.md#pipeline-hinges) and [angle calibration](incerto-ml-econometrics.md#pipeline-angles) from chapter 10/Appendix G |
| §11.6, p. 238 | Does a calibration error itself improve decisions? | [Separate probability reports from decision costs](incerto-calibration-payoff.md#calibration-nonlinear); no universal benefit of miscalibration is inferred |
| §11.7.1, p. 238 | What is the binary tally law? | [Exact finite binomial law, cumulants, and standardized limit](incerto-calibration-payoff.md#calibration-tally) |
| §11.7.2, pp. 239–240 | What is the Brier distribution? | [One-score densities, beta moments, and convolution identity](incerto-calibration-payoff.md#calibration-beta); simulation of averages with exact moment benchmarks |

## The comparison to keep visible

With loss $X\ge0$, threshold $u>0$, strict survival $p_u=P(X>u)>0$,
and finite mean excess $e(u)$,

$$
\begin{gathered}
\mathbb E[X\mathbf1_{\{X>u\}}]
=p_u\{u+e(u)\},\\
\mathbb E[(X-u)_+]=p_u e(u).
\end{gathered}
$$

Knowing $p_u$ leaves $e(u)$ unspecified. The
[asymmetric-payoff page](incerto-asymmetric-payoffs.md) gives a finite
example of the distinction; the
[Pareto moment theorem](incerto-pareto-moment-existence.md) gives the
integrability boundary, and
[moment estimation](incerto-moment-estimation.md) separates a finite
expectation from finite estimator variance. The source's $K$ becomes the
canon's threshold $u$; its forecast $f_i$ becomes local $Q_i$ here,
leaving density and CDF notation unambiguous.

## Source qualifications

The chapter's core payoff distinction does not establish that probability
calibration is useless or that experimental findings in psychology are
spurious. Such conclusions require the experiment's estimand, instructions,
observations, and alternative model. Our examples establish mathematical
nonidentification and sensitivity under stated laws only.

The notes also correct several mathematical overextensions. A tail-amount
multiplier is not necessarily a probability; the exact Pareto multiplier
has denominator $\alpha-1$ for $\alpha>1$. A specified threshold event
remains measurable. Jensen needs convexity in the chosen parameter or
payoff argument. A beta forecast marginal and a Bernoulli event marginal
need a joint-law assumption before their squared-error distribution is
determined. A bounded Brier score has every positive moment, but can have
kurtosis above three. Our binomial and iid-average derivations assume
independence; normal limits require centering and scaling. Finally, sMAPE is bounded,
and conditioning a cumulative payoff on survival does not generally preserve
the underlying payoff distribution.

## A finite score law

Use [`brier_score_distribution`](incerto-api-reference.md#api-brier-score-distribution).
With $p=q=0.2$, the expected Brier score is $p(1-p)=0.16$; increasing
sample size reduces its variance but supplies no missing severity information.

```python
from xmath.incerto.calibration import brier_score_distribution

for n in (1, 10, 100):
    score, mass = brier_score_distribution(n, .2, .2)
    mean = mass @ score
    variance = mass @ (score - mean)**2
    print(f"n={n}: mean={mean:.3f}, variance={variance:.5f}")
```

**Output**

```text
n=1: mean=0.160, variance=0.05760
n=10: mean=0.160, variance=0.00576
n=100: mean=0.160, variance=0.00058
```

The concept pages contain the
[payoff comparison](incerto-calibration-payoff.md#calibration-payoff-figure),
[calibration curves](incerto-calibration-payoff.md#calibration-curve-figure),
[score distributions](incerto-calibration-payoff.md#calibration-score-figure),
and [amount-loss comparison](incerto-point-forecast.md#point-forecast-loss-figure).
Generate all four with `uv run python demos/incerto/calibration_payoffs.py`.
Return to the [SCoFT overview](incerto-scoft-intro.md) for the chapter route.
