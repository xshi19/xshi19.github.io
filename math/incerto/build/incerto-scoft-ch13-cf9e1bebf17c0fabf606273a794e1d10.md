---
title: Chapter 13 — Probability and payoff risk
---

Chapter 13 of *Statistical Consequences of Fat Tails*, third edition,
arXiv:2001.10488v4, is titled “The probability conflation,” printed
pp. 253–260 [@taleb2025scoftv4]. The mathematical question is whether
accuracy about an event transfers to accuracy about its consequences.
The answer depends on what is known about severity and how uncertainty
enters the model.

Read [calibration and payoffs](incerto-calibration-payoff.md) for the
event/amount distinction, [probability conflation](incerto-probability-conflation.md)
for nonidentification and probability error, and
[VaR and expected shortfall](incerto-risk-measures.md) for quantiles and
tail-index uncertainty. The [point-forecast page](incerto-point-forecast.md)
supplies the loss-function and moment background from chapter 12.

## Section map

| V4 locator | Mathematical question | Note |
| --- | --- | --- |
| §13.1, p. 253; §13.2, pp. 253–256 | What does a binary score identify? | [Same scores, different payoff means](incerto-probability-conflation.md#conflation-score-gap) |
| §13.2, p. 254; §13.6, pp. 259–260 | What is missing from threshold times probability? | [Tail-level and excess identities](incerto-calibration-payoff.md#calibration-contract); [whole-curve error and tail control](incerto-probability-conflation.md#conflation-whole-curve) |
| §13.3, pp. 257–258 | How can bounded probability errors produce unbounded amounts? | [Beta report transformed by an inverse Pareto survival](incerto-probability-conflation.md#conflation-inverse-probability) |
| §13.3, pp. 257–258, risk-measure context | How do VaR and ES differ, and how does exponent uncertainty enter? | [Definitions and quantile atoms](incerto-risk-measures.md#risk-definitions); [convexity](incerto-risk-measures.md#risk-alpha-convexity); [predictive mixtures](incerto-risk-measures.md#risk-alpha-mixture), derived extensions |
| §13.4, pp. 258–259 | What can a finite performance record establish? | [Evaluation scope](incerto-probability-conflation.md#conflation-decisions); [sampling uncertainty](incerto-risk-measures.md#risk-estimation) |
| §13.5, p. 259 | When can better probabilities improve decisions? | [Exposure error and a specified binary decision](incerto-calibration-payoff.md#calibration-payoff-error) |

Section 13.1 is a heading on p. 253, followed immediately by §13.2;
we do not assign it a separate theorem. The attached paper
[@taleb2023conflation, §§I–IV, pp. 1–4] supplies the chapter-version
context. Its mathematical model is reviewed; personal, regulatory and
performance commentary is outside these notes.

## One equation, two kinds of information

For a nonnegative loss $X$, threshold $u>0$, and $\bar F(u)=P(X>u)>0$,
Tonelli's theorem gives the tail-level mean

$$
\begin{aligned}
\mathbb E[X\mathbf1_{\{X>u\}}]
&=u\bar F(u)\\
&\quad+\int_u^\infty\bar F(v)\,dv.
\end{aligned}
$$

The first term uses a single event probability. The second uses the
survival curve beyond the threshold and can be infinite. The calibration
page proves the identity without requiring a density or a finite first
moment. The chapter's $K$ maps to our $u$ when it is a fixed threshold;
our random $K$ is reserved for the transformed uncertain probability.
The source's constant Pareto scale maps to $x_m$, leaving $L$ for a
slowly varying function in the shared canon.

For a fixed Pareto exponent $\alpha=1$, a beta report with shapes
$a=2,b=3$ produces a finite mean quantile and an infinite second moment.
Use [`beta_pareto_quantile_moment`](incerto-api-reference.md#api-beta-pareto-quantile-moment).
This is an exact-model calculation, not a fitted probability forecast.

```python
from xmath.incerto.risk_measures import beta_pareto_quantile_moment

for order in (1, 2):
    value = beta_pareto_quantile_moment(2, 3, 1, order)
    print(f"quantile moment, order {order}: {value:.2f}")
```

**Output**

```text
quantile moment, order 1: 4.00
quantile moment, order 2: inf
```

## Scope to retain

- A fixed event has a fixed probability under a fixed law. A random report
  needs a stated joint or uncertainty model; neither a uniform nor a beta
  law follows automatically from boundedness or the probability integral
  transform. Bounded scores also do not guarantee independent observations.
- Zero regret for one binary event leaves conditional severity open.
  The complete distribution does determine payoff expectations, and a common
  cap or uniform tail control can turn probability error into a mean bound.
- Chapter 13 randomizes a probability at fixed exponent. The separate
  exponent-uncertainty results hold for exact Pareto laws at a common cutoff
  and fixed tail mass. Averaging model quantiles and taking the quantile of
  a mixture are different operations.
- The Pareto inverse survival is convex. In the beta-report calculation,
  the positive moment of order $r$ exists exactly when $r<a\alpha$.
  These conclusions are not formulas for every heavy-tailed distribution.
- An unconditional tail-level expectation needs division by the tail mass
  to become ES in a continuous law; quantile atoms require fractional mass.
  VaR can fail subadditivity, and ES can be infinite.
- The limit of unrooted high moments is not an expected maximum.
  For a fixed finite sample, the rooted power mean tends to the sample
  maximum, as proved in the
  [moment-estimation note](incerto-moment-estimation.md#moment-estimation-power-means).
  This is not an identity for the expectation of a random maximum.

The figures show [equal scores and unequal payoffs](incerto-probability-conflation.md#conflation-gap-figure),
[probability error propagation](incerto-probability-conflation.md#conflation-probability-figure),
[exponent uncertainty](incerto-risk-measures.md#risk-mixture-figure), and
[fitted-risk sampling variation](incerto-risk-measures.md#risk-estimation-figure).
Generate them with `uv run python demos/incerto/probability_conflation.py`.
Return to the [SCoFT overview](incerto-scoft-intro.md) for the chapter route.
