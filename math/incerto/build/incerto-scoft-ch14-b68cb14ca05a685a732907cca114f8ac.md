---
title: Chapter 14 — Forecasts through time
---

Chapter 14 of *Statistical Consequences of Fat Tails*, third edition,
arXiv:2001.10488v4, is titled “Election predictions as martingales:
an arbitrage approach,” pp. 261–273 [@taleb2025scoftv4]. Its mathematical
question is how forecasts for the same binary event should fit together
as information arrives. Our [martingale note](incerto-forecast-martingale.md)
contains the proofs, models, and synthetic experiments.

Read [calibration and payoffs](incerto-calibration-payoff.md) for conditional
quadratic scoring, and [probability conflation](incerto-probability-conflation.md)
for the limits of what a binary event forecast says about an amount.
These remain the canonical homes of the chapter-11 and chapter-13 material.

## Section map

| V4 locator | Mathematical question | Note |
| --- | --- | --- |
| Opening, pp. 261–263; §14.0.1, pp. 263–264 | What constrains reports for one event? | [Conditional-probability martingale](incerto-forecast-martingale.md#forecast-tower) |
| §14.0.2, pp. 264–266 | How do terminal distributions enter the calculation? | [Exact bounded transitions](incerto-forecast-martingale.md#forecast-dual) |
| §14.0.3, p. 266 | Which law makes the forecast or price a martingale? | [Fair bets and pricing laws](incerto-forecast-martingale.md#forecast-assessor) |
| §14.1, pp. 266–267 | How do level, horizon, and volatility determine a binary value? | [Gaussian approximation](incerto-forecast-martingale.md#forecast-gaussian); [bounded-model value](incerto-forecast-martingale.md#forecast-dual-value) |
| §14.2, pp. 268–269 | What drift makes a transformed process a martingale? | [Probit and logistic construction](incerto-forecast-martingale.md#forecast-dual) |
| §14.3, pp. 269–271 | How do fair event bets and Brier-optimal reports agree? | [Assessor connection](incerto-forecast-martingale.md#forecast-assessor) |
| §14.4, p. 271; addendum, pp. 271–273 | What does excessive forecast variation mean mathematically? | [Expected revision budget](incerto-forecast-martingale.md#forecast-budget); [pathwise trading identity](incerto-forecast-martingale.md#forecast-trading); [synthetic diagnostics](incerto-forecast-martingale.md#forecast-diagnostics) |

The organizing identity is short. For a fixed event $A$ resolved at the
terminal time and nested information sets $(\mathcal F_t)$ under one law, define

$$
\begin{gathered}
p_t=\mathbb E[\mathbf1_A\mid\mathcal F_t],\\
\mathbb E[p_t\mid\mathcal F_s]=p_s,
\qquad s\le t.
\end{gathered}
$$

The note proves this with the tower property. It does not require a
diffusion, continuous paths, a normal distribution, or a tradable vote share.

## Notation and source qualifications

Here $\Phi$ is the standard normal CDF, as in the
[shared notation](https://xshi19.github.io/math/notation/).
Our $Y$ is the bounded share and $p_t$ the binary forecast, corresponding
to the source's $Y$ and $B$. We use $u$ only for the share threshold and
$k=\Phi^{-1}(u)$ for its latent threshold. In §14.1, p. 267, the
threshold must be transformed consistently; a share threshold of $1/2$
becomes latent threshold zero. The source's sigmoid is
$\Phi(\sqrt2 x)$, so our latent coordinate is $z=\sqrt2 x$ and our
latent volatility is $\sqrt2$ times its volatility. Our $\sigma$ is
never an unqualified estimate of vote-share volatility.

- The bounded model's terminal value is derived from its exact Gaussian
  latent transition. We do not use the delta approximation to share
  variance in §14.0.2, pp. 265–266 as an exact volatility calibration.
- In §14.1, p. 267, the valuation must use the survival orientation:
  increasing the current level increases the value of the strict upper
  event. The time derivative in the backward equation is generally
  nonzero; the total drift, including that derivative, cancels.
- Proposition 14.1, p. 268 requires a conditional qualification: a
  midpoint start can preserve the unconditional mean of a sigmoid of
  Brownian motion, but does not make that process a martingale. The note
  gives the drift cancellation and a direct conditional-expectation proof.
- A probability tends to $1/2$ under increasing centered Gaussian noise
  at a fixed signal. The chapter's bounded martingale construction instead
  tends to the current share at fixed interior threshold. Neither result
  is a consequence of boundedness or no-arbitrage alone. The sensitivity
  to the underlying level also differs from forecast-path volatility.
- Risk-neutral pricing does not identify subjective or physical
  probabilities without extra assumptions. The finite betting argument
  proves no arbitrage under its specified law and execution rules; a
  positive expected trading gain alone is not a sure profit.
- Expected total squared revisions equal the initial Bernoulli variance
  when the final report resolves the event. A realized path can exceed
  that expectation. The addendum's constant-volatility Brownian example
  cannot simultaneously remain bounded without changing its dynamics.

For a numerical comparison use
[`gaussian_binary_probability`](incerto-api-reference.md#api-gaussian-binary-probability)
and [`probit_binary_probability`](incerto-api-reference.md#api-probit-binary-probability).
The two scales have different units, so these calls illustrate their
respective limits, not a volatility-matched model comparison.

```python
from xmath.incerto.martingales import (
    gaussian_binary_probability, probit_binary_probability,
)

for scale in (1, 10):
    normal = gaussian_binary_probability(.54, scale, 1)
    bounded = probit_binary_probability(.54, scale, 1)
    print(f"scale={scale}: Gaussian={normal:.4f}, bounded={bounded:.4f}")
```

**Output**

```text
scale=1: Gaussian=0.5160, bounded=0.5503
scale=10: Gaussian=0.5016, bounded=0.5400
```

The figures compare [uncertainty limits](incerto-forecast-martingale.md#forecast-uncertainty-figure),
[the two martingales and their latent coordinate](incerto-forecast-martingale.md#forecast-dual-paths-figure),
and [coherent versus deliberately altered revisions](incerto-forecast-martingale.md#forecast-revision-budget-figure).
The multiple-candidate heuristic and commentary about individual forecasters
are not used as mathematical results. All experiments use simulated data.
Return to the [SCoFT overview](incerto-scoft-intro.md) for the chapter route.
