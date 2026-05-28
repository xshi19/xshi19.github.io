# Pre-Asymptotic LLN Behavior

## Statement

This example compares running sample means for Pareto variables near the
critical tail exponent $\alpha=1$. Let

$$
\bar X_n=\frac{1}{n}\sum_{i=1}^n X_i.
$$

For a Pareto Type I sample:

- If $\alpha\le1$, $\bar X_n\to\infty$ almost surely.
- If $\alpha>1$, $\bar X_n\to \alpha/(\alpha-1)$ almost surely when
  $x_m=1$.
- If $\alpha$ is only slightly above 1, the second statement can be practically
  hard to see in finite samples.

The visualizer below fixes a pseudo-random seed so changing $\alpha$ and $n$
shows the same experiment under different tail exponents.

```{raw} html
<div class="incerto-widget incerto-widget-grid" data-incerto-widget="lln-preasymptotic">
  <div class="incerto-widget-controls">
    <div class="incerto-widget-control-row">
      <label for="lln-alpha-example">alpha</label>
      <output id="lln-alpha-example-value" data-lln-alpha-value for="lln-alpha-example">1.10</output>
      <input id="lln-alpha-example" data-lln-alpha type="range" min="0.70" max="2.50" step="0.05" value="1.10">
    </div>
    <div class="incerto-widget-control-row">
      <label for="lln-n-example">draws</label>
      <output id="lln-n-example-value" data-lln-n-value for="lln-n-example">5000</output>
      <input id="lln-n-example" data-lln-n type="range" min="500" max="10000" step="500" value="5000">
    </div>
  </div>
  <div class="incerto-widget-stats" aria-live="polite">
    <div class="incerto-widget-stat">
      <strong>Final mean</strong>
      <span data-lln-final-mean></span>
    </div>
    <div class="incerto-widget-stat">
      <strong>Largest share</strong>
      <span data-lln-largest-share></span>
    </div>
    <div class="incerto-widget-stat">
      <strong>Theoretical mean</strong>
      <span data-lln-theoretical-mean></span>
    </div>
  </div>
  <figure class="incerto-widget-plot">
    <svg data-lln-plot aria-label="Pareto running mean path"></svg>
  </figure>
</div>
```

## Intuition

The law of large numbers is an asymptotic statement. In thin-tailed settings,
the path toward the limit is often regular enough that the asymptotic statement
also feels like a finite-sample guide. Near $\alpha=1$, the path can instead
move by jumps. A single new maximum can rewrite the running mean long after the
sample looked settled.

This is the pedagogical point behind the phrase "pre-asymptotic": before the
limit has had enough room to express itself, the sample may be governed by the
largest observation seen so far.

## Proof

The asymptotic part follows from the two linked concept pages:

- [LLN Failure Under Infinite Mean](../theorems/lln-failure.md) proves
  $\bar X_n\to\infty$ almost surely for iid nonnegative variables with
  infinite mean.
- [Pareto Moment Existence](../theorems/pareto-moment-existence.md) gives the
  exact Pareto boundary: the mean exists if and only if $\alpha>1$.

The pre-asymptotic claim is not a theorem with a universal cutoff. It is a
finite-sample diagnostic: simulate or compute the running mean, the maximum
share, and sensitivity to the sample size under the stated model.

## Python

```python
import numpy as np
from scipy.stats import pareto
from incerto.estimators import max_to_sum_ratio, running_mean

rng = np.random.default_rng(20260523)
alpha = 1.1
n = 100_000

sample = pareto.rvs(b=alpha, size=n, random_state=rng)
path = running_mean(sample)

print(path[[99, 999, 9_999, 99_999]])
print(max_to_sum_ratio(sample))
print(alpha / (alpha - 1))
```

Changing only the random seed can materially change the path. That is part of
the lesson, not a plotting defect.

## Caveats

- This page illustrates a model. It does not claim that every empirical data
  set near a power law follows an exact Pareto distribution.
- For $\alpha>1$, convergence eventually occurs, but the useful question is how
  much data is needed for the target decision.
- For $\alpha\le1$, the running mean can have long quiet stretches even though
  the limit is infinite.
- The visualizer uses a deterministic pseudo-random path for reproducibility,
  not for statistical inference.

## References

- Taleb, *Statistical Consequences of Fat Tails*, Chapters 3 and 5
  [@taleb2020scoft].
- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [@feller1971introduction].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].

## Backlinks

- Depends on: [Pareto Distribution](../distributions/pareto.md),
  [Pareto Moment Existence](../theorems/pareto-moment-existence.md), and
  [LLN Failure Under Infinite Mean](../theorems/lln-failure.md).
- Used by: [Max-to-Sum Ratio](../methods/max-to-sum-ratio.md) and the
  [Chapter 5 reading guide](../../reading-guides/taleb-scoft/ch5.md).
