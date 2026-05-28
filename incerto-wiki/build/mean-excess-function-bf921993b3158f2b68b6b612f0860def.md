---
kernelspec:
  name: python3
  display_name: Python 3
---

# Mean Excess Function

## Statement

For a random variable $X$ with finite conditional expectation above threshold
$u$, the mean excess function is

$$
e(u)=\mathbb E[X-u\mid X>u].
$$

It measures the expected overshoot beyond $u$ after the threshold has already
been crossed.  For a Pareto Type I tail with exponent $\alpha>1$ and
$u\ge x_m$,

$$
e(u)=\frac{u}{\alpha-1}.
$$

For a generalized Pareto variable with shape $\xi<1$ and scale $\beta$, the
mean excess function is linear:

$$
e(u)=\frac{\beta+\xi u}{1-\xi}.
$$

The exponential distribution is the boundary case $\xi=0$, where $e(u)$ is
constant.

## Intuition

The mean excess function asks what remains after an event is already large.
Thin-tail intuition often expects the remaining excess to be tame once a high
threshold has been crossed.  A Pareto tail says the opposite: the expected
additional excess grows in proportion to the threshold itself.

This makes the mean excess plot a practical threshold diagnostic.  A roughly
flat plot suggests exponential-like exceedances.  A roughly increasing linear
plot suggests heavy-tail generalized Pareto behavior.  A strongly curved plot
usually says the chosen threshold range is mixing body and tail behavior, or
that the model class is too simple.

## Proof

For Pareto Type I,

$$
\mathbb P(X>x)=\left(\frac{x_m}{x}\right)^\alpha,
\qquad x\ge x_m.
$$

For $u\ge x_m$ and $\alpha>1$,

$$
e(u)
=
\mathbb E[X-u\mid X>u]
=
\int_0^\infty \mathbb P(X-u>y\mid X>u)\,dy.
$$

Using the conditional survival calculation from the
[Pickands-Balkema-de Haan Theorem](pickands-balkema-de-haan.md),

$$
\mathbb P(X-u>y\mid X>u)
=
\left(1+\frac{y}{u}\right)^{-\alpha}.
$$

Therefore

$$
e(u)
=
\int_0^\infty \left(1+\frac{y}{u}\right)^{-\alpha}\,dy
=
u\int_1^\infty z^{-\alpha}\,dz
=
\frac{u}{\alpha-1}.
$$

For a generalized Pareto distribution, the threshold-stability property gives
another generalized Pareto distribution above threshold $u$ with updated scale
$\beta+\xi u$.  Its mean exists only for $\xi<1$, and equals
$(\beta+\xi u)/(1-\xi)$.

## Python

The empirical mean excess function is noisy in the far tail because high
thresholds leave few exceedances.  The helper below reports both the mean
excess and the exceedance count.

```{code-cell} python
:label: mean-excess-python-check

import numpy as np
import matplotlib.pyplot as plt

from incerto.distributions import pareto_type1
from incerto.estimators import mean_excess
from incerto.figures import COLORS, FIGURE_SIZES, set_theme, style_axes

set_theme()

rng = np.random.default_rng(20260524)
n = 80_000

pareto_alpha = 1.6
pareto_sample = pareto_type1.rvs(pareto_alpha, size=n, random_state=rng)
exponential_sample = rng.exponential(scale=1.0, size=n)

thresholds = np.quantile(pareto_sample, np.linspace(0.50, 0.98, 35))
pareto_me = mean_excess(pareto_sample, thresholds, min_exceedances=100)
exponential_me = mean_excess(exponential_sample, thresholds, min_exceedances=100)

fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
ax.plot(
    pareto_me["threshold"],
    pareto_me["mean_excess"],
    color=COLORS["green"],
    label="Pareto sample",
)
ax.plot(
    pareto_me["threshold"],
    pareto_me["threshold"] / (pareto_alpha - 1),
    color=COLORS["accent"],
    ls="--",
    lw=1.0,
    label="Pareto e(u)=u/(alpha-1)",
)
ax.plot(
    exponential_me["threshold"],
    exponential_me["mean_excess"],
    color=COLORS["teal"],
    label="exponential sample",
)
ax.set_xlabel("threshold u")
ax.set_ylabel("mean excess e(u)")
ax.set_title("Empirical mean excess function")
ax.legend()
style_axes(ax, grid_axis="both")
plt.show()

last = -1
print(
    "Highest plotted Pareto threshold: "
    f"u={pareto_me['threshold'][last]:.2f}, "
    f"exceedances={pareto_me['exceedances'][last]:.0f}, "
    f"mean excess={pareto_me['mean_excess'][last]:.2f}"
)
```

The Pareto curve rises with the threshold.  The exponential comparison is much
flatter, but it becomes unavailable at very high Pareto thresholds because the
exponential sample has too few exceedances there.

## Caveats

- The mean excess function is itself a mean.  If the fitted tail has
  $\xi\ge1$, the theoretical mean excess is infinite.
- Empirical mean excess plots are unstable at high thresholds.  Always inspect
  exceedance counts.
- Linear-looking behavior is suggestive, not decisive.  Mixtures and finite
  upper truncation can create misleading curvature.
- For two-sided returns, apply the function to a one-sided loss variable or to
  absolute returns after stating the modeling choice.

## References

- Davison and Smith, "Models for Exceedances over High Thresholds"
  [@davison1990models].
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [@coles2001introduction].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Taleb, *Statistical Consequences of Fat Tails* [@taleb2020scoft].

## Backlinks

- Depends on: [Pickands-Balkema-de Haan Theorem](pickands-balkema-de-haan.md),
  [Pareto Distribution](../distributions/pareto.md), and threshold notation in
  [Notation](../../notation/index.md).
- Used by: [S&P 500 Tail Fit](../examples/sp500-tail.md) and future
  peaks-over-threshold examples.

<!-- incerto-provenance:start -->
<footer class="incerto-provenance">
<p><strong>Provenance.</strong> Source: <code>content/concepts/theorems/mean-excess-function.md</code>. Last verified: 2026-05-24. Checked against cited sources, page proof or computation, and executable examples.</p>
</footer>
<!-- incerto-provenance:end -->
