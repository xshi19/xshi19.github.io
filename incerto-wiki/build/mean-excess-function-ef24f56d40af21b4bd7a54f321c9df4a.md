---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: mean-excess-function
    type: theorem
    depends_on:
      - pickands-balkema-de-haan
      - generalized-pareto
      - pareto
    tags:
      - peaks-over-threshold
      - diagnostics
---

# Mean Excess Function

## Statement

For a random variable $X$ and a threshold $u$ with $\mathbb P(X>u)>0$ and finite
conditional expectation above $u$, the mean excess function is

$$
e(u)=\mathbb E[X-u\mid X>u].
$$

It measures the expected overshoot beyond $u$ after the threshold has already
been crossed.  For a [Pareto Type I](../distributions/pareto.md) tail with
exponent $\alpha>1$ and $u\ge x_m$,

$$
e(u)=\frac{u}{\alpha-1}.
$$

For a generalized Pareto variable with shape $\xi$, scale $\beta>0$, and
threshold $u$ in its support with $\beta+\xi u>0$, the mean excess function is
linear when $\xi<1$:

$$
e(u)=\frac{\beta+\xi u}{1-\xi}.
$$

The exponential distribution is the boundary case $\xi=0$, where $e(u)$ is
constant.  The symbols $X$, $\mathbb E$, $\alpha$, $\xi$, $\beta$, and $u$
follow the shared [notation table](../../notation/index.md).

## Diagnostic intuition

The mean excess function asks what remains after an event is already large.
Thin-tail intuition often expects the remaining excess to be tame once a high
threshold has been crossed.  A Pareto tail says the opposite: the expected
additional excess grows in proportion to the threshold itself.

This makes the mean excess plot a practical threshold diagnostic.  A roughly
flat plot suggests exponential-like exceedances.  A roughly increasing linear
plot suggests heavy-tail generalized Pareto behavior.  A downward line suggests
a finite endpoint.  Strong curvature usually says the chosen threshold range is
mixing body and tail behavior, or that the model class is too simple.

## Pareto derivation and GPD sketch

We derive the Pareto mean-excess formula from the conditional survival
function.  The generalized Pareto linear formula follows from standard GPD
threshold stability and the GPD mean formula; those facts are cited rather than
reproved from first principles here.

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
$\beta+\xi u$, provided $u$ is in the support and $\beta+\xi u>0$.  Its mean
exists only for $\xi<1$, and equals $(\beta+\xi u)/(1-\xi)$.

## Mean-excess plot

The first panel shows exact theoretical shapes.  The second panel estimates the
same diagnostic from samples and suppresses thresholds with too few
exceedances.

```{code-cell} python
:label: mean-excess-diagnostic-plot
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

from incerto.distributions import pareto_type1
from incerto.estimators import mean_excess
from incerto.figures import COLORS, FIGURE_SIZES, set_theme, style_axes

set_theme()

rng = np.random.default_rng(20260617)
n = 80_000
pareto_alpha = 1.6

u_pareto = np.linspace(1.0, 8.0, 200)
u_bounded = np.linspace(0.0, 4.95, 200)

pareto_sample = pareto_type1.rvs(pareto_alpha, size=n, random_state=rng)
exponential_sample = rng.exponential(scale=1.0, size=n)
bounded_sample = rng.uniform(0.0, 5.0, size=n)

pareto_thresholds = np.quantile(pareto_sample, np.linspace(0.50, 0.98, 35))
exponential_thresholds = np.quantile(exponential_sample, np.linspace(0.50, 0.98, 35))
bounded_thresholds = np.linspace(0.5, 4.8, 35)

pareto_me = mean_excess(pareto_sample, pareto_thresholds, min_exceedances=100)
exponential_me = mean_excess(
    exponential_sample,
    exponential_thresholds,
    min_exceedances=100,
)
bounded_me = mean_excess(bounded_sample, bounded_thresholds, min_exceedances=100)

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"])

axes[0].plot(u_pareto, u_pareto / (pareto_alpha - 1), label="Pareto")
axes[0].plot(u_pareto, np.ones_like(u_pareto), label="exponential")
axes[0].plot(u_bounded, (5.0 - u_bounded) / 2.0, label="bounded uniform")
axes[0].set_xlabel("threshold u")
axes[0].set_ylabel(r"theoretical $e(u)$")
axes[0].set_title("Theoretical mean excess shapes")
axes[0].legend()

axes[1].plot(
    pareto_me["threshold"],
    pareto_me["mean_excess"],
    color=COLORS["green"],
    label="Pareto sample",
)
axes[1].plot(
    exponential_me["threshold"],
    exponential_me["mean_excess"],
    color=COLORS["teal"],
    label="exponential sample",
)
axes[1].plot(
    bounded_me["threshold"],
    bounded_me["mean_excess"],
    color=COLORS["umber"],
    label="bounded sample",
)
axes[1].set_xlabel("threshold u")
axes[1].set_ylabel(r"empirical $e(u)$")
axes[1].set_title("Sample mean excess diagnostics")
axes[1].legend()

style_axes(axes, grid_axis="both")
plt.show()
```

**What to notice.** Pareto mean excess rises with the threshold, exponential
mean excess is flat, and bounded-tail mean excess slopes downward toward the
endpoint.  The sample curves are diagnostics; high thresholds trade lower bias
for fewer exceedances and more noise.

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

## Backlinks

- Depends on: [Pickands-Balkema-de Haan Theorem](pickands-balkema-de-haan.md),
  [Generalized Pareto Distribution](../distributions/generalized-pareto.md),
  [Pareto Distribution](../distributions/pareto.md), and threshold notation in
  [Notation](../../notation/index.md).
- Used by: [S&P 500 Tail Diagnostics](../examples/sp500-tail.md) and
  [Tail Threshold Selection](../methods/tail-threshold-selection.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/theorems/mean-excess-function.md`. Last verified: 2026-06-25. Checked against cited sources, scoped Pareto/GPD derivations, and executable mean-excess diagnostics.
:::
<!-- incerto-provenance:end -->
