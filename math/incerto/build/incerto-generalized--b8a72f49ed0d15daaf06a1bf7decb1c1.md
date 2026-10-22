---
title: Generalized Extreme-Value Distribution
options:
  concept:
    id: generalized-extreme-value
    type: distribution
    depends_on:
      - extreme-value-index
    tags:
      - extreme-value-theory
      - block-maxima
      - tail-risk
---

## Statement

The generalized extreme-value distribution is the standard limit family for
normalized block maxima.  With location $\mu$, scale $\sigma>0$, shape
$\xi$, and standardized variable $z=(x-\mu)/\sigma$, its CDF is

$$
H_{\xi,\mu,\sigma}(x)
=
\exp\left[-\left(1+\xi z\right)^{-1/\xi}\right],
\qquad \xi\ne0,
$$

where $1+\xi z>0$. For $\xi>0$, extend the CDF by zero at and
below $\mu-\sigma/\xi$; for $\xi<0$, extend it by one at and above
$\mu-\sigma/\xi$. The boundary case is the Gumbel law,

$$
H_{0,\mu,\sigma}(x)
=
\exp\left[-e^{-z}\right].
$$

The shape parameter $\xi$ is the same extreme-value index used in threshold
excesses: $\xi>0$ is [Frechet-type](./incerto-frechet.md) heavy-tail behavior,
$\xi=0$ is Gumbel-type behavior, and $\xi<0$ is
Weibull-type finite endpoint.
Here $X_i$ are observations and $M_n=\max_{1\le i\le n}X_i$. The Pareto
example below uses lower cutoff $x_m>0$ and exponent $\alpha>0$.
Gumbel attraction is broader than exponential parent tails: normal and
lognormal parents are examples, despite their different tail decay.

## Maxima intuition

The distribution is to block maxima what the [Generalized Pareto Distribution](./incerto-generalized-pareto.md)
is to threshold exceedances. For iid observations, if there are constants
$a_n>0$ and $b_n$ such that $(M_n-b_n)/a_n$ converges to a nondegenerate
law, that law is a GEV distribution up to location and scale. Existence
of such a limit is an assumption; it does not hold for every parent law.

For fat-tail work, $\xi>0$ is the main case.  A [Pareto tail](./incerto-pareto.md) with exponent
$\alpha$ has $\xi=1/\alpha$, so the fitted block-maximum shape should point to
the same tail coordinate as a Hill or GPD threshold analysis when the modeling
assumptions are reasonable.

## Shape regimes

(generalized-extreme-value-cdf-plot)=
With $\mu=0$ and $\sigma=1$, the support makes the three shapes explicit.

| Shape $\xi$ | Interior CDF | Support |
| --- | --- | --- |
| $-0.3$ | $\exp[-(1-0.3x)^{10/3}]$ | $x<10/3$; CDF is one above |
| $0$ | $\exp[-e^{-x}]$ | All real $x$ |
| $0.3$ | $\exp[-(1+0.3x)^{-10/3}]$ | $x>-10/3$; CDF is zero below |

Every curve passes through $H(0)=e^{-1}$. Positive shape gives a power-law
right tail; negative shape gives a finite right endpoint.

```{figure} ../assets/figures/incerto/gev-card.svg
:name: gev-distribution-card
:alt: GEV densities and CDFs show a left endpoint for positive shape and a right endpoint for negative shape. A third log-log panel shows survival only at positive thresholds.

Exact GEV laws with $(\mu,\sigma)=(0,1)$ and $\xi=-0.3,0,0.3$.
The dotted boundaries are $-10/3$ for positive shape and $10/3$ for
negative shape. The Gumbel law has no finite endpoint. The log-log panel
shows only positive thresholds, with survival below $10^{-8}$ outside
the viewport and exact zero omitted.
```

[`generalized_extreme_value`](incerto-api-reference.md#api-generalized-extreme-value)
uses the shape sign in this page's CDF. It passes `c=-xi` to
[SciPy's `genextreme`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html).

```python
from xmath.incerto.distributions import generalized_extreme_value

for xi in (-0.3, 0, 0.3):
    law = generalized_extreme_value(xi)
    left, right = law.support()
    print(f"xi={xi:.4g}: support=[{left:.4g}, {right:.4g}], cdf(0)={law.cdf(0):.4g}")
```

**Output**

```text
xi=-0.3: support=[-inf, 3.333], cdf(0)=0.3679
xi=0: support=[-inf, inf], cdf(0)=0.3679
xi=0.3: support=[-3.333, inf], cdf(0)=0.3679
```

Regenerate the card with `uv run python demos/incerto/distribution_cards.py`.

## Pareto block maxima example

If $X_1,\dots,X_n$ are iid [Pareto Type I](./incerto-pareto.md) with lower cutoff $x_m$ and exponent
$\alpha$, then for $x\ge x_m$,

$$
\mathbb P(M_n\le x)
=
\left(1-\left(\frac{x_m}{x}\right)^\alpha\right)^n.
$$

With $a_n=x_m n^{1/\alpha}$, for fixed $y>0$ and sufficiently large $n$,

$$
\mathbb P(M_n/a_n\le y)
=
\left(1-\frac{1}{ny^\alpha}\right)^n
\to
\exp(-y^{-\alpha}),\qquad y>0.
$$

This is the [Frechet member](./incerto-frechet.md) of the GEV family with
$\xi=1/\alpha$ after a standard location-scale reparameterization.

## Block-size calibration

(generalized-extreme-value-python-check)=
For an exact Pareto parent with $\alpha=1.8=9/5$, the limiting shape is
$\xi=5/9$. At block size $n=250$, the normalizing scale is
$a_n=x_m250^{5/9}$. At the normalized threshold $y=1$,

$$
\mathbb P(M_{250}/a_{250}\le1)
=(249/250)^{250}\approx0.367142,
\qquad \Phi_{1.8}(1)=e^{-1}\approx0.367879.
$$

This compares an exact finite-block CDF with its limit; it is not a GEV fit
or an assessment of sampling uncertainty. The standard Frechet limit has GEV
parameters $(\xi,\mu,\sigma)=(5/9,1,5/9)$ on the normalized scale.

## Caveats

- Block maxima discard within-block information.  Threshold methods may use
  extremes more efficiently, but they require a threshold choice.
- Blocks should be chosen with dependence and seasonality in mind.  Overlapping
  or strongly dependent blocks can make fitted uncertainty too optimistic.
- A GEV fit describes maxima, not the entire parent distribution.
- Software may use a different shape sign convention. Translate fitted
  parameters into the displayed CDF before comparing maxima and excesses.

## References

- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [2001](https://doi.org/10.1007/978-1-4471-3675-0).
- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  (2nd ed., Wiley, 1971).

## Backlinks

- Depends on: [Extreme-Value Index Estimation](./incerto-extreme-value-index.md).
- Used by: [Frechet Distribution and Frechet-Type Limits](./incerto-frechet.md),
  [Pickands-Balkema-de Haan Theorem](./incerto-pickands-balkema-de-haan.md)
  as the block-maxima counterpart to threshold excess limits, and by future
  return-level pages.
