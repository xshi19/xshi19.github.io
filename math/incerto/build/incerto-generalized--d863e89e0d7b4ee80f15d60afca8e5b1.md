---
title: Generalized Pareto Distribution
options:
  concept:
    id: generalized-pareto
    type: distribution
    depends_on:
      - extreme-value-index
    tags:
      - extreme-value-theory
      - peaks-over-threshold
      - tail-risk
---

## Statement

The generalized Pareto distribution is the standard limit family for threshold
excesses.  With location zero, an excess variable $Y\ge0$ has shape
$\xi$ and scale $\beta>0$ when

$$
G_{\xi,\beta}(y)
=
1-\left(1+\frac{\xi y}{\beta}\right)^{-1/\xi},
\qquad \xi\ne0,
$$

for $y\ge0$ with $1+\xi y/\beta>0$. For $y<0$ the CDF is zero;
for $\xi<0$ it is one at and above $-\beta/\xi$, by continuity.  The boundary case is

$$
G_{0,\beta}(y)=1-e^{-y/\beta}.
$$

For $\xi\ge0$, the support is $y\ge0$.  For $\xi<0$, the support is
$0\le y\le -\beta/\xi$, so the tail has a finite endpoint.  The survival
function for $\xi\ne0$ is

$$
\bar G_{\xi,\beta}(y)
=
\left(1+\frac{\xi y}{\beta}\right)^{-1/\xi}.
$$

The mean exists only for $\xi<1$ and equals $\beta/(1-\xi)$.  The variance
exists only for $\xi<1/2$ and equals

$$
\frac{\beta^2}{(1-\xi)^2(1-2\xi)}.
$$

## Shape and endpoint intuition

The generalized Pareto distribution is the peaks-over-threshold counterpart of
the generalized extreme-value distribution for block maxima.  Once a threshold
is high enough, the distribution of the excess over that threshold is modeled
with one shape parameter $\xi$ and one scale parameter $\beta$.

The sign and size of $\xi$ carry the tail story.  Positive $\xi$ gives a
[Pareto-type](./incerto-pareto.md) heavy tail.  Zero gives the exponential boundary.  Negative $\xi$
gives a finite endpoint.  In the fat-tail examples, the most important
boundary values are $\xi=1/2$ for variance and $\xi=1$ for the mean.

(generalized-pareto-survival-plot)=
For $\beta=1$, the three examples from the survival formula are:

| Shape $\xi$ | Survival $\bar G(y)$ | Upper endpoint |
| --- | --- | --- |
| $-0.4$ | $(1-0.4y)^{2.5}$ for $0\le y\le2.5$ | $2.5$ |
| $0$ | $e^{-y}$ for $y\ge0$ | Infinite |
| $0.4$ | $(1+0.4y)^{-2.5}$ for $y\ge0$ | Infinite |

**What to notice.** Positive $\xi$ gives the slowest-decaying curve, $\xi=0$
is the exponential boundary, and negative $\xi$ ends at the finite endpoint
$-\beta/\xi$.

```{figure} ../assets/figures/incerto/gpd-card.svg
:name: gpd-distribution-card
:alt: Stacked density, CDF and log-log survival panels compare negative, zero and positive GPD shape. The negative-shape tail ends at 2.5.

Exact GPD laws with $\beta=1$ and $\xi=-0.4,0,0.4$. All start at
zero; the negative-shape law ends at $-\beta/\xi=2.5$, marked by a
matching dotted line. Its CDF stays at one beyond that point. Zero
survival is omitted on logarithmic axes; it is never replaced by a small
positive probability. The survival viewport starts at $10^{-8}$.
```

Use [`generalized_pareto`](incerto-api-reference.md#api-generalized-pareto).
The endpoint calculation is visible in the returned CDF and survival:

```python
from xmath.incerto.distributions import generalized_pareto

law = generalized_pareto(-0.4, beta=1)
for y in (0, 1, 2.5, 3):
    print(f"y={y:.4g}: pdf={law.pdf(y):.4g}, cdf={law.cdf(y):.4g}, sf={law.sf(y):.4g}")
```

**Output**

```text
y=0: pdf=1, cdf=0, sf=1
y=1: pdf=0.4648, cdf=0.7211, sf=0.2789
y=2.5: pdf=0, cdf=1, sf=0
y=3: pdf=0, cdf=1, sf=0
```

Regenerate the card with `uv run python demos/incerto/distribution_cards.py`.

## Threshold-excess stability and Pareto special case

A GPD is stable under raising the threshold. If $Y$ has parameters
$\xi,\beta$ and $u\ge0$ satisfies $\beta+\xi u>0$, then, for admissible
$y\ge0$ and $\xi\ne0$,

$$
\frac{\bar G_{\xi,\beta}(u+y)}{\bar G_{\xi,\beta}(u)}
=\left(1+\frac{\xi y}{\beta+\xi u}\right)^{-1/\xi}.
$$

For $\xi=0$, the ratio is $e^{-y/\beta}$. Thus excesses have the same
shape and updated scale $\beta+\xi u$. Combining this with the GPD mean
gives the [mean-excess formula](./incerto-mean-excess-function.md).

The Pareto Type I distribution is an exact generalized Pareto excess model.
If $X$ has Pareto tail exponent $\alpha$ and threshold $u\ge x_m$, then

$$
\mathbb P(X-u>y\mid X>u)
=
\left(1+\frac{y}{u}\right)^{-\alpha}.
$$

This is the generalized Pareto survival function with

$$
\xi=\frac1\alpha,
\qquad
\beta(u)=\frac{u}{\alpha}.
$$

The [Pickands-Balkema-de Haan theorem](./incerto-pickands-balkema-de-haan.md)
explains why the same family appears asymptotically for a much larger class of
threshold exceedances.

## Formula check

A concrete choice of shape and scale illustrates both the survival and moments.

(generalized-pareto-python-check)=
For $\xi=0.4$ and $\beta=2$, direct substitution gives

$$
\bar G(y)=(1+0.2y)^{-2.5},\qquad
\mathbb E[Y]=\frac{10}{3},\qquad
\operatorname{Var}(Y)=\frac{500}{9}.
$$

Thus $\bar G(0)=1$, $\bar G(1)=1.2^{-2.5}$,
$\bar G(5)=2^{-2.5}$, and $\bar G(10)=3^{-2.5}$.
These are formula evaluations, with no numerical fitting or simulation.

## Caveats

- A GPD fit is a tail model for exceedances, not a model for the full
  distribution body.
- The threshold $u$ is not chosen by the theorem.  Shape and scale estimates
  should be inspected across a range of thresholds.
- When $\xi\ge1$, the fitted excess distribution has no finite mean.  When
  $\xi\ge1/2$, it has no finite variance.
- Dependence, truncation, censoring, seasonality, and mixture effects can
  distort threshold-excess fits.

## References

- Balkema and de Haan, "Residual Life Time at Great Age"
  [1974](https://doi.org/10.1214/aop/1176996548).
- Pickands, "Statistical Inference Using Extreme Order Statistics"
  [1975](https://doi.org/10.1214/aos/1176343003).
- Davison and Smith, "Models for Exceedances over High Thresholds"
  [1990](https://doi.org/10.1111/j.2517-6161.1990.tb01796.x).
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [2001](https://doi.org/10.1007/978-1-4471-3675-0).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).

## Backlinks

- Depends on: [Extreme-Value Index Estimation](./incerto-extreme-value-index.md)
  and the shared tail notation in [Shared notation](https://xshi19.github.io/math/notation/).
- Used by: [Pickands-Balkema-de Haan Theorem](./incerto-pickands-balkema-de-haan.md),
  [Mean Excess Function](./incerto-mean-excess-function.md), and
  [Tail Threshold Selection](./incerto-tail-threshold-selection.md).
