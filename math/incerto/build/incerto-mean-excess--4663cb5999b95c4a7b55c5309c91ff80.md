---
title: Mean Excess Function
options:
  concept:
    id: mean-excess-function
    type: theorem
    depends_on:
      - generalized-pareto
      - pareto
    tags:
      - peaks-over-threshold
      - diagnostics
---

## Statement

For a random variable $X$ and a threshold $u$ with $\mathbb P(X>u)>0$ and finite
conditional expectation above $u$, the mean excess function is

$$
e(u)=\mathbb E[X-u\mid X>u].
$$

It measures the expected overshoot beyond $u$ after the threshold has already
been crossed.  For a [Pareto Type I](./incerto-pareto.md) tail with
exponent $\alpha>1$ and $u\ge x_m$,

$$
e(u)=\frac{u}{\alpha-1}.
$$

For a [generalized Pareto variable](./incerto-generalized-pareto.md)
with location zero, shape $\xi$, scale $\beta>0$, and
threshold $u$ in its support with $\beta+\xi u>0$, the mean excess function is
linear when $\xi<1$:

$$
e(u)=\frac{\beta+\xi u}{1-\xi}.
$$

The exponential distribution is the boundary case $\xi=0$, where $e(u)$ is
constant. Here $u$ is the threshold, $\xi$ the GPD shape, and $\beta$
its scale at threshold zero.

## Diagnostic intuition

The mean excess function asks what remains after an event is already large.
Thin-tail intuition often expects the remaining excess to be tame once a high
threshold has been crossed.  A Pareto tail says the opposite: the expected
additional excess grows in proportion to the threshold itself.

The exponential, positive-shape GPD with $\xi<1$, and negative-shape GPD
models have constant, increasing linear, and decreasing linear population
mean excess functions, respectively. Curvature departs from an exact GPD model
over that threshold range. An empirical plot can also bend because of sampling
noise, especially when few observations exceed the threshold; its shape alone
does not identify a model failure or a body-to-tail transition.

## Pareto derivation and GPD sketch

We derive the Pareto mean-excess formula from the conditional survival
function.  The generalized Pareto linear formula follows from standard GPD
[threshold stability](./incerto-generalized-pareto.md#threshold-excess-stability-and-pareto-special-case)
and the GPD mean formula. The GPD page derives the stability identity; its
moment formulas are cited rather than reproved here.

For Pareto Type I,

$$
\mathbb P(X>x)=\left(\frac{x_m}{x}\right)^\alpha,
\qquad x\ge x_m.
$$

For $u\ge x_m$ and $\alpha>1$,

$$
\begin{gathered}
e(u)=\mathbb E[X-u\mid X>u]\\
=\int_0^\infty P(X-u>y\mid X>u)\,dy.
\end{gathered}
$$

Taking the ratio $\bar F(u+y)/\bar F(u)$ for $y\ge0$ gives

$$
\begin{aligned}
&\mathbb P(X-u>y\mid X>u)\\
&\quad=\left(1+\frac{y}{u}\right)^{-\alpha}.
\end{aligned}
$$

Therefore

$$
\begin{aligned}
e(u)&=\int_0^\infty\left(1+\frac yu\right)^{-\alpha}\,dy\\
&=u\int_1^\infty z^{-\alpha}\,dz\\
&=\frac{u}{\alpha-1}.
\end{aligned}
$$

For a generalized Pareto distribution, the threshold-stability property gives
another generalized Pareto distribution above threshold $u$ with updated scale
$\beta+\xi u$, provided $u$ is in the support and $\beta+\xi u>0$.  Its mean
exists only for $\xi<1$, and equals $(\beta+\xi u)/(1-\xi)$.

## Mean-excess diagnostic

Exact theoretical shapes provide a baseline for the empirical diagnostic.
Always report how many observations exceed each threshold.

(mean-excess-diagnostic-plot)=
The theoretical shapes can be compared directly:

| Distribution | Threshold range | Mean excess $e(u)$ |
| --- | --- | --- |
| Pareto, $x_m=1$, $\alpha=1.6$ | $u\ge1$ | $u/0.6$ |
| Exponential with mean $1$ | $u\ge0$ | $1$ |
| Uniform on $[0,5]$ | $0\le u<5$ | $(5-u)/2$ |

For observations $x_1,\ldots,x_n$, let $k(u)=\sum_i\mathbf1_{\{x_i>u\}}$.
The empirical diagnostic is

$$
\begin{gathered}
\widehat e_n(u)=\frac{\sum_{i:x_i>u}(x_i-u)}{k(u)},\\
k(u)>0.
\end{gathered}
$$

For the fixed observations $(1,2,2,4)$ at $u=2$, only $4$ exceeds the
threshold, so $k(2)=1$ and $\widehat e_n(2)=2$. At $u=4$ the mean excess
estimate is undefined because the exceedance count is zero. The
[counting example](./incerto-counting-exceedances.md) uses the same strict
inequality convention.

Pareto mean excess rises with the threshold, exponential mean excess is flat,
and the displayed uniform mean excess decreases toward its endpoint.
For any distribution with finite right endpoint
$x_F=\sup\{x:F(x)<1\}$ and $u<x_F$, conditioning on $X>u$ gives
$0<X-u\le x_F-u$ almost surely. Taking conditional expectations proves

$$
\begin{gathered}
0\le e(u)\le x_F-u,\\
e(u)\longrightarrow0\quad\text{as }u\uparrow x_F.
\end{gathered}
$$

This bound does not imply monotonicity. For example, if
$\mathbb P(X=1)=\mathbb P(X=3)=1/2$, both atoms exceed $0.9$, so
$e(0.9)=[(1-0.9)+(3-0.9)]/2=1.1$. At $1.1$ only the atom at $3$
exceeds the threshold, so $e(1.1)=3-1.1=1.9$. The mean excess has increased
even though $x_F=3$.

Empirical values are diagnostics; high thresholds aim to reduce tail-model
bias but leave fewer exceedances and more noise.

(mean-excess-synthetic)=
## Population curves and sample estimates

```{figure} ../assets/figures/incerto/mean-excess.svg
:alt: Three pairs of vertically stacked panels compare increasing Pareto, flat exponential and decreasing uniform population mean excess with noisy estimates and their retained counts.

Dashed curves are population mean excesses; solid curves are estimates
from $n=30{,}000$ iid observations per law. Here the Pareto exponent is
$1.7$ to reuse the exact-Pareto sample from the Hill comparison; the earlier
table uses $1.6$. The exponential mean is $1$ and the uniform support is
$[0,5]$. Seeds are 20261020, 20261022 and 20261023, respectively (PCG64).
Each estimate stops when fewer than 50 observations strictly exceed the
threshold; the shaded count region remains visible. The uniform population
curve is drawn only for $u<5$, where conditioning is defined. Pareto laws
with $\alpha\le1$ have infinite population mean excess: this is labelled,
with no finite population curve drawn for those laws.
```

Use [`mean_excess`](incerto-api-reference.md#api-mean-excess); masking
retains the counts and reports an unavailable estimate as `nan`:

```python
from xmath.incerto.estimators import mean_excess

result = mean_excess([1, 2, 2, 4], [1, 2, 4], min_exceedances=2)
for u, count, mean in zip(result["threshold"], result["exceedances"], result["mean_excess"]):
    print(f"u={u:.4g}: count={int(count):d}, mean_excess={mean:.4g}")
```

**Output**

```text
u=1: count=3, mean_excess=1.667
u=2: count=1, mean_excess=nan
u=4: count=0, mean_excess=nan
```

Regenerate the figure with `uv run python demos/incerto/tail_estimation.py`.
The [threshold comparison](incerto-tail-threshold-selection.md#threshold-synthetic-figures)
adds the shared mixture sample, whose body changes the population curve below $3$.

(mean-excess-hidden-tail)=
## From overshoot to an unconditional tail contribution

For an integrable upper tail with $\bar F(u)>0$, conditioning proves

$$
\mathbb E[X\mathbf1_{\{X>u\}}]
=\bar F(u)\{u+e(u)\}.
$$

The factor $\bar F(u)$ and the threshold term $u$ are both needed.
For an exact Pareto parent with $\alpha>1$, the formula agrees with the
[hidden mean above a sample maximum](incerto-hidden-tail.md#hidden-tail-pareto).
For $\alpha\le1$, both the conditional mean excess and the hidden
first moment are infinite at every finite admissible threshold, even
though a nonempty finite exceedance sample yields a finite empirical mean excess.

At the observed maximum, the empirical excess has no observations and
is undefined. The existing runnable count example prints `nan` there;
it does not supply evidence for a bounded population. The finite-endpoint
bound and nonmonotone counterexample earlier on this page still apply.
SCoFT v4 §§9.2–9.3, pp. 177–181, motivates this distinction
[@taleb2025scoftv4].

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
  @davison1990models.
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  @coles2001introduction.
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  @embrechts1997modelling.

## Backlinks

- Depends on: [Pickands-Balkema-de Haan Theorem](./incerto-pickands-balkema-de-haan.md),
  [Generalized Pareto Distribution](./incerto-generalized-pareto.md),
  [Pareto Distribution](./incerto-pareto.md), and threshold notation in
  [Shared notation](https://xshi19.github.io/math/notation/).
- Used by: [S&P 500 Tail Diagnostics](./incerto-sp500-tail.md) and
  [Tail Threshold Selection](./incerto-tail-threshold-selection.md).
