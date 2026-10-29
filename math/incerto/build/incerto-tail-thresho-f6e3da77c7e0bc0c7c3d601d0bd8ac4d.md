---
title: Tail Threshold Selection
options:
  concept:
    id: tail-threshold-selection
    type: method
    depends_on:
      - extreme-value-index
      - hill-estimator
      - mean-excess-function
      - generalized-pareto
    tags:
      - estimation
      - diagnostics
      - tail-risk
---

## Statement

Tail threshold selection is the modeling step that chooses where the body of a
sample ends and where a tail estimator or peaks-over-threshold model begins.
For a threshold $u$, the exceedance sample is

$$
\{x_i-u:x_i>u\}.
$$

For a [Hill estimator](./incerto-hill-estimator.md), the equivalent tuning parameter is
the number $k$ of upper order statistics, with empirical threshold
$x_{n-k:n}$.  Hill is designed for positive, regularly varying right tails
with extreme-value index $\xi>0$; for a Pareto-type tail,
$\alpha=1/\xi$.  A threshold selection report should include at least:

- the candidate thresholds or $k$ values;
- the number of exceedances retained;
- Hill or EVI stability over the candidate range;
- mean-excess or GPD-shape stability where relevant;
- the reason the selected range was kept or rejected.

The goal is not to find a universally correct threshold.  It is to make the
bias-variance tradeoff visible.

Here $x_{1:n}\le\cdots\le x_{n:n}$ are the sorted observations,
$k$ counts the upper values used by Hill, and $\xi$ is the
extreme-value index. Ties at the threshold can make the strict exceedance
count differ from $k$ and should be reported explicitly.

## Bias-variance intuition

Every tail model asks the same awkward question: how far out is "tail"?  Set
the threshold too low and the estimator is biased by body observations.  Set
it too high and the estimator is dominated by too few extremes.  A good
threshold analysis therefore looks for a region where several diagnostics stop
moving violently while enough exceedances remain to estimate anything at all.

This page is distinct from the
[body-shoulder-tail diagnostic](./incerto-body-shoulder-tail.md).  Body/shoulder geometry
describes how a distribution's density responds to variance mixing.  Threshold
selection is an empirical modeling decision for tail estimation.

## Diagnostics

For a one-sided positive sample:

1. Choose a grid of candidate thresholds, often empirical quantiles.
2. Record exceedance counts at each threshold.
3. Plot Hill estimates over $k$ or threshold values.
4. Plot the [mean-excess function](./incerto-mean-excess-function.md) for the
   same threshold range.
5. If using a GPD model, fit shape and scale over multiple thresholds.
6. Prefer a range where estimates are reasonably stable and exceedance counts
   are not too small.

The accepted range should be reported, not hidden.  Downstream quantities such
as [moment existence](./incerto-pareto-moment-existence.md), return levels, or
expected shortfall can be highly sensitive to $\xi$. Under an exact GPD model,
finite variance requires $\xi<1/2$, while finite mean and expected shortfall
require $\xi<1$. For a general regularly varying tail, moments at the boundary
also depend on the slowly varying factor. High return-level estimates are
sensitive to $\xi$ but do not have mathematical singularities specifically
at those two values.

## Static body-plus-tail example

(tail-threshold-selection-diagnostics)=
Consider the population model from the source example: with probability
$0.82$, $X$ is uniform on $[1,3]$; with probability $0.18$, it is Pareto
with cutoff $3$ and exponent $\alpha=1.7=17/10$. Thus

$$
\begin{gathered}
\bar F(u)=\\
\begin{cases}
1, & u<1,\\
\dfrac{0.82(3-u)}2+0.18, & 1\le u<3,\\
0.18(3/u)^{1.7}, & u\ge3.
\end{cases}
\end{gathered}
$$

Above $3$, the conditional tail is exactly Pareto. Its shape is $\xi=10/17$,
its excess scale is $\beta(u)=10u/17$, and its mean excess is
$e(u)=u/(\alpha-1)=10u/7$. For an iid sample of size $n=30{,}000$,
the expected number of exceedances is $n\bar F(u)$.

| Threshold $u$ | Expected exceedances (rounded) | Exact tail mean excess | Modified scale $\beta(u)-\xi u$ |
| --- | --- | --- | --- |
| $2$ | $17{,}700$ | Body and tail both contribute | GPD tail formula does not apply |
| $3$ | $5{,}400$ | $30/7$ | $0$ |
| $6$ | $1{,}662$ | $60/7$ | $0$ |
| $12$ | $512$ | $120/7$ | $0$ |

These counts are expectations under a specified population, not realized sample
counts or fitted estimates. At $u=2$, the exact mean excess instead is

$$
\begin{aligned}
e(2)&=\frac{0.205+0.18(51/7-2)}{0.59}\\
&\approx1.960,
\end{aligned}
$$

where $0.205$ is the body contribution to $\mathbb E[(X-2)_+]$ and $51/7$
is the conditional tail mean. Extending the Pareto line below its cutoff
would incorrectly give $20/7$.

At thresholds at least $3$, raising $u$ introduces no tail-model bias in this
exact construction but still loses observations. In empirical data, the cutoff
is unknown and there may be no exact Pareto region at any finite threshold.
The table therefore does not select an empirical threshold or report an
uncertainty interval.

For a general GPD tail, threshold stability is
$\beta(u')=\beta(u)+\xi(u'-u)$. Hence shape and modified scale
$\beta(u)-\xi u$ should be inspected together. The simpler relation
$\beta(u)/u=\xi$ is specific to the exact Pareto tail. Selected-threshold
uncertainty, dependence, and tail approximation error are additional to the
sampling uncertainty of a fit at a fixed threshold.

(threshold-synthetic-figures)=
## Coordinated diagnostics on one sample

```{figure} ../assets/figures/incerto/threshold-diagnostics.svg
:alt: Three vertically aligned panels show strict survival, Hill index and mean excess for one mixture sample as the threshold rises.

One iid sample of $30{,}000$ from the model above, PCG64 seed 20261021.
Every panel uses the same threshold axis; dotted lines mark the known
population join $u=3$. Thresholds are observed order statistics near a
fixed grid, so Hill uses the same baseline as the strict exceedance count.
Dashed references are population quantities; the exact Pareto references
start at $3$. The mean-excess and fitted-parameter curves are masked below
50 exceedances, with the excluded threshold region shaded. Counts remain
visible in the companion GPD figure below; numerical fit failures would
appear there as crosses.
The survival panel uses strict empirical survival and omits its zero endpoint
on log axes. No threshold is selected by the plot.
```

```{figure} ../assets/figures/incerto/threshold-gpd.svg
:alt: Three aligned GPD panels show fitted shape, modified scale and strict exceedance counts, retaining counts where fits are masked below fifty observations.

The same sample, thresholds and dotted join at $u=3$. Shape and modified
scale have population references only above the join. Counts below 50 remain
visible even when fits are masked. The count-panel title reports actual
eligible-fit status: all eligible fits succeeded for this seed; four other
thresholds have too few exceedances to fit. Numerical success is not
validation of the parent model.
```

The GPD fits hold excess location at zero and check numerical convergence,
positive scale and support for every excess. We plot both fitted shape
and modified scale $\widehat\beta(u)-\widehat\xi(u)u$. Success of these
checks means the numerical fit is usable for the display; it does not
establish a global likelihood optimum or validate the model.

```{figure} ../assets/figures/incerto/threshold-pp.svg
:alt: GPD probability-probability checks at thresholds 2, 3 and 12 compare fitted CDF values with empirical plotting positions, with counts shown for each threshold.

The same sample at three fixed thresholds. Each panel places fitted GPD
CDF values at the ordered excesses on the horizontal axis and $i/(m+1)$
on the vertical axis, where $m$ is that panel's exceedance count and
$i=1,\ldots,m$. These are plotting positions,
not strict survival probabilities. Parameters are fitted on the displayed
data, so proximity to the diagonal is an in-sample diagnostic with no
confidence-band interpretation. The body affects the fit at $u=2$;
raising the threshold loses observations. Agreement or a plateau across
these checks does not single out a unique threshold.
```

Use [`pareto_type1`](incerto-api-reference.md#api-pareto-type1) and
[`gpd_threshold_fit`](incerto-api-reference.md#api-gpd-threshold-fit).
This reproduces the three fixed-threshold fits without writing figures:

:::{dropdown} Code

```python
import numpy as np
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.estimators import gpd_threshold_fit

rng = np.random.default_rng(20261021)
tail = rng.random(30000) < 0.18
x = rng.uniform(1, 3, 30000)
x[tail] = pareto_type1.rvs(1.7, scale=3, size=tail.sum(), random_state=rng)
for u in (2, 3, 12):
    fit = gpd_threshold_fit(x, u, min_exceedances=50)
    print(f"u={u:d}: count={fit['count']:d}, status={fit['status']}")
    print(f"  xi={fit['xi']:.4g}, beta={fit['beta']:.4g}, modified_scale={fit['modified_scale']:.4g}")
```

**Output**

```text
u=2: count=17596, status=ok
  xi=0.5065, beta=0.8523, modified_scale=-0.1608
u=3: count=5344, status=ok
  xi=0.5929, beta=1.744, modified_scale=-0.03471
u=12: count=501, status=ok
  xi=0.5553, beta=7.346, modified_scale=0.683
```
:::

Regenerate both figures with `uv run python demos/incerto/tail_estimation.py`.
The runner also exports the Hill and mean-excess comparisons; its fixed
sample sizes and fit-iteration caps bound the work.

## Caveats

- Threshold selection is a modeling judgment, not a theorem.  Different
  diagnostics can disagree.
- The Hill estimator is a right-tail regular-variation tool for $\xi>0$, not a
  general estimator for Gumbel-type or bounded-tail domains.
- The ratio $\hat\beta(u)/u$ is an exact-Pareto diagnostic.  For a general GPD
  threshold model, threshold stability is
  $\beta(u')=\beta(u)+\xi(u'-u)$, so $\beta(u)-\xi u$ is the stable quantity.
- Very high thresholds may reduce approximation bias but can leave too few
  exceedances for stable estimation.
- Dependence, volatility clustering, rounding, reporting limits, and
  truncation can all make threshold diagnostics misleading.
- A selected threshold for one sample window or horizon should not be reused
  mechanically for a different dataset.
- Thresholds for right-tail positive data do not automatically apply to
  two-sided returns or losses without an explicit transformation.

## References

- Hill, "A Simple General Approach to Inference About the Tail of a
  Distribution" @hill1975simple.
- Davison and Smith, "Models for Exceedances over High Thresholds"
  @davison1990models.
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  @coles2001introduction.
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  @embrechts1997modelling.

## Backlinks

- Depends on: [Extreme-Value Index Estimation](./incerto-extreme-value-index.md),
  [Hill Estimator](./incerto-hill-estimator.md),
  [Mean Excess Function](./incerto-mean-excess-function.md), and
  [Generalized Pareto Distribution](./incerto-generalized-pareto.md).
- Related: [Body, Shoulders, and Tails](./incerto-body-shoulder-tail.md) is a density
  geometry diagnostic, not a threshold-selection rule.
- Used by: [S&P 500 Tail Diagnostics](./incerto-sp500-tail.md).
