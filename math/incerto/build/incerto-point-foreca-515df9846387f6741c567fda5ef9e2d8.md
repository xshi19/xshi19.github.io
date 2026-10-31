---
title: Point forecasts under heavy-tailed losses
options:
  concept:
    id: point-forecast
    type: method
    prerequisites: [pareto-moment-existence, calibration-payoff]
    related: [moment-estimation, ml-econometrics]
    tags: [forecasting, loss-functions, moments]
---

A point forecast needs a target and a loss function. The mean, median, and
an event probability summarize different aspects of a distribution.
[Binary calibration](incerto-calibration-payoff.md) evaluates the last of
these after an event has been specified. This note develops the amount-score
comparisons needed for SCoFT v4 chapter 11, §11.4, pp. 231–235
[@taleb2025scoftv4]. It makes no empirical claim about forecasting contests
or pandemics. Broader forecasting questions belong to the chapter 12 treatment.

Write $d\in\mathbb R$ for a fixed point forecast and $X$ for the future
amount. We use $\ell(d,X)$ for a loss, leaving $L$ for the slowly varying
function in the [shared notation](https://xshi19.github.io/math/notation/).
The expected loss $\mathbb E\ell(d,X)$ evaluates the action $d$ under a
specified probability law. Estimating that law from observations is a
further problem.

(point-forecast-targets)=
## Which point minimizes expected loss?

We prove the squared- and absolute-loss claims at their finite-risk scope.
If $\mathbb E X^2<\infty$, expansion around $\mu=\mathbb E X$ gives

$$
\mathbb E[(X-d)^2]
=\operatorname{Var}(X)+(d-\mu)^2.
$$

The cross term vanishes because $\mathbb E(X-\mu)=0$. The mean is the
unique minimizing forecast. Finite variance of the *sample mean* is not
needed to define a population median, and existence of a population mean
alone does not make the squared-loss objective finite.

For absolute loss, assume $\mathbb E|X|<\infty$. For $d_2>d_1$,
integrate the pointwise slopes of $|x-d|$ in $d$. Their magnitude is at
most one, so interchanging expectation and this finite integral is valid.
With $F$ the CDF, the result is

$$
\begin{aligned}
&\mathbb E|X-d_2|-\mathbb E|X-d_1|\\
&\qquad=\int_{d_1}^{d_2}\{2F(t)-1\}\,dt.
\end{aligned}
$$

The value of a slope exactly at an atom does not affect the integral.
Monotonicity of $F$ shows that the minimizing points are precisely the
medians: $F(d-)\le1/2\le F(d)$. This covers nonunique medians and atoms.
A median is a property of any real probability distribution; the preceding
finite-expected-loss optimization requires the additional integrability
assumption.

(point-forecast-moments)=
## A finite point does not remove an infinite expected loss

For exact Pareto $X$ with $x_m>0$, the
[moment theorem](incerto-pareto-moment-existence.md) gives a sharp
boundary. For each fixed finite $d$,

$$
\begin{gathered}
\mathbb E|X-d|<\infty
\quad\Longleftrightarrow\quad\alpha>1,\\
\mathbb E[(X-d)^2]<\infty
\quad\Longleftrightarrow\quad\alpha>2.
\end{gathered}
$$

To prove necessity, for $X>2|d|$ the absolute error is at least $X/2$.
The tail integral of the corresponding power diverges at and below the
moment boundary. For sufficiency use $|X-d|\le X+|d|$ and
$(X-d)^2\le2X^2+2d^2$. Thus every finite point has infinite squared
risk when $1<\alpha\le2$, even though a finite mean exists.
One can compare suitable differences of losses under weaker conditions,
but that is a different optimization formulation; it is not a finite
minimum of the ordinary squared-risk objective.

For $\alpha>1$ the exact Pareto mean and median are

$$
\mu=\frac{\alpha x_m}{\alpha-1},
\qquad d_{1/2}=x_m2^{1/\alpha}.
$$

The median formula follows from $F(d_{1/2})=1/2$.
For $d\ge x_m$, use $|X-d|=d-X+2(X-d)_+$ and the
[excess-mean formula](incerto-calibration-payoff.md#calibration-tail-models)
to obtain

$$
\mathbb E|X-d|
=d-\mu+\frac{2x_m^\alpha d^{1-\alpha}}{\alpha-1}.
$$

This finite expression permits an exact comparison; a finite Monte Carlo
average cannot prove that an expected loss exists.

Use [`pareto_tail_bet`](incerto-api-reference.md#api-pareto-tail-bet) for
the excess mean and [`point_forecast_loss`](incerto-api-reference.md#api-point-forecast-loss)
for observed errors. The simulation uses exact Pareto draws at
$\alpha=1.5,x_m=1$, PCG64 seed 1110, and 20,000 observations.
It has a finite absolute-risk target and an infinite squared-risk target.

:::{dropdown} Code: exact absolute risk and finite simulated losses
```python
import numpy as np
from xmath.incerto.calibration import pareto_tail_bet, point_forecast_loss

alpha, xm = 1.5, 1.0
mean = alpha * xm / (alpha - 1)
median = xm * 2**(1 / alpha)
rng = np.random.default_rng(1110)
x = xm * (1 + rng.pareto(alpha, 20000))
for label, d in (("median", median), ("mean", mean)):
    excess = pareto_tail_bet(alpha, d, xm=xm)["excess_mean"]
    risk = d - mean + 2 * excess
    print(f"{label}: point={d:.3f}, exact absolute risk={risk:.3f}")
    print(f"  sample absolute={point_forecast_loss(x, d).mean():.3f}")
    squared = point_forecast_loss(x, d, metric="squared").mean()
    print(f"  sample squared={squared:.3f}; population squared=inf")
```

**Output**

```text
median: point=1.587, exact absolute risk=1.762
  sample absolute=1.702
  sample squared=80.337; population squared=inf
mean: point=3.000, exact absolute risk=2.309
  sample absolute=2.252
  sample squared=78.520; population squared=inf
```
:::

The [moment-estimation note](incerto-moment-estimation.md) distinguishes
existence of an estimand from finite variance of its estimator. That
distinction also applies to losses: iid averages of integrable absolute
errors can be consistent even when their sampling variance is infinite.

(point-forecast-metrics)=
## Normalization changes the score's tail

Absolute error, squared error, MASE, and sMAPE are different functions of
the same forecast and outcome. For a fixed finite $d$ and sufficiently
large $t$, exact Pareto $X$ satisfies

$$
\begin{gathered}
P(|X-d|>t)=\bar F(t+d),\\
P((X-d)^2>t)=\bar F(\sqrt t+d).
\end{gathered}
$$

For the first identity, $t>|d|$ eliminates the lower branch $X<d-t$;
for the second use $\sqrt t>|d|$. Once the remaining thresholds exceed
$x_m$, the Pareto formula applies. Taking ratios as $t\to\infty$
shows tail exponents $\alpha$ and $\alpha/2$, respectively.
Squaring the error changes its moment requirements.

MASE divides absolute error by a positive training-sample scale $s_0$
[@hyndman2006accuracy, author manuscript §3, pp. 12–13].
For a fixed realized $s_0>0$, it preserves the absolute error's tail
exponent. If that scale is random, a conditional statement given training
data does not automatically extend to the unconditional ratio: values
near zero can change its moments. A constant training series can give
$s_0=0$, requiring an explicit policy before scoring.

For a fixed training vector $y_1,\ldots,y_m$ and seasonal lag $h<m$,
a conventional scale is the mean of $|y_j-y_{j-h}|$ over $j=h+1,\ldots,m$.
It is a naive forecast's training error, not the dispersion of future
observations around their sample mean. The helper accepts the scale
explicitly, so the chosen lag and training window remain visible.

Several denominator conventions have been called sMAPE
[@hyndman2014smape]. The convention used here is

$$
\begin{gathered}
\ell_{\rm s}(d,x)=\frac{2|x-d|}{|x|+|d|},\\
|x|+|d|>0,\\
\ell_{\rm s}(0,0)=0.
\end{gathered}
$$

The triangle inequality proves $0\le\ell_{\rm s}\le2$. A percentage
presentation multiplies this by 100. Thus this normalized score is bounded
even for unbounded $X$. A statement that every M4-style score has the
same tail class as the underlying amount would be false for this sMAPE.
The source's §11.4 formula also needs absolute values in its numerator to
describe either of these absolute-error metrics: signed errors can cancel.
The chosen zero convention is stated explicitly, rather than evaluating
$0/0$.

Use [`point_forecast_loss`](incerto-api-reference.md#api-point-forecast-loss).
The outputs show how bounded normalization can hide changes in magnitude.

```python
from xmath.incerto.calibration import point_forecast_loss

for metric in ("absolute", "squared", "smape"):
    values = point_forecast_loss([0, 10, 10000], 0, metric=metric)
    print(metric + ": " + ", ".join(f"{v:.2f}" for v in values))
try:
    point_forecast_loss([1, 2], 1, metric="mase", scale=0)
except ValueError as error:
    print(type(error).__name__ + ": " + str(error))
```

**Output**

```text
absolute: 0.00, 10.00, 10000.00
squared: 0.00, 100.00, 100000000.00
smape: 0.00, 2.00, 2.00
ValueError: MASE requires a positive finite training scale
```

(point-forecast-loss-figure)=
```{figure} ../assets/figures/incerto/point-forecast-losses.svg
:alt: Absolute and squared error grow with a Pareto loss amount, while sMAPE approaches two; a lower panel shows exact absolute risk minimized at the median.

Exact model curves, with $\alpha=1.5,x_m=1$. Top: the same median forecast
is scored by absolute error, squared error, and sMAPE; both axes are
logarithmic and the grid starts strictly above the median. These scores
have different units and are not ranked by their vertical positions.
Bottom: exact expected absolute loss for $1\le d\le5$, with median and
mean marked. Expected squared loss is infinite for every $d$ in this
model and is not drawn as a finite curve.
```

(point-forecast-decision)=
## Forecast accuracy and the decision

Changing the score can change the preferred point and the sampling
uncertainty. It does not by itself specify a useful action. A probability
forecast may guide a threshold decision; an amount forecast may guide
capacity; an explicit nonlinear payoff may require several parts of the
predictive distribution. The [hinge representation](incerto-ml-econometrics.md#pipeline-hinges)
and [payoff identity](incerto-calibration-payoff.md#calibration-nonlinear)
show how to evaluate a stated exposure when the required expectations
exist. None of these arguments says that every point forecast of a
heavy-tailed variable is meaningless.

For an event score, reporting the conditional probability minimizes expected
Brier loss. For absolute amount error, a median minimizes finite expected
loss. For squared amount error, the mean does so when the second moment
is finite. These are separate statements with separate domains. Quantile
losses, predictive ranges, and the broader chapter 12 argument will extend
this page at their own scope.

The [chapter 11 guide](incerto-scoft-ch11.md) gives the source map.
Reproduce the model figure with
`uv run python demos/incerto/calibration_payoffs.py`.
