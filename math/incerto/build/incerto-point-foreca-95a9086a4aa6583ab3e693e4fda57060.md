---
title: Point forecasts under heavy-tailed losses
options:
  concept:
    id: point-forecast
    type: method
    prerequisites: [pareto-moment-existence, calibration-payoff]
    related: [moment-estimation, ml-econometrics, generalized-central-limit-theorem, kappa, hill-estimator, ruin]
    tags: [forecasting, loss-functions, moments, pandemics]
---

A point forecast needs a target and a loss function. The mean, median, and
an event probability summarize different aspects of a distribution.
[Binary calibration](incerto-calibration-payoff.md) evaluates the last of
these after an event has been specified. A single number leaves the probability
of a large error and its consequences unspecified. Heavy tails make this
omission particularly costly: the sample mean may converge slowly, and an
average forecast score may lack the moments needed for its usual error bars.
This note treats SCoFT v4 chapter 11, §11.4, pp. 231–235, and chapter 12,
§§12.1–12.4, pp. 241–246 [@taleb2025scoftv4].

We prove the loss and transformation identities below, cite the general
stable-limit theorem, and label simulated sampling quantiles as estimates.
The pandemic discussion uses synthetic quantities; it does not estimate
COVID-19 fatalities or evaluate a particular intervention.

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
is finite. These are separate statements with separate domains.

(point-forecast-scaling)=
## Estimating a mean and predicting an outcome

Let $X_1,\ldots,X_n$ be iid training observations, with sample mean
$\bar X_n$. An independent future amount is denoted locally by $X_{n+1}$.
Even when the training mean consistently estimates $\mu$, the future
amount remains random. For a nondegenerate integrable law, Slutsky's theorem
gives $X_{n+1}-\bar X_n\Rightarrow X-\mu$: convergence of the estimate
does not collapse the predictive distribution. In the finite-variance case,
independence gives the exact comparison

$$
\begin{gathered}
\operatorname{Var}(\bar X_n)=\sigma^2/n,\\
\operatorname{Var}(X_{n+1}-\bar X_n)
=\sigma^2(1+1/n).
\end{gathered}
$$

The distinction matters before choosing a heavy-tail model. More training
data can reduce estimation error while leaving substantial outcome risk.

For an exact scaling benchmark, take iid symmetric stable observations
with location $\mu$, scale $\sigma>0$, and index $0<\alpha\le2$:

$$
\mathbb E e^{it(X-\mu)}
=e^{-|\sigma t|^\alpha}.
$$

Here $\mu$ denotes a location, and is a population mean only for
$\alpha>1$. At $\alpha=2$ the variance is $2\sigma^2$.
We prove the average-scale identity by multiplying characteristic
functions at $t/n$:

$$
\begin{gathered}
\mathbb E e^{it(\bar X_n-\mu)}
=e^{-n|\sigma t/n|^\alpha},\\
s_n=\sigma n^{1/\alpha-1},\\
\bar X_n-\mu\ \overset d=\ s_n Z_\alpha.
\end{gathered}
$$

$Z_\alpha$ is the zero-location unit-scale law. Every fixed interquantile
width scales by $s_n$, including when moments do not exist. For
$1<\alpha<2$ the width shrinks more slowly than $n^{-1/2}$; for the
Cauchy case $\alpha=1$ it stays constant; below one it increases.
Symmetric stable variables can be negative, so this benchmark is an error
model, not a literal fatality model.

Use [`stable_average_scale`](incerto-api-reference.md#api-stable-average-scale),
the existing [`symmetric_stable_mean_abs`](incerto-api-reference.md#api-symmetric-stable-mean-abs),
and [`stable_kappa`](incerto-api-reference.md#api-stable-kappa).
The scale exists throughout the grid; finite absolute error and kappa do not.

```python
from xmath.incerto.forecasting import stable_average_scale
from xmath.incerto.limits import symmetric_stable_mean_abs
from xmath.incerto.concentration import stable_kappa

for alpha in (.8, 1, 1.5, 2):
    scale = stable_average_scale(alpha, 100)
    mad = symmetric_stable_mean_abs(alpha, n=100)
    kappa = stable_kappa(alpha)
    print(f"alpha={alpha:g}: scale={scale:.3f}, MAD={mad:.3f}, kappa={kappa:.3f}")
```

**Output**

```text
alpha=0.8: scale=3.162, MAD=inf, kappa=nan
alpha=1: scale=1.000, MAD=inf, kappa=nan
alpha=1.5: scale=0.215, MAD=0.367, kappa=0.500
alpha=2: scale=0.100, MAD=0.113, kappa=0.000
```

The exact identity applies to stable parents. For iid exact positive
Pareto observations with $1<\alpha<2$, the
[stable-attraction theorem](incerto-generalized-central-limit-theorem.md#generalized-clt-statement)
instead gives the asymptotic statement

$$
\frac{\bar X_n-\mu}{x_m n^{1/\alpha-1}}
\Rightarrow Z_\alpha^+.
$$

The superscript distinguishes this asymmetric limit from the symmetric
benchmark; it does not assert positive support after centering. We cite,
rather than prove, attraction [@durrett2019probability, Theorem 3.8.2,
p. 186]. Durrett's fifth-edition convention normalizes the sum as
$(S_n-b_n)/a_n$, where $S_n=\sum_{i=1}^n X_i$, using

$$
\begin{gathered}
a_n=\inf\{x>0:{}\\
P(|X|>x)\le 1/n\},\\
b_n=n\mathbb E[X\mathbf1_{\{|X|\le a_n\}}].
\end{gathered}
$$

For exact Pareto, $a_n=x_m n^{1/\alpha}$ for $n\ge2$.
Our display is $(S_n-n\mu)/a_n$. When $1<\alpha<2$,
$(n\mu-b_n)/a_n=\alpha/(\alpha-1)$, so this full-mean normalization
gives Durrett's limit shifted downward by $\alpha/(\alpha-1)$.
This tail normalization fixes the scale and location of $Z_\alpha^+$.
For general regularly varying tails, the denominator is $a_n/n$,
where $nP(|X|>a_n)\to1$ and $a_n$ may contain a slowly varying factor.
Convergence in distribution by itself supplies neither finite-sample
coverage nor convergence of mean absolute errors.

For a nonnegative Pareto law with $0<\alpha<1$, use zero centering;
$\bar X_n/(x_m n^{1/\alpha-1})$ has a nondegenerate positive stable limit.
Here $b_n/a_n\to\alpha/(1-\alpha)$, so removing Durrett's truncated
centering shifts his limit upward by $\alpha/(1-\alpha)$.
There is no finite mean to forecast. In fact $\bar X_n\to\infty$ almost
surely when $\alpha\le1$: apply the bounded iid strong law to
$\min(X_i,b)$ for every integer $b$, then let its expectation increase
without bound. At the exact Pareto boundary $\alpha=1$, Durrett's constants
are $a_n=x_m n$ and $b_n=x_m n\log n$. His centered stable limit thus
applies to $\bar X_n/x_m-\log n$. The separate
[truncation argument](incerto-pareto.md#exact-pareto-boundary-calculation)
gives $\bar X_n/(x_m\log n)\to1$ in probability. This positive-law
boundary is different from the symmetric Cauchy location example.
At a regular-variation moment boundary, the slowly varying factor can
change integrability; these boundary conclusions use **exact Pareto** laws.

For $\alpha>2$, a Pareto mean has the ordinary finite-variance CLT.
At $\alpha=2$ its infinite variance requires a logarithmic adjustment
to Gaussian normalization. Neither case follows by extending the
$\alpha<2$ stable power formula. There is no universal impossibility
threshold at $\alpha=2.5$; a practical sample size requires a stated
loss, tolerance, body, scale, and dependence model. See
[kappa](incerto-kappa.md) for finite-sample absolute-error comparisons.

(point-forecast-scaling-figure)=
```{figure} ../assets/figures/incerto/point-forecast-scaling.svg
:alt: Stable average scales fall slowly, remain constant or grow according to alpha; repeated Pareto mean errors at alpha 1.5 approach a stable-width pattern under n to the one-third rescaling.

Top: exact unit-scale symmetric stable average scales, with both axes
logarithmic. Bottom: estimated 10th, 50th and 90th quantiles of
$n^{1/3}(\bar X_n-3)$ for iid Pareto $\alpha=1.5,x_m=1$.
There are 2,000 independent replications at each size
30/100/300/1000/3000, using PCG64 seeds 1200–1204. The shaded range
is sampling variation, not a confidence interval for an estimated exponent.
Finite simulation illustrates the cited limit and does not prove it.
```

(point-forecast-evaluation)=
## Why a forecast leaderboard can move

Evaluate a fixed forecast $d$ on iid test observations using the average
of $\ell(d,X_i)$. The earlier tail calculation gives exact Pareto loss
exponents $\alpha$ for absolute error and $\alpha/2$ for squared error.
Consequently:

| Score | Finite expected score | Finite score variance |
| --- | --- | --- |
| Absolute error | $\alpha>1$ | $\alpha>2$ |
| Squared error | $\alpha>2$ | $\alpha>4$ |
| Our sMAPE convention | Every $\alpha>0$ | Every $\alpha>0$ |

These boundaries follow by applying the Pareto moment test to the loss
and its square. If a nonnegative score has infinite expectation, its iid
average diverges almost surely by the same truncation argument. If its
mean is finite but its variance infinite, the strong law still applies;
the usual variance-based standard error lacks its population justification.
In particular, finite variance of $X$ is insufficient for a conventional
CLT for the **squared score**. An RMSE transformation does not restore a
missing population MSE.

Paired comparisons can behave better than either score separately.
For fixed finite forecasts $d_1,d_2$, the reverse triangle inequality gives

$$
\bigl||X-d_1|-|X-d_2|\bigr|
\le |d_1-d_2|.
$$

Thus their paired absolute-score difference is bounded even when both
expected absolute scores are infinite. Under iid evaluation it has the
ordinary finite-variance CLT, with a possible degenerate variance. For
squared losses, cancellation gives

$$
\begin{aligned}
&(X-d_1)^2-(X-d_2)^2\\
&\quad=2(d_2-d_1)X+d_1^2-d_2^2.
\end{aligned}
$$

For $d_1\ne d_2$, its mean exists under exact Pareto when $\alpha>1$,
and its variance is finite only when $\alpha>2$. A consistent strict paired
ranking is therefore possible when the expected loss difference is nonzero,
even when individual MSEs are infinite; it compares an integrable difference,
not two finite expected scores. For either paired comparison, a zero expected
loss difference is a tie in expectation, whatever the sample averages show.
Time-varying forecasts, training selection, dependence, and model changes
require their own analysis. These identities do not validate an arbitrary
leaderboard or establish that all forecasting comparisons fail.

Use [`pareto_forecast_scores`](incerto-api-reference.md#api-pareto-forecast-scores).
Each replication evaluates the fixed median forecast on fresh test data.
All samples and scores are retained; the helper raises on numerical overflow.

:::{dropdown} Code: repeated evaluation of one forecast
```python
import numpy as np
from xmath.incerto.forecasting import pareto_forecast_scores

for n in (30, 300, 3000):
    samples = pareto_forecast_scores(1.5, n, 2**(2/3),
                                     repetitions=2000, seed=1210+n)
    print(f"n={n}")
    for metric in ("absolute", "squared", "smape"):
        quantiles = np.quantile(samples[metric], [.1, .5, .9])
        print(f"  {metric}: " + ", ".join(f"{v:.3f}" for v in quantiles))
```

**Output**

```text
n=30
  absolute: 0.724, 1.264, 2.743
  squared: 1.410, 7.047, 62.107
  smape: 0.329, 0.411, 0.504
n=300
  absolute: 1.199, 1.542, 2.314
  squared: 8.512, 23.456, 170.817
  smape: 0.388, 0.415, 0.442
n=3000
  absolute: 1.485, 1.669, 2.035
  squared: 26.927, 62.111, 414.730
  smape: 0.406, 0.415, 0.424
```
:::

(point-forecast-evaluation-figure)=
```{figure} ../assets/figures/incerto/point-forecast-evaluation.svg
:alt: Across repeated Pareto test samples, absolute scores concentrate slowly, squared scores remain broad and drift upward, and bounded sMAPE scores concentrate.

Estimated median and 10th–90th sampling ranges of average scores for the
same fixed median forecast, Pareto $\alpha=1.5,x_m=1$. Each size
30/100/300/1000/3000 has 2,000 independent replications, PCG64 seed
$1210+n$. Absolute and squared panels have logarithmic score axes;
sMAPE has a linear score axis. Panels have different units. The dashed
absolute-risk reference is exact. Population squared risk is infinite;
every displayed squared score is a finite realization.
```

(point-forecast-quantiles)=
## Predictive ranges and tail payoffs

For $0<\tau<1$, define the quantile loss

$$
\ell_\tau(d,x)
=(\tau-\mathbf1_{\{x<d\}})(x-d).
$$

When $\mathbb E|X|<\infty$, the same slope argument used for absolute
loss proves

$$
\begin{aligned}
&\mathbb E\ell_\tau(d_2,X)
-\mathbb E\ell_\tau(d_1,X)\\
&\qquad=\int_{d_1}^{d_2}\{F(t)-\tau\}\,dt.
\end{aligned}
$$

The minimizers are precisely the $\tau$-quantiles satisfying
$F(d-)\le\tau\le F(d)$. Without integrability, quantiles still exist,
and expected **paired differences** remain finite because their magnitude
is at most $|d_2-d_1|$. The same integral then describes those differences;
it must not be read as subtraction of two infinite expectations.
A strict preference requires a nonzero expected paired loss difference;
zero means a tie in expectation, whatever the sample averages show.

For exact Pareto $X$, solving $F(q_\tau)=\tau$ gives

$$
q_\tau=x_m(1-\tau)^{-1/\alpha}.
$$

For $\alpha=0.8,x_m=1$, both a median and a 99th percentile are finite
despite the infinite mean. The interval between any two specified
quantiles has its stated probability under the continuous **known** model.
Fitting that model introduces additional uncertainty. Compare finite
quantiles using [`pareto_type1`](incerto-api-reference.md#api-pareto-type1)
and observed losses using [`quantile_loss`](incerto-api-reference.md#api-quantile-loss).

```python
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.forecasting import quantile_loss

law = pareto_type1(.8)
print("quantiles: " + ", ".join(f"{v:.2f}" for v in law.ppf([.5, .9, .99])))
print(f"mean: {law.mean():.2f}")
values = quantile_loss([1, 3, 20], 3, tau=.9)
print("observed losses: " + ", ".join(f"{v:.2f}" for v in values))
```

**Output**

```text
quantiles: 2.38, 17.78, 316.23
mean: inf
observed losses: 0.20, 0.00, 15.30
```

One quantile or one event probability still omits severity beyond the
threshold. For $X\ge0$, Tonelli's theorem gives, with infinite values allowed,

$$
\mathbb E[(X-u)_+]
=\int_u^\infty\bar F(t)\,dt.
$$

Thus the **whole survival curve** determines expected excess; its value
at one threshold does not. The fixed-threshold empirical fraction has
variance $p_u(1-p_u)/n$, where $p_u=P(X>u)$, regardless of the amount's
moments. Its RMSE is of order $n^{-1/2}$ when $0<p_u<1$; its MSE is
of order $n^{-1}$, not $o(n^{-1})$. For rare events its relative RMSE
is $\sqrt{(1-p_u)/(np_u)}$, which can be large. See
[counting exceedances](incerto-counting-exceedances.md) and
[tail beyond the sample](incerto-hidden-tail.md) for the proofs and limits.

(point-forecast-bounds)=
## A population cap and a log transform answer different questions

For fatalities in a fixed population over a fixed horizon, a deterministic
population bound $H$ implies $0\le X\le H$, hence every positive moment
is finite. An unbounded Pareto model can approximate an intermediate range,
but an exponent below one cannot establish an infinite physical mean under
that bound. Pooling events across changing populations also changes the model.

A cap alone need not make a sample informative about rare, severe losses.
Let $V$ be an unbounded exact Pareto variable and define locally
$X_H=\min(V,H)$, with $H\ge x_m$. The survival-integral calculation in
the [LLN note](incerto-lln-failure.md#lln-failure-truncated-mean-check) gives

$$
\mathbb E X_H
=x_m+x_m^\alpha\int_{x_m}^{H}t^{-\alpha}\,dt.
$$

Writing $h=H/x_m$, for $\alpha\ne1$ this is

$$
\mathbb E X_H
=x_m\left(1+\frac{h^{1-\alpha}-1}{1-\alpha}\right).
$$

At $\alpha=1$ it is $x_m(1+\log h)$.

For $\alpha<1$ this mean grows as a power of the remote cap. The iid
bounded strong law holds at every fixed $H$; no cap-independent sample
size follows. Capping creates an atom at $H$, and differs from conditioning
$V$ to fall below $H$.

Two logarithms have different purposes. If $V$ is exact Pareto, then
$\log(V/x_m)$ is exponential with rate $\alpha$, since its survival at
$z\ge0$ is $e^{-\alpha z}$. This explains why log spacings can support
[Hill estimation](incerto-hill-estimator.md) while the raw sample mean
is erratic. It does not recover raw expected harm by exponentiating a log
mean; nonlinear expectations require the full law.

For a bounded variable, an endpoint-removing *dual transform* is a separate
modeling device. At lower endpoint zero, write

$$
\begin{gathered}
Z=\phi_H(X)=-H\log(1-X/H),\\
X=\phi_H^{-1}(Z)=H(1-e^{-Z/H}).
\end{gathered}
$$

The forward map is increasing on $0\le X<H$, maps zero to zero,
and tends to infinity at $H$. Solving the equation gives its inverse.
For $X/H\to0$, Taylor expansion gives
$Z=X+X^2/(2H)+O(X^3/H^2)$, so ordinary-scale observations move little.
At the endpoint, use the extended value $\phi_H(H)=\infty$.
An endpoint atom remains an atom at infinity; finite dual-tail modeling
requires handling it explicitly.

An increasing transformation preserves quantiles, not means. To obtain
an expectation in physical units from an assumed dual law, evaluate
$\mathbb E[H(1-e^{-Z/H})]$, which is at most $H$, even if
$\mathbb E Z=\infty$. A heavy dual tail is an assumption to check,
not a consequence of the transformation. This is the zero-lower-endpoint
special case of the construction discussed by Cirillo and Taleb
[@cirillo2020pandemics, §II, p. 4]; no fitted shadow mean is computed here.

Use [`bounded_to_dual`](incerto-api-reference.md#api-bounded-to-dual)
and [`dual_to_bounded`](incerto-api-reference.md#api-dual-to-bounded).
The endpoint and invalid-input behavior are included.

:::{dropdown} Code: endpoint transformation and inverse
```python
from xmath.incerto.forecasting import bounded_to_dual, dual_to_bounded

for x in (0, 1, 50, 99, 100):
    z = bounded_to_dual(x, upper=100)
    back = dual_to_bounded(z, upper=100)
    print(f"x={x}: dual={z:.3f}, inverse={back:.3f}")
for x in (101, float("nan")):
    try:
        bounded_to_dual(x, upper=100)
    except ValueError as error:
        print(type(error).__name__ + ": " + str(error))
```

**Output**

```text
x=0: dual=0.000, inverse=0.000
x=1: dual=1.005, inverse=1.000
x=50: dual=69.315, inverse=50.000
x=99: dual=460.517, inverse=99.000
x=100: dual=inf, inverse=100.000
ValueError: values must lie in [0, upper]
ValueError: values must contain finite values
```
:::

(point-forecast-bounds-figure)=
```{figure} ../assets/figures/incerto/point-forecast-bounds.svg
:alt: The dual transform is near the identity far below the endpoint and rises steeply near it; three capped Pareto models with the same median have very different mean excesses as the cap increases.

Exact synthetic curves. Top: dimensionless dual coordinate $Z/H$ versus
$X/H$, with the identity for comparison; the endpoint itself is excluded.
Bottom: expected excess above $u=10$ for capped Pareto variables with
exponents 0.6/0.8/1.2 and the same median 2. Both axes are logarithmic.
The cap ranges from 20 to 100,000 in abstract loss units. This panel uses
capping, not a fitted dual model, and contains no pandemic observations.
```

(point-forecast-pandemics)=
## Pandemic tails, action, and conditional forecasts

SCoFT v4 §12.2, p. 242, refers to heavy tails in historical epidemic
fatalities. The underlying study fits historical event totals and a bounded
transformation [@cirillo2020pandemics, §§II–IV, pp. 2–6]. This is empirical
model evidence, not a theorem that each disease, country, or future wave
has an unbounded Pareto law. Event definitions, missing episodes, uncertain
death counts, changing population, and interventions matter when applying
that inference. A tail parameter estimated across historical events is not
automatically a time-series parameter within one outbreak.

**Data-source note.** The cited study describes its historical compilation
in §II and Table I. We do not import, redistribute, or refit that table.
Every numerical example on this page is synthetic.

Multiplicative growth can amplify parameter uncertainty without proving
a power law. For a deliberately simple model, fix $t>0$ and $x_0>0$,
and set $X=x_0e^{tR}$. If $R$ is exponential with rate $\lambda$,
then for $x\ge x_0$,

$$
\begin{aligned}
P(X>x)
&=P\!\left(R>\frac{\log(x/x_0)}{t}\right)\\
&=(x/x_0)^{-\lambda/t}.
\end{aligned}
$$

This proves a Pareto law with $\alpha=\lambda/t$ under the stated rate
model. A normal $R$ gives a lognormal amount instead; deterministic
exponential growth is not by itself mathematical chaos. The uncapped
model cannot remain a physical fatality law past a population endpoint.
Convexity gives $\mathbb E e^{tR}\ge e^{t\mathbb E R}$ when the
right-hand side is defined; an increase under uncertainty requires a
specified ordering, such as a mean-preserving spread. Variance alone
does not order arbitrary models, and a saturation cap changes convexity.

The decision question can be asked before one can estimate an average
accurately. In an abstract capped-loss model, define the harm
$J=(X_H-u)_+$ for $x_m\le u<H$. Pointwise subtraction of two caps gives

$$
J=\min(V,H)-\min(V,u).
$$

Suppose an action costs $c\ge0$ in the same loss units and, by assumption,
reduces this harm by a known fraction $r\in[0,1]$. Its expected total
loss is $c+(1-r)\mathbb E J$, compared with $\mathbb E J$ without
action. Subtraction proves the decision rule

$$
c<r\,\mathbb E J.
$$

This is a model-specific precaution calculation. It includes action costs
and effectiveness; the word “uncertainty” alone does not determine either.
Under a set of credible models, the sufficient condition
$c<r\inf\mathbb E J$ makes the action preferable in every model in
that set. A catastrophe-probability constraint is another possible
criterion; it requires an explicitly chosen tolerance.

Use the existing [`pareto_capped_mean`](incerto-api-reference.md#api-pareto-capped-mean).
The models share median 2 but disagree sharply about capped tail harm.
The action parameters below are hypothetical loss-unit choices, not
clinical effects or monetary values assigned to deaths.

:::{dropdown} Code: same median, different tail harm
```python
from xmath.incerto.limits import pareto_capped_mean

upper, threshold, cost, reduction = 10000, 10, 5, .5
for alpha in (.6, .8, 1.2):
    xm = 2 * 2**(-1/alpha)
    harm = (pareto_capped_mean(upper, alpha, x_m=xm)
            - pareto_capped_mean(threshold, alpha, x_m=xm))
    benefit = reduction * harm
    print(f"alpha={alpha:g}: excess={harm:.2f}, benefit={benefit:.2f}, "
          f"act={benefit > cost}")
```

**Output**

```text
alpha=0.6: excess=70.67, benefit=35.33, act=True
alpha=0.8: excess=20.57, benefit=10.28, act=True
alpha=1.2: excess=2.71, benefit=1.36, act=False
```
:::

Finally, a forecast must state the action regime it assumes. Denote the
future amount under action $a$ by $X(a)$ and available information by
$\mathcal I$. The predictive object is

$$
P(X(a)\le x\mid\mathcal I).
$$

An observed outcome after action $a=1$ does not directly test a forecast
conditional on $a=0$. It also does not prove that the action caused the
difference: the unobserved counterfactual needs a causal design or model.
Policy feedback changes which forecast is being tested; it does not make
all evaluation impossible. SCoFT v4 §§12.3–12.4, pp. 245–246, motivates
this distinction. No country comparison in the source is used here as a
causal estimate.

The [chapter 11 guide](incerto-scoft-ch11.md) maps the original score
comparison; the [chapter 12 guide](incerto-scoft-ch12.md) maps the broader
forecasting and pandemic discussion. Reproduce the earlier loss figure with
`uv run python demos/incerto/calibration_payoffs.py` and the three new
figures with `uv run python demos/incerto/point_forecasts.py`.
