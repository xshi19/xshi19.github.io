---
title: Moment estimation and its limits
options:
  concept:
    id: moment-estimation
    type: method
    prerequisites:
      - pareto-moment-existence
      - plug-in-tail-estimation
    related:
      - kappa
      - mixture-uncertainty
      - dispersion-ratio
      - lln-preasymptotic
    tags:
      - estimation
      - moments
      - finite-samples
---

A finite sample produces a finite empirical moment in exact arithmetic.
It does not establish that the corresponding population moment exists.
Even when that moment exists, the estimator can have infinite variance or
be dominated by events that most samples miss. We separate these questions
using an exact Pareto model and a finite Gaussian mixture.

The sample notation follows the [shared canon](https://xshi19.github.io/math/notation/).
For iid $X_i\ge0$ and order $q>0$, write

$$
\begin{aligned}
m_q&=\mathbb E X^q,\\
\widehat m_{q,n}&=\frac1n\sum_{i=1}^nX_i^q.
\end{aligned}
$$

These are raw moments. For a signed variable, replace $X^q$ by $|X|^q$;
a signed odd moment requires separate integrability conditions.

(moment-estimation-gates)=
## The estimator needs a higher moment

For [Pareto](incerto-pareto-moment-existence.md) with $\alpha,x_m>0$,

$$
\begin{aligned}
m_q&=\frac{\alpha x_m^q}{\alpha-q},\\
&\hspace{2em}q<\alpha,\\
m_q&=\infty,\\
&\hspace{2em}q\ge\alpha.
\end{aligned}
$$

Assume $q<\alpha$. Then the sample moment is unbiased for every $n\ge1$,
but it has finite variance **only if $2q<\alpha$**. In that regime,

$$
\begin{aligned}
n\operatorname{Var}(\widehat m_{q,n})
&=\frac{\alpha x_m^{2q}}{\alpha-2q}\\
&\quad-\left(\frac{\alpha x_m^q}{\alpha-q}\right)^2.
\end{aligned}
$$

We prove both assertions. Linearity gives the expectation whenever $m_q$
is finite. For $2q<\alpha$, independence makes the cross-covariances zero.
For $q<\alpha\le2q$, nonnegativity gives
$\widehat m_{q,n}^2\ge X_1^{2q}/n^2$, whose expectation is infinite.
Since the estimator has finite mean, its variance is infinite too. Equality
at $2q=\alpha$ is excluded by the logarithmic divergence of the Pareto
moment integral.

For example, $\alpha=3,x_m=1$ gives $m_2=3$, but
$\widehat m_{2,n}$ has infinite variance at every finite $n$.
At $\alpha=6$, $m_2=3/2$, $m_4=3$, and the estimator variance is
$3/(4n)$. A simulated finite standard error in the first model cannot
certify the population variance of the estimator.

### Consistency does not require estimator variance

We prove convergence in probability when $m_q<\infty$, without a finite
$2q$th moment. Set $T_i=X_i^q$ and $T_i^{(B)}=\min(T_i,B)$.
For fixed $B$, Chebyshev's inequality gives convergence of the average of
$T_i^{(B)}$ to its expectation, because these variables are bounded and iid.
The omitted nonnegative part has expected average
$\mathbb E(T_1-B)_+\to0$ as $B\to\infty$. Markov's inequality controls
that average uniformly in $n$. For any error tolerance, first choose $B$
to make this expectation small, then $n$ to control the bounded average.
The triangle inequality proves $\widehat m_{q,n}\to m_q$ in probability.
This argument supplies no uniform rate over all integrable laws.

When $m_q=\infty$, the stronger almost-sure divergence follows from the
[nonnegative infinite-mean theorem](incerto-lln-failure.md), applied to
$X_i^q$. That linked note proves the result by truncation. Finite observed
values and floating-point overflow are separate matters; neither determines
which population regime applies.

(moment-estimation-root)=
## A square root changes the moment question

Let $R_n=\sqrt{\widehat m_{2,n}}$. If $m_2<\infty$, then

$$
\begin{gathered}
\mathbb E R_n^2=m_2,\\
\mathbb E R_n\le\sqrt{m_2},\\
\operatorname{Var}(R_n)<\infty.
\end{gathered}
$$

The identity is immediate from the definition. Cauchy–Schwarz gives the
inequality and integrability of $R_n$; hence its variance is finite.
Thus infinite variance of the empirical **second moment** does not imply
infinite variance of the empirical **root mean square**. For the sample
standard deviation $s_n^2=n^{-1}\sum(X_i-\bar X_n)^2$, the identity
$s_n^2=\widehat m_{2,n}-\bar X_n^2\le\widehat m_{2,n}$ gives the same
finite-second-moment conclusion for $s_n$. A fixed $n/(n-1)$ correction
for $n>1$ does not change this conclusion.

The square root can introduce downward bias through concavity. The sample
raw moment remains unbiased. Bias, estimator variance, and the frequency
of an underestimate are different properties.

(moment-estimation-normal-comparison)=
## A comparison under an exact Gaussian model

Here we know the center is zero and assume iid $X_i\sim N(0,\sigma^2)$,
with $\sigma>0$. Compare the root mean square $R_n$ with the
Gaussian-calibrated mean absolute deviation
$A_n=\sqrt{\pi/2}\,n^{-1}\sum|X_i|$. Both target $\sigma$ under these
assumptions. We can compute their mean squared errors without simulation.

The normal absolute-moment integrals give
$\mathbb EA_n=\sigma$ and
$\operatorname{Var}(A_n)=\sigma^2(\pi/2-1)/n$.
For $T=\sum(X_i/\sigma)^2$, changing variables from a normal density gives
a gamma shape $1/2$, scale 2 for each square. Independence and the
[gamma convolution identity](incerto-tail-class-catalog.md#exact-light-tail-checks)
give shape $n/2$, scale 2 for $T$. Integrating $\sqrt T$ against that
density yields

$$
\begin{aligned}
b_n&=\frac{\mathbb ER_n}{\sigma}\\
&=\sqrt{\frac2n}\frac{\Gamma((n+1)/2)}{\Gamma(n/2)},\\
\frac{\mathbb E(R_n-\sigma)^2}{\sigma^2}
&=2(1-b_n).
\end{aligned}
$$

The last identity uses $\mathbb ER_n^2=\sigma^2$. The corresponding MSE
for $A_n$, divided by $\sigma^2$, is $(\pi/2-1)/n$.
These are exact finite-$n$ risks under squared-error loss, including
$R_n$'s bias. The demo prints them for $n=10,100,1000$.

This comparison does not carry over unchanged to the rare-state mixture:
$\mathbb E|X|=\sqrt{2/\pi}\,\mathbb E\sqrt Y$, so the same $A_n$ now
targets $\mathbb E\sqrt Y$, whereas the population standard deviation is
$\sqrt{\mathbb EY}$. Jensen's inequality makes these unequal for
nonconstant $Y$ with finite mean. Changing the model can change both the
sampling error and the target of a calibrated statistic.

(moment-estimation-rare-state)=
## Finite moments that most samples miss

Use the [two-state variance mixture](incerto-mixture-uncertainty.md#mixture-finite-states)
with $P(Y=0.5)=0.99$, $P(Y=50.5)=0.01$, and
$X=\sqrt YZ$, with independent standard-normal $Z$.
Draw the pairs $(Y_i,Z_i)$ independently. The population second moment is 1
and fourth moment is 77.25, so

$$
\operatorname{Var}\left(\frac1n\sum X_i^2\right)=\frac{76.25}{n}.
$$

The probability that no wide component is selected is $0.99^n$.
Conditional on that event, all observations are independent $N(0,0.5)$
and the empirical second moment has expectation 0.5. At $n=50$, this event
has probability about 0.605. Rare samples compensate in the unconditional
mean; the raw second moment estimator is still unbiased.

```{figure} ../assets/figures/incerto/finite-moment-estimation.svg
:alt: Two vertically stacked plots show 10th, median and 90th percentiles of the empirical second moment for Gaussian, rare-state mixture, Pareto three and Pareto six samples, relative to their true moments.

Independent repeated samples, 600 per law and sample size. Lines show
medians; bands show the 10th–90th percentiles of the estimator divided by
its population target. These are sampling-distribution summaries, not
confidence bands for a fitted model or Monte Carlo confidence intervals.
Every population second moment here is finite; Pareto with exponent 3
has an infinite fourth moment. The finite mixture also converges slowly
on this grid, despite having every moment.
```

(moment-estimation-plugin)=
## Fitting a model introduces another boundary

The [known-cutoff Pareto plug-in](incerto-plug-in-tail-estimation.md#plugin-caveats)
uses $\widehat\xi_n=n^{-1}\sum\log(X_i/x_m)$.

Use [`pareto_mean_plugin`](incerto-api-reference.md#api-pareto-mean-plugin).

```python
from xmath.incerto.estimators import pareto_mean_plugin

for sample in ([1, 1], [1, 2], [4, 4]):
    print(f"{sample} {pareto_mean_plugin(sample, x_m=1):.4g}")
```

**Output**

```text
[1, 1] 1
[1, 2] 1.53
[4, 4] inf
```

Under an exact iid Pareto model with $\alpha>1$, the log-excess average has
a gamma density that is positive and continuous at 1. Consequently even
conditional on a finite fit,

$$
\begin{gathered}
\mathbb E\left[\frac{x_m}{1-\widehat\xi_n}
\;\middle|\;\widehat\xi_n<1\right]\\
=\infty.
\end{gathered}
$$

The linked note derives that density. Its positive lower bound on a small
left neighborhood of 1 makes the expectation at least a positive multiple
of $\int_{1-\delta}^1(1-t)^{-1}\,dt=\infty$. The squared estimator diverges
there as well. A concentrated central part of a sampling distribution can
coexist with divergent moments. Simulations must count non-finite fits;
removing them does not remove the singularity of the remaining distribution.
No universal ranking of the sample mean and this plug-in follows.

(moment-estimation-power-means)=
## Rising power means and different targets

For fixed observations and positive weights $w_i$ summing to 1, define
$A_p=(\sum_iw_i|x_i|^p)^{1/p}$. For $0<p<q$,

$$
A_p\le A_q.
$$

We prove this with Jensen's inequality for the strictly convex function
$t\mapsto t^{q/p}$ on $[0,\infty)$, then take a $q$th root. Equality holds
exactly when all $|x_i|$ are equal. The inequality also holds below 1;
those expressions are not norms. At fixed positive weights,
$A_p\to\max_i|x_i|$ as $p\to\infty$: bound the weighted sum above by
$M^p$ and below by $w_jM^p$ at a maximizing coordinate $j$.

For equal weights and observations $(0,0,0,4)$,
$A_1=1$, $A_2=2$, $A_4=2\sqrt2$, and $A_\infty=4$.
The order changes which quantity is summarized; a larger value alone
neither proves a power law nor diagnoses estimation failure.

For geometric unit balls, use the **unnormalized** norm
$\|x\|_p=(\sum|x_i|^p)^{1/p}$. Its value decreases with $p$ at fixed
$x$. To prove this for $x\ne0$, divide each coordinate by $\|x\|_p$;
the resulting magnitudes are at most 1, so their $q$th powers sum to at
most their $p$th powers, namely 1. The $p$-unit balls therefore expand.
In two dimensions, $p=1,2,\infty$ give a diamond, circle, and square,
with areas 2, $\pi$, and 4. The factor $n^{-1/p}$ in $A_p$ explains
why its monotonicity goes in the other direction.

```{figure} ../assets/figures/incerto/finite-moment-norms.svg
:alt: Unit contours for unnormalized norms of orders one, two, four and infinity in two dimensions, expanding from a diamond to a square.

Exact geometric contours. These use the sum of powers, whereas the
empirical power mean uses the average of powers. No probability model
is needed for this comparison.
```

## Mean absolute deviation is a different estimand

If $\mu=\mathbb EX$ and $\mathbb E|X|<\infty$, then
$\mathbb E|X-\mu|\le\mathbb E|X|+|\mu|<\infty$.
If the variance is finite, Cauchy–Schwarz gives
$\mathbb E|X-\mu|\le\sqrt{\operatorname{Var}(X)}$.
Equality requires $|X-\mu|$ to be constant almost surely.
These are unscaled deviations; Gaussian calibration multiplies the mean
absolute deviation by $\sqrt{\pi/2}$, so that universal inequality does
not apply to the calibrated value.

For example, the symmetric law on $\{-1,1\}$ has standard deviation and
unscaled mean absolute deviation both equal to 1. Its Gaussian-calibrated
Std/MeanAD ratio is $\sqrt{2/\pi}<1$. The Gaussian benchmark 1 in the
[dispersion diagnostic](incerto-dispersion-ratio.md) is a model reference,
not a universal lower bound. Mean and median absolute deviations also
answer different questions. Neither universally replaces a second-moment
estimand or dominates every estimator under every loss.

(moment-estimation-sufficiency)=
## Matching a Gaussian sample by absolute error

For iid integrable observations, let $M(n)=\mathbb E|S_n-n\mu|$ and
assume $M(1)>0$. The [kappa note](incerto-kappa.md#kappa-definition)
compares this sum deviation at two sizes. Here the target is the
relative mean absolute error of the sample mean:

$$
R(n)=\frac{M(n)}{nM(1)}.
$$

Scaling two parent laws to the same single-observation MAD makes $R(n)$
a comparison in common units. A Gaussian sample of size $n_g$ has
$R_g(n_g)=n_g^{-1/2}$. Define the minimum matching integer by

$$
\begin{aligned}
\mathcal N_g&=\{n\ge1:R(n)\le n_g^{-1/2}\},\\
n_*&=\min\mathcal N_g.
\end{aligned}
$$

This criterion matches one expectation of an estimation error. It does
not equate variances, quantiles, or confidence coverage. For example,
Markov's inequality gives only

$$
\begin{gathered}
P(|\bar X_n-\mu|>\varepsilon)
\le\frac{M(n)}{n\varepsilon},\\
\varepsilon>0.
\end{gathered}
$$

The bound may be loose. There is no universal kappa cutoff that certifies
a normal confidence interval. Under variance matching, every iid law
with the same finite variance has the same variance of its sample mean
at the same size; that is a different comparison.

### Why the search is valid

We prove that $R(n)$ is nonincreasing. Center $n+1$ observations and
form the $n+1$ averages that each omit one observation. Their average is
the full $(n+1)$-observation average. Convexity of absolute value and
identical distributions therefore give

$$
\mathbb E|\bar X_{n+1}-\mu|
\le\mathbb E|\bar X_n-\mu|.
$$

It also tends to zero under integrability. To see this, clip each centered
variable to $[-B,B]$ and center the clipped variable again. Its average
has expected absolute value at most its standard deviation divided by
$\sqrt n$. The remaining centered part has expected absolute value at
most twice the expected clipping error, uniformly in $n$, and this tends
to zero as $B\to\infty$. First take $n\to\infty$, then $B\to\infty$.
Thus $n_*$ exists for every finite integer $n_g$ under these assumptions.
A finite computational search can still stop before reaching it.

Use [`gaussian_equivalent_n`](incerto-api-reference.md#api-gaussian-equivalent-n)
with deterministic population MADs, such as
[`student_sum_mad`](incerto-api-reference.md#api-student-sum-mad) and
[`gamma_sum_mad`](incerto-api-reference.md#api-gamma-sum-mad).

```python
from xmath.incerto.concentration import gaussian_equivalent_n, student_sum_mad, gamma_sum_mad

target = 1/30**0.5
print(f"Gaussian target: {target:.4f}")
for name, mad in (("Student 3", student_sum_mad), ("Exponential", gamma_sum_mad)):
    n = gaussian_equivalent_n(mad, 30)
    print(f"{name}: n={n}")
    for j in (n-1, n):
        print(f"  n={j}: relative error={mad(j)/(j*mad(1)):.4f}")
print("Cap at 10:", gaussian_equivalent_n(student_sum_mad, 30, max_n=10))
```

**Output**

```text
Gaussian target: 0.1826
Student 3: n=44
  n=43: relative error=0.1837
  n=44: relative error=0.1817
Exponential: n=36
  n=35: relative error=0.1829
  n=36: relative error=0.1803
Cap at 10: None
```

The cap result means only that the requested finite search did not find a
match. A noisy Monte Carlo curve must not be passed to this binary search:
sampling noise can destroy monotonicity and shift the first crossing.

### When a constant kappa is enough

If a model has the **exact** power relation
$M(n)/M(1)=n^{1/(2-\kappa)}$ at every size, with constant $\kappa<1$,
solving the matching inequality gives

$$
\begin{aligned}
n_*&=\left\lceil n_g^{\,p}\right\rceil,\\
p&=\frac{2-\kappa}{2(1-\kappa)}.
\end{aligned}
$$

For an integrable stable law, $\kappa=2-\alpha$ and
$p=\alpha/[2(\alpha-1)]$. At $\alpha=1.5$ this is $p=1.5$.
For a general law, substituting a small-size kappa into this formula is a
heuristic. Its kappa changes with size, so the exact equation is implicit.
The cubic Student and exponential examples use their full MAD curves.

```{figure} ../assets/figures/incerto/kappa-sample-size.svg
:name: kappa-sample-size-figure
:alt: Minimum observation counts versus Gaussian reference size for Gaussian, exponential, cubic Student and stable index 1.5 laws, all matched by single-observation MAD.

Exact integer searches using population MAD curves, and the exact stable
power law, over Gaussian reference sizes 2–1000. The dashed diagonal is
equal sample sizes. This compares expected absolute error under specified
iid laws, not observed data or confidence levels.
```

(moment-estimation-portfolio)=
## Equal-weight exposures and apparent stabilization

For independent, identically distributed centered exposures $X_i-\mu$,
the equally weighted portfolio fluctuation is
$n^{-1}\sum_i(X_i-\mu)$. Its MAD is exactly $M(n)/n$.
The [aggregation figure](incerto-kappa.md#kappa-aggregation-figure) therefore
also quantifies this narrowly specified diversification calculation.
A smooth sample path or a visually concentrated center can still conceal
substantial expected absolute error.

Dependence changes the result. If every exposure equals the same integrable
random variable $Y$, then the portfolio average is $Y$ at every size and
its MAD never decreases. More generally, writing $X_i=C+\epsilon_i$ with
an integrable common component $C$ independent of iid centered integrable
$\epsilon_i$ gives an average converging to $C$ in $L^1$. Its MAD about
$\mathbb EC$ consequently tends to $\mathbb E|C-\mathbb EC|$: apply
the reverse triangle inequality and the $L^1$ averaging argument above.
Increasing the number of components cannot remove the common fluctuation.

These model identities are the scope used for @taleb2025scoftv4,
§§8.3.2/8.5, pp. 159–164. They do not establish an optimal allocation,
a universal risk multiple, or empirical kappa values for stocks.
For different weights and dependence, return to the
[multivariate note](incerto-multivariate-tails.md) and
[shared-shock example](incerto-decentralization.md).

The chapter-8 figures are regenerated by
`uv run python demos/incerto/concentration.py`.

## Reproduce the moment-estimation experiment

Use [`pareto_type1`](incerto-api-reference.md#api-pareto-type1), [`pareto_moment_estimator_summary`](incerto-api-reference.md#api-pareto-moment-estimator-summary).

:::{dropdown} Code

```python
import numpy as np
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.finite_moments import pareto_moment_estimator_summary

rng = np.random.default_rng(20261020)
for alpha in (3, 6):
    sample = pareto_type1.rvs(alpha, size=(100, 200), random_state=rng)
    estimates = np.mean(sample**2, axis=1)
    print(f"alpha={alpha:.4g}")
    for name, value in pareto_moment_estimator_summary(alpha, q=2, n=200).items():
        print(f"  {name}: {value:.4g}")
    quantiles = np.quantile(estimates, [0.1, 0.5, 0.9])
    print("Sample quantiles: [" + ", ".join(f"{v:.4g}" for v in quantiles) + "]")
```

**Output**

```text
alpha=3
  moment: 3
  estimator_variance: inf
Sample quantiles: [2.335, 2.726, 3.491]
alpha=6
  moment: 1.5
  estimator_variance: 0.00375
Sample quantiles: [1.433, 1.497, 1.592]
```
:::

The quantiles describe 100 seeded second-moment estimates from samples of size 200.

Figure regeneration: `uv run python demos/incerto/finite_moments.py`. NumPy's PCG64 generator
uses seed `20261018`; sample sizes are 50, 200, 1000, and 5000, with 600
independent replications per law and size. The four laws are unit normal,
the stated rare-state mixture, and Pareto with $x_m=1$, $\alpha=3,6$.
The script allocates one sample size at a time, about 3 million observations
at the largest size, and normally takes seconds. It performs no clipping,
censoring, external-data access, or silent non-finite removal.

The experiment illustrates finite-sample spread. The proofs above establish
moment boundaries; a finite Monte Carlo variance cannot test an infinite
population variance. Later chapters develop convergence rates and broader
comparisons. The Cauchy running-statistics experiment belongs to chapter 7.

## References

- @taleb2025scoftv4, *Statistical Consequences of Fat Tails*, third edition,
  2025, ch. 4, §§4.4–4.4.4, printed pp. 79–86, and §4.5, pp. 88–89:
  dispersion, moment estimation, and norm geometry. The precise estimator
  distinctions and proofs here are our own. The recommendation in §4.4.5,
  pp. 87–88 is not used as a universal statistical theorem.
- The linked Pareto moment, infinite-mean, and plug-in notes supply the
  prerequisite proofs at their stated scopes.
