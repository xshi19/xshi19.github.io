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

We distinguish existence of the target from variance of its estimator.

:::{prf:proposition} Pareto sample-moment expectation and variance
:label: prop-moment-estimation-pareto-variance

Let $X_1,\ldots,X_n$ be iid Pareto Type I observations with
$x_m,\alpha>0$, let $0<q<\alpha$, and let $n\ge1$ be an integer.
Then $\widehat m_{q,n}=n^{-1}\sum_{i=1}^nX_i^q$ is unbiased for
$m_q=\alpha x_m^q/(\alpha-q)$. Its variance is finite exactly when
$2q<\alpha$, in which case

$$
\begin{aligned}
n\operatorname{Var}(\widehat m_{q,n})
&=\frac{\alpha x_m^{2q}}{\alpha-2q}\\
&\quad-\left(\frac{\alpha x_m^q}{\alpha-q}\right)^2.
\end{aligned}
$$

For $q<\alpha\le2q$, its variance is infinite; the equality
$2q=\alpha$ is the logarithmically divergent boundary.
:::

:::{prf:proof}
Linearity gives the expectation whenever $m_q$
is finite. For $2q<\alpha$, independence makes the cross-covariances zero.
For $q<\alpha\le2q$, nonnegativity gives
$\widehat m_{q,n}^2\ge X_1^{2q}/n^2$, whose expectation is infinite.
Since the estimator has finite mean, its variance is infinite too. Equality
at $2q=\alpha$ is excluded by the logarithmic divergence of the Pareto
moment integral.
:::

For example, $\alpha=3,x_m=1$ gives $m_2=3$, but
$\widehat m_{2,n}$ has infinite variance at every finite $n$.
At $\alpha=6$, $m_2=3/2$, $m_4=3$, and the estimator variance is
$3/(4n)$. A simulated finite standard error in the first model cannot
certify the population variance of the estimator.

### Consistency does not require estimator variance

The integrable weak law applies to the powered observations. Its clipping
argument also gives a direct proof of this consequence.

:::{prf:corollary} Consistency of an integrable empirical moment
:label: cor-moment-estimation-consistency

Let $X_i\ge0$ be iid and let $q>0$ with $m_q=\mathbb EX_1^q<\infty$.
Then $\widehat m_{q,n}=n^{-1}\sum_{i=1}^nX_i^q\to m_q$ in probability,
even if $\mathbb EX_1^{2q}=\infty$. For iid real $X_i$, the same
conclusion holds for empirical absolute moments when
$\mathbb E|X_1|^q<\infty$.
:::

:::{prf:proof}
The integrable weak law [](#prop-lln-failure-integrable-weak-law)
applies to $X_i^q$, or $|X_i|^q$ in the signed case.
To spell out its argument here, for a fixed truncation level $B>0$,
Chebyshev's inequality
gives convergence of the average of $\min(X_i^q,B)$ to its expectation,
because these variables are bounded and iid.
The omitted nonnegative part has expected average
$\mathbb E(X_1^q-B)_+\to0$ as $B\to\infty$. Markov's inequality controls
that average uniformly in $n$. For any error tolerance, first choose $B$
to make this expectation small, then $n$ to control the bounded average.
The triangle inequality proves $\widehat m_{q,n}\to m_q$ in probability.
:::

This argument supplies no uniform rate over all integrable laws.

When $m_q=\infty$, the stronger almost-sure divergence follows from the
[nonnegative infinite-mean theorem](incerto-lln-failure.md), applied to
$X_i^q$. That linked note proves the result by truncation. Finite observed
values and floating-point overflow are separate matters; neither determines
which population regime applies.

(moment-estimation-root)=
## A square root changes the moment question

:::{prf:proposition} Finite variance after taking a square root
:label: prop-moment-estimation-square-root

Let $X_1,\ldots,X_n$ be iid real observations, $n\ge1$, with
$m_2=\mathbb EX_1^2<\infty$, and put
$\widehat m_{2,n}=n^{-1}\sum X_i^2$ and
$R_n=\sqrt{\widehat m_{2,n}}$. Then

$$
\begin{gathered}
\mathbb E R_n^2=m_2,\\
\mathbb E R_n\le\sqrt{m_2},\\
\operatorname{Var}(R_n)<\infty.
\end{gathered}
$$

Also, for $s_n=\sqrt{n^{-1}\sum(X_i-\bar X_n)^2}$,
$\mathbb Es_n^2\le m_2$ and $\operatorname{Var}(s_n)<\infty$.
For $n>1$, multiplying $s_n^2$ by $n/(n-1)$ preserves these
finiteness conclusions.
:::

:::{prf:proof}
Linearity gives $\mathbb ER_n^2=m_2$. Cauchy–Schwarz gives
$\mathbb ER_n\le\sqrt{m_2}$ and finite variance.
The identity $s_n^2=\widehat m_{2,n}-\bar X_n^2\le\widehat m_{2,n}$
gives the same finite-second-moment conclusion for $s_n$.
A fixed finite correction factor preserves it.
:::

Thus infinite variance of the empirical **second moment** does not imply
infinite variance of the empirical **root mean square**.

The square root can introduce downward bias through concavity. The sample
raw moment remains unbiased. Bias, estimator variance, and the frequency
of an underestimate are different properties.

(moment-estimation-normal-comparison)=
## A comparison under an exact Gaussian model

:::{prf:proposition} Exact Gaussian risks for RMS and calibrated absolute deviation
:label: prop-moment-estimation-gaussian-risk

Let $X_i$ be iid $N(0,\sigma^2)$ with $\sigma>0$, and let $n\ge1$.
With the center known to be zero, define
$R_n=(n^{-1}\sum X_i^2)^{1/2}$ and
$A_n=\sqrt{\pi/2}\,n^{-1}\sum|X_i|$. Then

$$
\mathbb EA_n=\sigma,\qquad
\frac{\mathbb E(A_n-\sigma)^2}{\sigma^2}=\frac{\pi/2-1}{n},
$$

$$
\frac{\mathbb ER_n}{\sigma}
=\sqrt{\frac2n}\frac{\Gamma((n+1)/2)}{\Gamma(n/2)},\qquad
\frac{\mathbb E(R_n-\sigma)^2}{\sigma^2}
=2\left(1-\frac{\mathbb ER_n}{\sigma}\right).
$$
:::

:::{prf:proof}
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
\frac{\mathbb ER_n}{\sigma}
&=\sqrt{\frac2n}\frac{\Gamma((n+1)/2)}{\Gamma(n/2)},\\
\frac{\mathbb E(R_n-\sigma)^2}{\sigma^2}
&=2\left(1-\frac{\mathbb ER_n}{\sigma}\right).
\end{aligned}
$$

The last identity uses $\mathbb ER_n^2=\sigma^2$. The corresponding MSE
for $A_n$, divided by $\sigma^2$, is $(\pi/2-1)/n$.
:::

These are exact finite-$n$ risks under squared-error loss, including
$R_n$'s bias. The demo prints them for $n=10,100,1000$.

:::{prf:corollary} The calibrated absolute deviation has a different mixture target
:label: cor-moment-estimation-mixture-target

Let $X_i$ be iid copies of $X=\sqrt YZ$, where $Y>0$ is
independent of $Z\sim N(0,1)$ and $\mathbb EY<\infty$.
For $A_n=\sqrt{\pi/2}\,n^{-1}\sum|X_i|$,
$\mathbb EA_n=\mathbb E\sqrt Y\le\sqrt{\mathbb EY}
=\sqrt{\operatorname{Var}(X)}$.
The inequality is strict if $Y$ is not almost surely constant.
:::

:::{prf:proof}
The mixture moment identity [](#prop-mixture-uncertainty-moment-transfer)
gives $\mathbb E|X|=\sqrt{2/\pi}\,\mathbb E\sqrt Y$ and
$\operatorname{Var}(X)=\mathbb EY$. Linearity gives the target of
$A_n$; strict concavity of the square root gives Jensen's inequality
and its equality condition.
:::

Changing the model can change both sampling error and the target of a
calibrated statistic.

(moment-estimation-rare-state)=
## Finite moments that most samples miss

:::{prf:proposition} A finite second moment missed by many samples
:label: prop-moment-estimation-rare-state

Let the pairs $(Y_i,Z_i)$ be iid, with $Z_i\sim N(0,1)$ independent
of $Y_i$, $P(Y_i=0.5)=0.99$ and $P(Y_i=50.5)=0.01$.
Put $X_i=\sqrt{Y_i}Z_i$. Then $\mathbb EX_i^2=1$,
$\mathbb EX_i^4=77.25$, and
$\operatorname{Var}(n^{-1}\sum X_i^2)=76.25/n$.
The event that no wide component is selected has probability $0.99^n$;
conditional on it, the empirical second moment has expectation $0.5$.
:::

:::{prf:proof}
The [two-state mixture calculation](incerto-mixture-uncertainty.md#mixture-finite-states)
gives $\mathbb EY=1$ and $\mathbb EY^2=25.75$.
Conditioning gives $\mathbb EX^2=\mathbb EY=1$ and
$\mathbb EX^4=3\mathbb EY^2=77.25$. Independence yields the
estimator variance $(77.25-1)/n$. The event probability is the product
$0.99^n$. Conditional on it, the observations are independent $N(0,0.5)$,
giving conditional second-moment expectation $0.5$.
:::

At $n=50$, this event has probability about 0.605. Rare samples
compensate in the unconditional mean; the raw second moment estimator
is still unbiased.

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

:::{prf:proposition} Infinite moments even conditional on a finite Pareto fit
:label: prop-moment-estimation-plugin-singularity

Let $X_1,\ldots,X_n$ be iid Pareto Type I with known cutoff
$x_m>0$, exponent $\alpha>1$, and integer $n\ge1$. Put
$\widehat\xi_n=n^{-1}\sum_{i=1}^n\log(X_i/x_m)$.
Conditional on $\widehat\xi_n<1$, the finite mean plug-in
$x_m/(1-\widehat\xi_n)$ has infinite expectation and infinite
second moment.
:::

:::{prf:proof}
The [plug-in note](incerto-plug-in-tail-estimation.md#plugin-caveats)
derives the gamma density of $\widehat\xi_n$, which is positive
and continuous at 1. Its positive lower bound on a small left
neighborhood of 1 bounds the conditional first moment below by a
positive multiple of $\int_{1-\delta}^1(1-t)^{-1}\,dt=\infty$.
The event being conditioned on has positive probability. Replacing the
integrand by $(1-t)^{-2}$ proves divergence of the second moment.
:::

A concentrated central part of a sampling distribution can
coexist with divergent moments. Simulations must count non-finite fits;
removing them does not remove the singularity of the remaining distribution.
No universal ranking of the sample mean and this plug-in follows.

(moment-estimation-power-means)=
## Rising power means and different targets

:::{prf:proposition} Weighted power means increase with their order
:label: prop-moment-estimation-weighted-power-means

Let $x_1,\ldots,x_n\in\mathbb R$, $n\ge1$, and let $w_i>0$
with $\sum_iw_i=1$. Define $A_p=(\sum_iw_i|x_i|^p)^{1/p}$
for $p>0$. For $0<p<q$, $A_p\le A_q$, with equality exactly when
all $|x_i|$ are equal. Also $A_p\to\max_i|x_i|$ as $p\to\infty$.
:::

:::{prf:proof}
Apply Jensen to the strictly convex function $t\mapsto t^{q/p}$
on $[0,\infty)$ and take a $q$th root. Positive weights give the
stated equality condition. At a maximizing coordinate $j$,
$w_j|x_j|^p\le\sum_iw_i|x_i|^p\le|x_j|^p$.
Taking $p$th roots and then letting $p\to\infty$ proves the limit,
including the zero-vector case.
:::

The inequality also holds below 1; those expressions are not norms.

For equal weights and observations $(0,0,0,4)$,
$A_1=1$, $A_2=2$, $A_4=2\sqrt2$, and $A_\infty=4$.
The order changes which quantity is summarized; a larger value alone
neither proves a power law nor diagnoses estimation failure.

:::{prf:proposition} Unnormalized power sums decrease with their order
:label: prop-moment-estimation-unnormalized-powers

For $x\in\mathbb R^n$ and $p>0$, put
$\|x\|_p=(\sum_{i=1}^n|x_i|^p)^{1/p}$.
For $0<p<q$, $\|x\|_q\le\|x\|_p$; consequently
$\{x:\|x\|_p\le1\}\subseteq\{x:\|x\|_q\le1\}$.
The notation denotes a norm only when $p\ge1$.
:::

:::{prf:proof}
The case $x=0$ is immediate.
For $x\ne0$, without loss of generality, assume $\|x\|_p=1$: both expressions
scale by the same positive factor when $x$ is divided by $\|x\|_p$.
Within this argument, $x$ now denotes that normalized vector. Its coordinate
magnitudes are at most 1, so their $q$th powers sum to at most their $p$th
powers, namely 1. The $p$-unit balls therefore expand.
:::

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

:::{prf:proposition} Mean absolute deviation and standard deviation
:label: prop-moment-estimation-mad-bound

Let $X$ be real with $\mathbb E|X|<\infty$ and
$\mu=\mathbb EX$. Then
$\mathbb E|X-\mu|\le\mathbb E|X|+|\mu|<\infty$.
If $\operatorname{Var}(X)<\infty$, then
$\mathbb E|X-\mu|\le\sqrt{\operatorname{Var}(X)}$,
with equality exactly when $|X-\mu|$ is constant almost surely.
:::

:::{prf:proof}
The first bound is the triangle inequality in expectation.
For the second, apply Cauchy–Schwarz to $|X-\mu|$ and the constant 1.
Its equality condition is precisely that $|X-\mu|$ be constant
almost surely, including the degenerate case.
:::

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

For iid integrable observations with mean $\mu$, let $M(n)=\mathbb E|S_n-n\mu|$ and
assume $M(1)>0$. The [kappa note](incerto-kappa.md#kappa-definition)
compares this sum deviation at two sizes. Here the target is the
relative mean absolute error of the sample mean:

$$
R(n)=\frac{M(n)}{nM(1)}.
$$

Scaling two parent laws to the same single-observation MAD makes $R(n)$
a comparison in common units. For a Gaussian sample of size $n_g$, this
relative error is $n_g^{-1/2}$. Define the minimum matching integer by

$$
n_*=\min\{n\in\mathbb N:n\ge1,\ R(n)\le n_g^{-1/2}\}.
$$

This criterion matches one expectation of an estimation error. It does
not equate variances, quantiles, or confidence coverage. For example, the
corresponding probability bound is only Markov's inequality.

:::{prf:proposition} A probability bound from expected absolute error
:label: prop-moment-estimation-mad-probability-bound

For iid integrable $X_i$ with mean $\mu$, let
$S_n=\sum_{i=1}^nX_i$ and $M(n)=\mathbb E|S_n-n\mu|$.
For every $n\ge1$ and $\varepsilon>0$,
$P(|\bar X_n-\mu|>\varepsilon)\le M(n)/(n\varepsilon)$.
:::

:::{prf:proof}
Apply Markov's inequality to $|\bar X_n-\mu|$, whose expectation
is $M(n)/n$.
:::

The bound may be loose. There is no universal kappa cutoff that certifies
a normal confidence interval. Under variance matching, every iid law
with the same finite variance has the same variance of its sample mean
at the same size; that is a different comparison.

### Why the search is valid

We justify the population search using monotonicity and convergence.

:::{prf:lemma} Existence of a Gaussian absolute-error match
:label: lem-moment-estimation-matching-search

Let $X_i$ be iid integrable observations with mean $\mu$ and
$M(1)=\mathbb E|X_1-\mu|>0$. Define
$M(n)=\mathbb E|S_n-n\mu|$ and $R(n)=M(n)/(nM(1))$.
Then $\mathbb E|\bar X_n-\mu|=M(n)/n$ is nonincreasing and tends to
zero, and so does $R(n)$. For every finite integer
$n_g\ge1$, the set $\{n\ge1:R(n)\le n_g^{-1/2}\}$ is nonempty
and has a finite smallest integer $n_*$.
:::

:::{prf:proof}
Center $n+1$ observations and
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
:::

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

:::{prf:corollary} Exact sample-size matching under a constant kappa
:label: cor-moment-estimation-constant-kappa

Under the assumptions of [](#lem-moment-estimation-matching-search),
suppose $M(n)/M(1)=n^{1/(2-\kappa)}$ at every integer $n\ge1$,
for a fixed $\kappa<1$. Then for every integer $n_g\ge1$,

$$
n_*=\left\lceil n_g^{\,(2-\kappa)/[2(1-\kappa)]}\right\rceil.
$$

For an integrable nondegenerate stable law of index $1<\alpha\le2$,
$\kappa=2-\alpha$, giving exponent $\alpha/[2(\alpha-1)]$.
:::

:::{prf:proof}
Substitution gives $R(n)=n^{-(1-\kappa)/(2-\kappa)}$.
Its exponent is negative, so $R(n)\le n_g^{-1/2}$ is equivalent to
$n\ge n_g^{(2-\kappa)/[2(1-\kappa)]}$. Taking the smallest integer
gives the ceiling. The stable kappa formula [](#cor-kappa-stable)
gives the last assertion.
:::

At $\alpha=1.5$, this exponent is $1.5$.

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

:::{prf:corollary} A common component survives averaging
:label: cor-moment-estimation-common-component

Let $C$ be integrable, independent of iid centered integrable
$\epsilon_i$, and set $X_i=C+\epsilon_i$. Then
$\bar X_n\to C$ in $L^1$ and
$\mathbb E|\bar X_n-\mathbb EC|\to\mathbb E|C-\mathbb EC|$.
If $\epsilon_i=0$ almost surely, the average is exactly $C$ at every size.
:::

:::{prf:proof}
The clipping argument in [](#lem-moment-estimation-matching-search)
gives $\mathbb E|n^{-1}\sum_i\epsilon_i|\to0$. If the deviations
are identically zero this is immediate; otherwise the lemma applies with
positive normalization. Since $\bar X_n-C=n^{-1}\sum_i\epsilon_i$,
this is $L^1$ convergence. The reverse triangle inequality bounds the
difference of the two MADs by $\mathbb E|\bar X_n-C|$.
:::

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
