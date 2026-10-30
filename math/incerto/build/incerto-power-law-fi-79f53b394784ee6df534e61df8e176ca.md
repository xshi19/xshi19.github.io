---
title: Fitting a power-law tail
options:
  concept:
    id: power-law-fitting
    type: method
    prerequisites:
      - pareto
      - hill-estimator
      - tail-threshold-selection
    tags:
      - estimation
      - power-law
      - diagnostics
---

A fitted power law answers a conditional question: which exponent best
describes observations above a proposed lower threshold? It does not establish
that the tail is a power law. Clauset, Shalizi, and Newman (CSN)
@clauset2009powerlaw combine maximum likelihood with a threshold search;
[goodness-of-fit and model comparisons](incerto-power-law-goodness-of-fit.md)
then examine the resulting model.

This page develops those calculations for independent, identically distributed
positive observations. The [Pareto distribution](incerto-pareto.md) supplies
the exact continuous model. The [Hill estimator](incerto-hill-estimator.md)
allows a broader asymptotic interpretation under regular variation, while
[threshold diagnostics](incerto-tail-threshold-selection.md) explain why
choosing the fitting region matters.

:::{note} Worked example
The [tech layoffs example](incerto-layoffs-tails.md) applies these methods
to event counts and workforce log losses. The examples on this theory page
remain synthetic.
:::

(csn-exponent-convention)=
## Two exponent conventions

We retain the [shared notation](https://xshi19.github.io/math/notation/):
$\alpha>0$ is the survival-tail exponent and $\xi=1/\alpha$ is the positive
extreme-value index. Write $u=x_{\min}>0$ for the fitting threshold. On this
page, $a$ is a local symbol for the **density or mass exponent** used by CSN:

$$
\begin{aligned}
a&=\alpha+1,\\
\bar F(x\mid X\ge u)&=(x/u)^{-\alpha},\qquad x\ge u.
\end{aligned}
$$

Every `alpha` argument or result in the package below uses $\alpha$, including
for integer data. For an integer power law the survival is a zeta ratio,
not exactly $(x/u)^{-\alpha}$; its large-$x$ exponent is still $\alpha=a-1$.
The change of exponent convention must precede any comparison with software
that reports $a$.

Let $N$ denote the full sample size and $n=n_{\mathrm{tail}}$ the number of
observations satisfying $x_i\ge u$. All likelihood sums below run over those
$n$ retained values. Equality is included, which matters for integer data and
for thresholds chosen from observations.

(csn-continuous-mle)=
## Continuous likelihood and its relation to Hill

At a fixed threshold, the conditional density with respect to Lebesgue measure is

$$
\begin{aligned}
f(x\mid\alpha,u)&=\frac{\alpha}{u}(x/u)^{-(\alpha+1)},\\
&\qquad x\ge u.
\end{aligned}
$$

We derive the MLE within this exact model. Let
$T=\sum_{i=1}^n\log(x_i/u)$. Its log likelihood and score are

$$
\begin{aligned}
\ell(\alpha)&=n\log\alpha-n\log u-(\alpha+1)T,\\
\ell'(\alpha)&=n/\alpha-T,\\
\ell''(\alpha)&=-n/\alpha^2.
\end{aligned}
$$

If $T>0$, strict concavity gives a unique finite maximum:

$$
\begin{aligned}
\widehat\alpha&=n/T,\\
\widehat a&=1+n/T.
\end{aligned}
$$

If every retained value equals $u$, the likelihood increases without a finite
maximizer as $\alpha\to\infty$. Reporting a finite exponent in that case would
invent information about the tail's spread.

The reciprocal estimate is the average log excess used by Hill
[@hill1975simple]. There is a small but real counting distinction. For distinct
continuous observations, a Hill fit above the order statistic $x_{N-k:N}$
uses the $k$ larger observations. Setting $u=x_{N-k:N}$ and retaining equality
in the CSN fit includes one more observation with zero log excess. Thus

$$
\widehat\alpha_{\mathrm{inclusive}}
=\frac{k+1}{k}\widehat\alpha_{\mathrm{Hill}}.
$$

With a fixed external cutoff and the same retained sample, the reciprocal
mean-log-excess calculation is identical. Ties require the actual counts;
the displayed factor assumes a single threshold observation.

Use [`sample_powerlaw`](incerto-api-reference.md#api-sample-powerlaw),
[`powerlaw_mle_continuous`](incerto-api-reference.md#api-powerlaw-mle-continuous),
and [`powerlaw_standard_error`](incerto-api-reference.md#api-powerlaw-standard-error):

```python
from xmath.incerto.powerlaw import (
    sample_powerlaw, powerlaw_mle_continuous, powerlaw_standard_error,
)
x = sample_powerlaw(1.6, 2, 1500, random_state=310)
alpha = powerlaw_mle_continuous(x, 2)
se = powerlaw_standard_error(alpha, len(x))
print(f"Survival exponent: {alpha:.3f}; density exponent: {alpha+1:.3f}")
print(f"Fixed-threshold SE: {se:.3f}")
```

**Output**

```text
Survival exponent: 1.591; density exponent: 2.591
Fixed-threshold SE: 0.041
```

(csn-discrete-mle)=
## Integer data: the zeta likelihood

For positive integer observations and integer $u\ge1$, specify a mass function
with respect to counting measure:

$$
\begin{aligned}
p(k\mid a,u)&=\frac{k^{-a}}{\zeta(a,u)},\qquad k\ge u,\\
\zeta(a,u)&=\sum_{j=u}^{\infty}j^{-a},\qquad a>1.
\end{aligned}
$$

A Yule-type mechanism can produce an asymptotic power tail without producing
this exact finite-threshold mass function. The zeta model is a hypothesis to
check, even when the data are integers and look heavy-tailed.

Writing primes for derivatives in $a$, the score equation is

$$
\begin{aligned}
\ell(a)&=-n\log\zeta(a,u)-a\sum_{i=1}^n\log x_i,\\
\frac{\zeta'(\widehat a,u)}{\zeta(\widehat a,u)}
&=-\frac1n\sum_{i=1}^n\log x_i.
\end{aligned}
$$

This generally needs a numerical solve. The reason the objective is concave
also supplies its information formula. Differentiating the normalized series
for $a>1$ gives

$$
\begin{aligned}
\partial_a\log\zeta(a,u)&=-\mathbb E_a[\log X],\\
\partial_a^2\log\zeta(a,u)&=\operatorname{Var}_a(\log X)>0.
\end{aligned}
$$

Logarithmic moments are finite at every interior $a>1$; this calculation does
not require $\mathbb E X$ or $\operatorname{Var}(X)$ to exist. The likelihood
is strictly concave. Its score crosses zero when the sample mean of $\log X$
exceeds $\log u$; an all-$u$ tail again has only an infinite-exponent limit.

### The half-integer approximation

Replacing the integer sum by a continuous integral starting halfway below the
first integer suggests

$$
\widehat a_{\mathrm{approx}}
=1+\frac{n}{\sum_{i=1}^n\log\!\left(x_i/(u-\tfrac12)\right)}.
$$

The half-unit correction matters: substituting $u$ in this approximation is
a different and generally worse approximation. Neither expression is the
exact zeta MLE. When $u$ is small or $a$ is large, a few lattice points carry
much of the mass, so replacing their sum by an integral can distort the
normalization substantially. Larger $u$ helps only relative to the local
rate of decay. There is no universal cutoff beyond which the approximation
is accurate for every exponent. A large sample reduces sampling error while
leaving this approximation error in place.

[`powerlaw_mle_discrete`](incerto-api-reference.md#api-powerlaw-mle-discrete)
uses the zeta likelihood by default. Its optional approximation returns the
survival exponent $\widehat a_{\mathrm{approx}}-1$. Here we compare both on
exact zeta samples with the same population exponent:

```python
from xmath.incerto.powerlaw import sample_powerlaw, powerlaw_mle_discrete

for u in (1, 8):
    x = sample_powerlaw(2, u, 2000, discrete=True, random_state=320+u)
    exact = powerlaw_mle_discrete(x, u)
    approx = powerlaw_mle_discrete(x, u, approximate=True)
    print(f"u={u}: exact alpha={exact:.3f}; approximate alpha={approx:.3f}")
```

**Output**

```text
u=1: exact alpha=2.032; approximate alpha=1.173
u=8: exact alpha=2.059; approximate alpha=2.043
```

(csn-ks-threshold)=
## Choose a threshold by refitting at each candidate

For each admissible threshold $u$, fit its own exponent and compute the
Kolmogorov–Smirnov distance between the conditional empirical CDF $F_{n,u}$
and the fitted model CDF $F_{\widehat\alpha(u),u}$:

$$
\begin{aligned}
D(u)&=\sup_{x\ge u}
\left|F_{n,u}(x)-F_{\widehat\alpha(u),u}(x)\right|,\\
\widehat u&\in\arg\min_{u\in\mathcal U}D(u).
\end{aligned}
$$

Here $\mathcal U$ is the declared candidate set. A usual choice is the distinct
observed values with enough observations above them to permit a fit. A fixed
grid can reduce computation, but then the result minimizes over that grid
only. Record the minimum tail count, candidate rule, selected count, and
nearby values of $D(u)$; a small difference between two minima is not evidence
for a sharply located population boundary.

For continuous data, the supremum checks both sides of every empirical jump.
For integer data the model jumps too:

$$
F_{a,u}(k)=1-\frac{\zeta(a,k+1)}{\zeta(a,u)}.
$$

Check $k-1$ and $k$ for every observed integer $k$. These endpoints also cover
gaps containing no observations: the empirical CDF is constant there and the
model CDF is monotone. Comparing just the right sides at observed values can
miss the largest discrepancy. Use
[`ks_distance_powerlaw`](incerto-api-reference.md#api-ks-distance-powerlaw)
and [`select_xmin`](incerto-api-reference.md#api-select-xmin).

```{figure} ../assets/figures/incerto/powerlaw-fitted-tail.svg
:alt: Log-log empirical survival for a simulated uniform-body and Pareto-tail mixture, with the fitted power tail starting at the selected minimum and a dotted population join at four.

Synthetic iid mixture: uniform on $[1,4]$ with probability $0.7$; otherwise
Pareto with minimum $4$ and survival exponent $1.8$. PCG64 seed 982,
$N=6000$. The empirical curve counts strict exceedances and omits its zero
endpoint on log axes. The fitted line starts at the selected minimum and has
height $n/N$ there: it represents the fitted tail's unconditional mass.
Its height can differ from strict empirical survival by the observations
exactly at the threshold. The population join is known only because we
specified this simulation.
```

```{figure} ../assets/figures/incerto/powerlaw-ks-threshold.svg
:alt: KS distance and retained count versus candidate minimum on two stacked panels, with the selected KS minimum and the known population join marked.

The same sample, refitted at 21 thresholds from $1$ to $6$ in increments of
$0.25$, requiring at least 200 retained observations. The lower panel exposes
the loss of observations as the threshold rises. The minimum is a sample
result on this candidate grid, not a proof of the true join or tail class.
```

The following code reproduces the fit underlying both figures:

:::{dropdown} Code

```python
import numpy as np
from xmath.incerto.powerlaw import sample_powerlaw, select_xmin

rng = np.random.default_rng(982)
x = rng.uniform(1, 4, 6000)
tail = rng.random(x.size) < 0.3
x[tail] = sample_powerlaw(1.8, 4, int(tail.sum()), random_state=rng)
fit = select_xmin(x, candidates=np.arange(1, 6.01, 0.25), min_tail=200)
print(f"Selected minimum: {fit['xmin']:.2f}")
print(f"Survival exponent: {fit['alpha']:.3f}")
print(f"Retained count: {fit['n_tail']}")
print(f"KS distance: {fit['ks']:.3f}")
```

**Output**

```text
Selected minimum: 4.00
Survival exponent: 1.725
Retained count: 1815
KS distance: 0.021
```
:::

Regenerate the SVGs with `uv run python demos/incerto/powerlaw_fitting.py`.
The shared figure style keeps the diagnostic panels in one column.

(csn-fitting-uncertainty)=
## Fixed-threshold errors and jointly refitted uncertainty

For the exact continuous law at fixed $u$, $\log(X/u)$ is exponential with
rate $\alpha$. Consequently the per-observation Fisher information is
$1/\alpha^2$. The ordinary interior-MLE approximation gives

$$
\begin{aligned}
\operatorname{se}(\widehat\alpha)&\simeq\frac{\widehat\alpha}{\sqrt n},\\
\operatorname{se}(\widehat a)&\simeq\frac{\widehat a-1}{\sqrt n}.
\end{aligned}
$$

The second line is the familiar CSN expression, often written
$(\widehat\alpha_{\mathrm{CSN}}-1)/\sqrt n$. Adding one changes the exponent's
name and level, but not its standard error. This is asymptotic sampling error
conditional on the threshold and retained count, not a finite-sample unbiased
variance formula.

For the discrete model, the same MLE approximation uses

$$
\begin{aligned}
I(a;u)&=\frac{\zeta''(a,u)}{\zeta(a,u)}
-\left[\frac{\zeta'(a,u)}{\zeta(a,u)}\right]^2,\\
\operatorname{se}(\widehat a)&\simeq[nI(\widehat a;u)]^{-1/2}.
\end{aligned}
$$

The variance identity above derives $I$; asymptotic normality and the MLE
standard-error approximation are cited results, not proved here
[@clauset2009powerlaw, §3.1 and appendix B]. The helper evaluates the discrete
curvature numerically. Very steep laws or extreme numerical scales can make
that calculation ill-conditioned.

When $u$ was selected from the data, both formulas omit selection uncertainty.
For an empirical bootstrap, resample $N$ observations with replacement from
the **whole sample**, then repeat the threshold search and exponent fit.
Retain the pairs $(\widehat\alpha^{*(b)},\widehat u^{*(b)})$. Their sample
standard deviations estimate the two marginal errors; their covariance
records how the estimates move together. Resampling only the selected tail
at a fixed cutoff cannot measure uncertainty in choosing that cutoff.

[`powerlaw_bootstrap_se`](incerto-api-reference.md#api-powerlaw-bootstrap-se)
performs this full refit. The small replicate count here illustrates the
calculation; it is insufficient for stable interval endpoints.

```python
from xmath.incerto.powerlaw import sample_powerlaw, powerlaw_bootstrap_se

x = sample_powerlaw(1.6, 2, 240, random_state=330)
boot = powerlaw_bootstrap_se(x, replicates=24, min_tail=40, random_state=331)
print(f"Bootstrap SE(alpha): {boot['alpha_se']:.3f}")
print(f"Bootstrap SE(xmin): {boot['xmin_se']:.3f}")
```

**Output**

```text
Bootstrap SE(alpha): 0.126
Bootstrap SE(xmin): 0.185
```

The threshold selector is discontinuous and may jump between competing
minima. Bootstrap standard deviations are useful sensitivity summaries;
they do not by themselves establish confidence-interval coverage. Inspect
the paired replicates and repeat the analysis under reasonable candidate
rules. This empirical bootstrap is different from the fitted-null bootstrap
used for [goodness of fit](incerto-power-law-goodness-of-fit.md#csn-gof-bootstrap).

## Practical limits

An exact power tail above some finite threshold is stronger than
[regular variation](incerto-regular-variation.md). A slowly varying correction
may remain detectable at every available threshold. Raising the cutoff can
hide that discrepancy by leaving little data; report the retained fraction
alongside the fit.

Rounded continuous measurements need a measurement model; integer support
alone does not make the zeta mass correct. Censoring, reporting floors, and a
hard upper limit change the likelihood. A multiplicative change of continuous
units preserves the fitted exponent when the cutoff changes with it, whereas
adding a constant changes the model. Integer units fix the lattice and cannot
be rescaled arbitrarily while preserving the zeta family.

The likelihood and resampling calculations assume iid observations.
Dependence requires an appropriate sampling model or resampling design.
Least-squares slopes of logged histograms do not reproduce these likelihoods:
binning changes weights, empty bins vanish, and the errors are not those of
ordinary linear regression. Finally, a small fitted KS distance needs its own
calibration. It is a discrepancy statistic, not a p-value.

## See also

- [Power-law goodness of fit and alternatives](incerto-power-law-goodness-of-fit.md).
- [Hill estimator](incerto-hill-estimator.md) and
  [tail threshold selection](incerto-tail-threshold-selection.md).
- [Plug-in tail estimation](incerto-plug-in-tail-estimation.md) for consequences
  of exponent uncertainty near moment boundaries.

## References

- Clauset, Shalizi, and Newman, *Power-law distributions in empirical data*
  @clauset2009powerlaw: §§3.1–3.4 and appendix B for fitting, threshold selection,
  and uncertainty. The derivations and synthetic examples above are written
  for this note's notation and scope.
- Hill, *A Simple General Approach to Inference About the Tail of a Distribution*
  @hill1975simple: origin of the upper-order log-spacing estimator.
