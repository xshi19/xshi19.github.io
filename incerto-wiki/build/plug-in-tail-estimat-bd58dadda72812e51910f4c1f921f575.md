---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: plug-in-tail-estimation
    type: method
    prerequisites:
      - pareto
      - regular-variation
      - hill-estimator
      - tail-threshold-selection
      - mean-excess-function
      - pareto-moment-existence
    related:
      - lln-preasymptotic
      - generalized-central-limit-theorem
    tags:
      - estimation
      - tail-risk
      - moments
---

# Plug-In Tail Estimation

## Overview

Plug-in tail estimation fits a tail model and evaluates a quantity of interest
under that fitted model. If a distribution family is indexed by a parameter
$\theta$ and the target functional is $T(\theta)$, its plug-in estimate is
$T(\widehat\theta)$. The parameter can include a tail exponent, scale, and tail
probability; an exponent alone generally does not determine the target.

For an exact [Pareto distribution](../distributions/pareto.md), estimating the
exponent from logarithms can give more concentrated mean estimates than the
sample mean. This page establishes a comparison of asymptotic error scales and
illustrates finite-sample error probabilities. It does not claim uniformly
smaller mean squared error (MSE). The unregularized plug-in mean has its own
singularity at the mean-existence boundary.

A [regularly varying tail](../theorems/regular-variation.md) only specifies
limiting tail ratios. Using a [Hill estimate](hill-estimator.md) in a functional
requires a [threshold choice](tail-threshold-selection.md), a tail approximation,
and any body contribution to that functional. The exact model below isolates
the estimation mechanism before these modeling uncertainties enter.

The sample, tail exponent $\alpha$, cutoff $x_m$, and
[extreme-value index](extreme-value-index.md) $\xi=1/\alpha$ follow the
[notation table](../../notation/index.md).
We write $\mu=\mathbb E[X]$ and $\bar X_n=n^{-1}\sum_{i=1}^n X_i$ for the
population and sample means.

(plugin-exact-pareto)=
## Exact Pareto mean with a known cutoff

Assume $X_1,\ldots,X_n$ are independent and identically distributed (iid), with

$$
\mathbb P(X_i>x)=\left(\frac{x_m}{x}\right)^\alpha,
\qquad x\ge x_m>0,
$$

where $x_m$ is known and fixed, and the true exponent satisfies $\alpha>1$.
These are assumptions about the entire distribution. The
[Pareto moment formula](../theorems/pareto-moment-existence.md) gives

$$
\mu=\frac{\alpha x_m}{\alpha-1}=\frac{x_m}{1-\xi}.
$$

The maximum-likelihood estimate of the exponent and its reciprocal are

$$
\widehat\xi_n=\frac1n\sum_{i=1}^n\log(X_i/x_m),
\qquad
\widehat\alpha_n=\frac1{\widehat\xi_n}.
$$

They use all observations above a known cutoff. Hill instead averages the top
$k$ log-excesses above the random order-statistic threshold $X_{n-k:n}$.
The formulas share a log-excess mechanism, but their sample sizes and threshold
assumptions differ.

The plug-in mean is

$$
\widehat\mu_n=\frac{x_m}{1-\widehat\xi_n}
=\frac{\widehat\alpha_n x_m}{\widehat\alpha_n-1},
\qquad \widehat\xi_n<1.
$$

If $\widehat\xi_n\ge1$, the fitted Pareto model has no finite mean. We report
$+\infty$ in this case, rather than extend the rational formula to a negative
number or silently discard the fit. If every observation equals $x_m$, the
exponent has no finite maximizer; the limiting plug-in mean is $x_m$.

(plugin-pareto-error-scale)=
## Why the logarithmic fit can improve concentration

For each fixed $\alpha>1$, the finite plug-in estimates have the asymptotic law

$$
\mathcal L\!\left(\sqrt n(\widehat\mu_n-\mu)
\mid \widehat\xi_n<1\right)
\Rightarrow
\mathcal N\!\left(0,\frac{x_m^2\alpha^2}{(\alpha-1)^4}\right),
\qquad n\to\infty.
$$

Here $\mathcal L(\cdot\mid\cdot)$ denotes a conditional sampling law,
$\Rightarrow$ denotes convergence in distribution, and $\mathcal N(0,v)$ is
the centered normal law with variance $v$. The probability of a non-finite fit
tends to zero. Thus central error quantiles shrink on the $n^{-1/2}$ scale.
This is a distributional claim; the variance in the normal limit is not a
claim about the finite-sample variance of $\widehat\mu_n$.

For $1<\alpha<2$, the sample mean remains consistent, but its error scale is
$n^{1/\alpha-1}$. More precisely, the
[generalized central limit theorem](../theorems/generalized-central-limit-theorem.md)
gives a nondegenerate, non-Gaussian $\alpha$-stable limit for
$n^{1-1/\alpha}(\bar X_n-\mu)$ in this exact Pareto model
[@resnick2007heavy]. That scale decays more slowly than $n^{-1/2}$.
The [pre-asymptotic LLN examples](../examples/lln-preasymptotic.md) show how
large observations continue to move sample means in this regime.

### Derivation of the Pareto fit and its normal limit

We derive the known-cutoff fit and the displayed plug-in limit using the
ordinary iid central limit theorem and the delta method. The general limit
theorems are cited from van der Vaart [@vandervaart1998asymptotic], Chapters 2–3;
the stable limit for the sample mean is cited above.

Set $Y_i=\log(X_i/x_m)$. For $y\ge0$,

$$
\mathbb P(Y_i>y)=\mathbb P(X_i>x_m e^y)=e^{-\alpha y}.
$$

Thus $Y_i$ is exponential with mean $1/\alpha$ and variance $1/\alpha^2$,
even when $X_i$ has infinite variance. Up to terms independent of $\alpha$,
the log likelihood is $n\log\alpha-\alpha\sum_iY_i$. Its derivative vanishes
at $\widehat\alpha_n=n/\sum_iY_i$, and its second derivative is
$-n/\alpha^2<0$. This proves the fit formula when $\sum_iY_i>0$, an event
of probability one under the continuous model.

The law of large numbers gives $\widehat\xi_n\to1/\alpha<1$ almost surely.
The central limit theorem gives

$$
\sqrt n(\widehat\xi_n-\xi)
\Rightarrow\mathcal N(0,\xi^2).
$$

For the locally smooth function $h(t)=x_m/(1-t)$,
$h'(\xi)=x_m/(1-\xi)^2$. The delta method therefore gives limiting variance

$$
[h'(\xi)]^2\xi^2
=\frac{x_m^2\xi^2}{(1-\xi)^4}
=\frac{x_m^2\alpha^2}{(\alpha-1)^4}.
$$

Because $\mathbb P(\widehat\xi_n<1)\to1$, conditioning on finite fits does
not change this weak limit. The gain comes from averaging exponential
log-excesses and using the assumed Pareto shape to extrapolate the mean.
It depends on that shape being correct.

(plugin-finite-sample-comparison)=
## Repeated-sample comparison

We compare both estimators on the same iid samples with $\alpha=1.5$ and known
$x_m=1$. The table reports quantiles of absolute relative error
$|\widehat\mu-\mu|/\mu$ and the fraction exceeding a fixed tolerance.
Non-finite estimates count as infinite errors in every summary. Empirical
quantiles use order statistics so that infinities need no interpolation.

```{code-cell} python
:label: plugin-pareto-error-comparison

import numpy as np

from incerto.distributions import pareto_type1
from incerto.estimators import pareto_mean_plugin

rng = np.random.default_rng(20260912)
alpha, x_m = 1.5, 1.0
mu = float(pareto_type1.mean(alpha, scale=x_m))
replications, tolerance = 4_000, 0.25

print(f"Exact Pareto: alpha={alpha}, known cutoff={x_m}, mean={mu:.3f}")
print(f"{replications:,} replications; errors relative to the population mean")
print("    n  estimator     median      90%      99%  P(error>25%)  nonfinite")
for n in (50, 200, 1_000):
    samples = pareto_type1.rvs(
        alpha, scale=x_m, size=(replications, n), random_state=rng
    )
    estimates = {
        "sample mean": samples.mean(axis=1),
        "plug-in": pareto_mean_plugin(samples, x_m=x_m, axis=1),
    }
    for name, estimate in estimates.items():
        error = np.abs(estimate - mu) / mu
        q50, q90, q99 = np.quantile(
            error, [0.5, 0.9, 0.99], method="inverted_cdf"
        )
        print(
            f"{n:5d}  {name:11s}  {q50:8.3f} {q90:8.3f} {q99:8.3f}"
            f"  {np.mean(error > tolerance):12.3%}"
            f"  {np.sum(~np.isfinite(estimate)):9d}"
        )
```

In this run, the plug-in errors concentrate more tightly at the larger sample
sizes. At the smallest size, its upper error quantiles exceed those of the
sample mean, and some fits have no finite mean. These results concern the
specified model and loss summaries. Neither the table nor the normal limit
establishes MSE dominance or tests whether an empirical dataset is Pareto.

(plugin-tail-body)=
## When only the tail is modeled

Let $X\ge0$ and fix a threshold $u>0$ with tail probability
$q_u=\mathbb P(X>u)>0$. Suppose the conditional tail is exactly Pareto:

$$
\mathbb P(X>x\mid X>u)=(u/x)^\alpha,
\qquad x\ge u,\quad \alpha>1.
$$

The [mean-excess formula](../theorems/mean-excess-function.md) gives
$\mathbb E[X\mid X>u]=u+e(u)=u\alpha/(\alpha-1)$, where
$e(u)=\mathbb E[X-u\mid X>u]$. Splitting the expectation at $u$ yields

$$
\mu=\mathbb E[X\mathbf1_{\{X\le u\}}]
+q_u\frac{u\alpha}{\alpha-1}.
$$

For an iid sample, let $k=\sum_i\mathbf1_{\{X_i>u\}}$ be the exceedance count.
Estimate $q_u$ by $k/n$ and fit $\widehat\alpha_u$ from those exceedances.
When $k>0$ and $\widehat\alpha_u>1$, a body-plus-tail estimate is

$$
\widehat\mu_u
=\frac1n\sum_{i=1}^n X_i\mathbf1_{\{X_i\le u\}}
+\frac{k}{n}\frac{u\widehat\alpha_u}{\widehat\alpha_u-1}.
$$

The first term is the body's contribution per original observation, not the
mean conditional on being in the body. The second includes the threshold
itself as well as the mean excess. With no exceedances there is no fitted tail
exponent; a missing tail fit should be reported. If the fitted exponent is at
most 1, the fitted conditional mean is infinite.

We now generate a uniform body on $[1,3]$ with probability $0.8$, and a Pareto
tail above the known threshold $u=3$ otherwise. Fitting the entire sample as
Pareto from 1 deliberately violates the model assumption.

```{code-cell} python
:label: plugin-body-tail-example

rng = np.random.default_rng(20260913)
n, u, tail_probability, alpha = 30_000, 3.0, 0.2, 1.5
is_tail = rng.random(n) < tail_probability
sample = rng.uniform(1.0, u, size=n)
sample[is_tail] = pareto_type1.rvs(
    alpha, scale=u, size=np.sum(is_tail), random_state=rng
)
exceeds = sample > u
tail_mean = pareto_mean_plugin(sample[exceeds], x_m=u)
combined = np.sum(sample[~exceeds]) / n + np.mean(exceeds) * tail_mean
true_mean = (
    (1 - tail_probability) * (1 + u) / 2
    + tail_probability * pareto_type1.mean(alpha, scale=u)
)

print(f"Known threshold u={u}; exceedances={np.sum(exceeds):,} of {n:,}")
print(f"Generating mean:                     {true_mean:.3f}")
print(f"Direct sample mean:                  {np.mean(sample):.3f}")
print(f"Empirical body plus fitted tail:     {combined:.3f}")
print(f"Incorrect whole-sample Pareto fit:   {pareto_mean_plugin(sample, x_m=1):.3f}")
```

This single realization illustrates the need to retain the body and tail mass;
it is not a performance ranking. For empirical thresholds, repeat the analysis
over plausible $u$ or $k$. Regular variation supports a Pareto approximation
far into the tail, not an exact conditional model at a chosen finite threshold.
The known-cutoff $\sqrt n$ calculation does not supply a rate or confidence
interval for a Hill fit with a selected threshold.

(plugin-caveats)=
## Caveats

The moment boundary is also an estimation boundary. Differentiating the
population mean with respect to the exponent gives

$$
\frac{d\mu}{d\alpha}=-\frac{x_m}{(\alpha-1)^2}.
$$

Small exponent errors can therefore produce large mean errors near $\alpha=1$.
An uncertainty interval for $\alpha$ that reaches 1 cannot be mapped to a finite
upper bound for this Pareto mean. The fixed-$\alpha$ normal approximation above
is not uniform as $\alpha\downarrow1$.

There is a stronger finite-sample warning: even conditional on a finite fit,
the unregularized plug-in estimator has infinite first and second moments.
To see this under the exact iid model, $\widehat\xi_n$ has a gamma density
$f_n(t)=(n\alpha)^n t^{n-1}e^{-n\alpha t}/\Gamma(n)$ for $t>0$, obtained by
averaging $n$ independent exponential log-excesses. Here $\Gamma$ is the gamma
function. The density is continuous and strictly positive at $t=1$.
For any sufficiently small $\varepsilon>0$, the first-moment integral includes

$$
\int_{1-\varepsilon}^1\frac{x_m}{1-t}f_n(t)\,dt=\infty;
$$

replacing $x_m/(1-t)$ by its square gives a divergent second-moment integral.
Conditioning on $t<1$ only divides these integrals by a positive probability.
Moreover, $\mathbb P(\widehat\xi_n
\ge1)>0$ for every finite $n$. Finite simulations can easily miss the
singularity. Clipping or constraining the exponent away from 1 changes the
estimator and must be disclosed, with its bias and any risk claim assessed
separately.

Model uncertainty remains after sampling error shrinks. A fitted exponent
does not determine the scale, tail probability, or body of the distribution.
Truncation, dependence, measurement limits, or an incorrectly chosen threshold
can invalidate the exact-model calibration. Report those assumptions alongside
threshold sensitivity and uncertainty propagated through the functional.
If the true exact Pareto exponent is at most 1, there is no finite population
mean to recover by this method.

## References

- van der Vaart, *Asymptotic Statistics*, Chapters 2–3: central limit theorem,
  continuous mapping, and delta method [@vandervaart1998asymptotic].
- Resnick, *Heavy-Tail Phenomena*: regularly varying tails and stable limits
  [@resnick2007heavy].
- Taleb, *Statistical Consequences of Fat Tails*: motivation for estimating
  functionals through fitted tail parameters [@taleb2020scoft]. The
  [Chapter 3 guide](../../reading-guides/taleb-scoft/ch3.md) maps that argument
  to this method.

## Backlinks

- Depends on: [Pareto Distribution](../distributions/pareto.md),
  [Regular Variation](../theorems/regular-variation.md),
  [Hill Estimator](hill-estimator.md),
  [Tail Threshold Selection](tail-threshold-selection.md),
  [Mean Excess Function](../theorems/mean-excess-function.md), and
  [Pareto Moment Existence](../theorems/pareto-moment-existence.md).
- Used by: [Chapter 3](../../reading-guides/taleb-scoft/ch3.md).

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/methods/plug-in-tail-estimation.md`. Last verified: 2026-09-12. Checked against cited sources, page proof or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
