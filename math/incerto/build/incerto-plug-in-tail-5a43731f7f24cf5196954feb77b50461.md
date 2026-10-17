---
title: Plug-In Tail Estimation
options:
  concept:
    id: plug-in-tail-estimation
    type: method
    prerequisites:
      - pareto
      - regular-variation
      - hill-estimator
      - mean-excess-function
      - pareto-moment-existence
    tags:
      - estimation
      - tail-risk
      - moments
---

## Overview

Plug-in tail estimation fits a tail model and evaluates a quantity of interest
under that fitted model. If a distribution family is indexed by a parameter
$\theta$ and the target functional is $T(\theta)$, its plug-in estimate is
$T(\widehat\theta)$. The parameter can include a tail exponent, scale, and tail
probability; an exponent alone generally does not determine the target.

For an exact [Pareto distribution](./incerto-pareto.md), estimating the
exponent from logarithms can give more concentrated mean estimates than the
sample mean. This page establishes a comparison of asymptotic error scales and
explains how finite-sample error comparisons must handle non-finite fits. It does not claim uniformly
smaller mean squared error (MSE). The unregularized plug-in mean has its own
singularity at the mean-existence boundary.

A [regularly varying tail](./incerto-regular-variation.md) only specifies
limiting tail ratios. Using a [Hill estimate](./incerto-hill-estimator.md) in a functional
requires a threshold choice, a tail approximation,
and any body contribution to that functional. The exact model below isolates
the estimation mechanism before these modeling uncertainties enter.

Here $\alpha$ is the tail exponent, $x_m$ is the lower cutoff, and the
[extreme-value index](./incerto-extreme-value-index.md) is $\xi=1/\alpha$.
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
[Pareto moment formula](./incerto-pareto-moment-existence.md) gives

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
[generalized central limit theorem](./incerto-generalized-central-limit-theorem.md)
gives a nondegenerate, non-Gaussian $\alpha$-stable limit for
$n^{1-1/\alpha}(\bar X_n-\mu)$ in this exact Pareto model
[2007](https://doi.org/10.1007/978-0-387-45024-7). That scale decays more slowly than $n^{-1/2}$.
Large observations can continue to move sample means in this regime;
a separate treatment of pre-asymptotic LLN behavior is planned.

### Derivation of the Pareto fit and its normal limit

We derive the known-cutoff fit and the displayed plug-in limit using the
ordinary iid central limit theorem and the delta method. The general limit
theorems are cited from van der Vaart
[1998](https://www.cambridge.org/core/books/asymptotic-statistics/A3C7DAD3F7E66A1FA60E9C8FE132EE1D), Chapters 2–3;
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
## Error-scale comparison

(plugin-pareto-error-comparison)=
At $\alpha=1.5$ and $x_m=1$, $\mu=3$ and the normal-limit variance
coefficient for the plug-in mean is $36$. Its central absolute error scale
is therefore $6/\sqrt n$, or $2/\sqrt n$ relative to $\mu$.
The sample mean's stable-limit scale is proportional to $n^{-1/3}$.
The unknown comparison constants and finite-sample behavior are not supplied
by comparing these powers of $n$.

If a repeated-sample experiment is performed, non-finite fits must count as
infinite errors, including when summarizing quantiles and exceedance
probabilities. The original simulation is not run in this static adaptation;
there is no measured performance table here.

Neither the normal limit nor a finite simulation establishes MSE dominance or
tests whether an empirical dataset is Pareto.

(plugin-tail-body)=
## When only the tail is modeled

Let $X\ge0$ and fix a threshold $u>0$ with tail probability
$q_u=\mathbb P(X>u)>0$. Suppose the conditional tail is exactly Pareto:

$$
\mathbb P(X>x\mid X>u)=(u/x)^\alpha,
\qquad x\ge u,\quad \alpha>1.
$$

The [mean-excess formula](./incerto-mean-excess-function.md) gives
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

A mixture calculation shows why both contributions are needed.

(plugin-body-tail-example)=
Consider a uniform body on $[1,3]$ with probability $0.8$, and a Pareto
tail with cutoff $u=3$ and exponent $\alpha=1.5$ with probability $0.2$.
The body conditional mean is $2$, the tail conditional mean is $9$, and

$$
\mathbb E[X]=0.8\cdot2+0.2\cdot9=3.4.
$$

Using $9$ as the whole-distribution mean would omit the body and tail mass.
Fitting every observation as exact Pareto from $1$ would also impose the
wrong model. This is a population calculation, not a simulated estimator ranking.

For empirical thresholds, repeat the analysis
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
  continuous mapping, and delta method
  [1998](https://www.cambridge.org/core/books/asymptotic-statistics/A3C7DAD3F7E66A1FA60E9C8FE132EE1D).
- Resnick, *Heavy-Tail Phenomena*: regularly varying tails and stable limits
  [2007](https://doi.org/10.1007/978-0-387-45024-7).
- Taleb, *Statistical Consequences of Fat Tails*: motivation for estimating
  functionals through fitted tail parameters [2020](https://arxiv.org/abs/2001.10488). Reading guides are deferred.

## Backlinks

- Depends on: [Pareto Distribution](./incerto-pareto.md),
  [Regular Variation](./incerto-regular-variation.md),
  [Hill Estimator](./incerto-hill-estimator.md),
  Tail Threshold Selection (planned),
  [Mean Excess Function](./incerto-mean-excess-function.md), and
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md).
- Further reading guides are planned.

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/methods/plug-in-tail-estimation.md`, revision `9717c9c`
(2026-09-13 import). Copyright (c) 2023 xshi19. Licensed under MIT.
Links and notation were adapted for this site; executable figures and simulations
were replaced with static calculations. No upstream execution or formal-proof
verification is claimed for this adaptation.

:::{dropdown} MIT permission notice

MIT License

Copyright (c) 2023 xshi19

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
:::
