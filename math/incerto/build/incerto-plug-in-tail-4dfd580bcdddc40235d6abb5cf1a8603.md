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
\begin{gathered}
\mathbb P(X_i>x)=\left(\frac{x_m}{x}\right)^\alpha,\\
x\ge x_m>0,
\end{gathered}
$$

where $x_m$ is known and fixed, and the true exponent satisfies $\alpha>1$.
These are assumptions about the entire distribution. The
[Pareto moment formula](./incerto-pareto-moment-existence.md) gives

$$
\mu=\frac{\alpha x_m}{\alpha-1}=\frac{x_m}{1-\xi}.
$$

The maximum-likelihood estimate of the exponent and its reciprocal are

$$
\begin{gathered}
\widehat\xi_n=\frac1n\sum_{i=1}^n\log(X_i/x_m),\\
\widehat\alpha_n=\frac1{\widehat\xi_n}.
\end{gathered}
$$

They use all observations above a known cutoff. Hill instead averages the top
$k$ log-excesses above the random order-statistic threshold $X_{n-k:n}$.
The formulas share a log-excess mechanism, but their sample sizes and threshold
assumptions differ.

The plug-in mean is

$$
\begin{gathered}
\begin{aligned}
\widehat\mu_n&=\frac{x_m}{1-\widehat\xi_n}\\
&=\frac{\widehat\alpha_n x_m}{\widehat\alpha_n-1},
\end{aligned}\\
\widehat\xi_n<1.
\end{gathered}
$$

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

An all-cutoff sample uses the limiting mean $x_m$; a fit with
$\widehat\xi_n\ge1$ has no finite mean.

(plugin-pareto-error-scale)=
## Why the logarithmic fit can improve concentration

For each fixed $\alpha>1$, the finite plug-in estimates have the asymptotic law

$$
\begin{gathered}
\mathcal L\!\left(\sqrt n(\widehat\mu_n-\mu)
\mid \widehat\xi_n<1\right)\\
\Rightarrow\mathcal N\!\left(0,\frac{x_m^2\alpha^2}{(\alpha-1)^4}\right),\\
n\to\infty.
\end{gathered}
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
@resnick2007heavy. That scale decays more slowly than $n^{-1/2}$.
Large observations can continue to move sample means in this regime;
[pre-asymptotic LLN behavior](./incerto-lln-preasymptotic.md) illustrates this effect.

### Derivation of the Pareto fit and its normal limit

We derive the known-cutoff fit and the displayed plug-in limit using the
ordinary iid central limit theorem and the delta method. The general limit
theorems are cited from van der Vaart
[1998](https://www.cambridge.org/core/books/asymptotic-statistics/A3C7DAD3F7E66A1FA60E9C8FE132EE1D), Chapters 2–3;
the stable limit for the sample mean is cited above.

Set $Y_i=\log(X_i/x_m)$. For $y\ge0$,

$$
\begin{aligned}
\mathbb P(Y_i>y)&=\mathbb P(X_i>x_m e^y)\\
&=e^{-\alpha y}.
\end{aligned}
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
\begin{aligned}
[h'(\xi)]^2\xi^2&=\frac{x_m^2\xi^2}{(1-\xi)^4}\\
&=\frac{x_m^2\alpha^2}{(\alpha-1)^4}.
\end{aligned}
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

In a repeated-sample experiment with a finite true mean, non-finite fits count
as infinite errors. Removing them changes the sampling law. Below we report
fit frequencies, not an unconditional MSE or a finite-only error comparison.

(plugin-singularity-figure)=
## The singularity and the frequency of infinite fits

```{figure} ../assets/figures/incerto/plugin-mean-singularity.svg
:alt: The fitted Pareto mean diverges as the fitted exponent approaches one from above. A companion panel counts infinite fits at true exponents 0.8, 1 and 1.5.

Top: the population functional $\widehat\alpha/(\widehat\alpha-1)$
on $\widehat\alpha>1$, with the infinite-mean region shaded. No negative
algebraic values are interpreted as means below one. Bottom: exact
infinite-fit probabilities (lines) and frequencies from 600 iid samples per
size and law (dots), $n=10,30,100,300$, true $\alpha=0.8,1,1.5$ and
$x_m=1$. PCG64 seeds 20261090–20261092. Finite-fit frequencies are the
complements; every sample is counted. Only the true $\alpha=1.5$ law has
a finite target mean. These are fit-status frequencies, not error bars.
```

For any true $\alpha>0$, the earlier logarithmic transformation gives
$T_n=\sum_i\log(X_i/x_m)\sim\mathrm{Gamma}(n,\text{rate }\alpha)$.
The fit is infinite exactly when $T_n\ge n$. Thus

$$
P(\widehat\mu_n=\infty)=P(T_n\ge n).
$$

The Gamma law follows by multiplying the exponential MGFs
$(\alpha/(\alpha-t))^n$; the boundary event has probability zero.
This derives the exact probability used in the figure. A finite estimate
can occur even when the true mean is infinite. At the boundary $\alpha=1$,
the ordinary CLT for the log-excesses gives an infinite-fit probability
tending to $1/2$.

Use [`pareto_plugin_nonfinite_probability`](incerto-api-reference.md#api-pareto-plugin-nonfinite-probability).

```python
from xmath.incerto.limits import pareto_plugin_nonfinite_probability

for alpha in (.8, 1, 1.5):
    p = pareto_plugin_nonfinite_probability(alpha, 30)
    print(f"alpha={alpha:g}: finite={1-p:.4g}, infinite={p:.4g}")
```

**Output**

```text
alpha=0.8: finite=0.1321, infinite=0.8679
alpha=1: finite=0.5243, infinite=0.4757
alpha=1.5: finite=0.9927, infinite=0.007337
```

Regenerate with `uv run python demos/incerto/visualization_p2.py`.

Neither the normal limit nor a finite simulation establishes MSE dominance or
tests whether an empirical dataset is Pareto.

(plugin-tail-body)=
## When only the tail is modeled

Let $X\ge0$ and fix a threshold $u>0$ with tail probability
$q_u=\mathbb P(X>u)>0$. Suppose the conditional tail is exactly Pareto:

$$
\begin{gathered}
\mathbb P(X>x\mid X>u)=(u/x)^\alpha,\\
x\ge u,\quad \alpha>1.
\end{gathered}
$$

The [mean-excess formula](./incerto-mean-excess-function.md) gives
$\mathbb E[X\mid X>u]=u+e(u)=u\alpha/(\alpha-1)$, where
$e(u)=\mathbb E[X-u\mid X>u]$. Splitting the expectation at $u$ yields

$$
\begin{aligned}
\mu&=\mathbb E[X\mathbf1_{\{X\le u\}}]\\
&\quad+q_u\frac{u\alpha}{\alpha-1}.
\end{aligned}
$$

For an iid sample, let $k=\sum_i\mathbf1_{\{X_i>u\}}$ be the exceedance count.
Estimate $q_u$ by $k/n$ and fit $\widehat\alpha_u$ from those exceedances.
When $k>0$ and $\widehat\alpha_u>1$, a body-plus-tail estimate is

$$
\begin{aligned}
\widehat\mu_u&=\frac1n\sum_{i=1}^n X_i\mathbf1_{\{X_i\le u\}}\\
&\quad+\frac{k}{n}\frac{u\widehat\alpha_u}{\widehat\alpha_u-1}.
\end{aligned}
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

(plugin-hidden-tail)=
## A missing contribution requires a tail model

The [hidden-tail result](incerto-hidden-tail.md#hidden-tail-pareto) gives,
for an exact Pareto parent and $\alpha>1$,

$$
\begin{gathered}
T_1(t)=\frac{\alpha x_m}{\alpha-1}
       \left(\frac{x_m}{t}\right)^{\alpha-1},\\
t\ge x_m.
\end{gathered}
$$

Replacing $\alpha$ by an estimate turns this into a model-based tail
contribution. It does not make the true population's unobserved tail
empirical. With a tail-only model, its fitted mass and body contribution
must also be included, as in the preceding decomposition. In particular,
adding the true-model $T_1(M_n)$ to a sample mean produces upward bias:
a finite-mean iid sample mean is already unbiased, and the added term has
a strictly positive expectation.

(plugin-exponent-truncation)=
## Debiasing, conditional selection, and clipping

The [inverse-gamma law](incerto-hill-estimator.md#hill-sampling-law)
provides exact calibration for a known Pareto cutoff. Let
$A=(m-1)\widehat\alpha/m$, $m>1$, and fix $h>1$. Its density $f_A$
has inverse-gamma shape $m$ and scale $(m-1)\alpha$. Conditioning on a
retained estimate has density

$$
f_{A\mid A>h}(a)=\frac{f_A(a)}{P(A>h)},\qquad a>h.
$$

This is a conditional law, not the original estimator's unconditional
sampling law. Clipping $A$ to $\max(A,h)$ instead gives an atom of size
$P(A\le h)$ at $h$ and retains the original density above it. Either
choice changes the estimator or the samples being reported. Under
conditioning, the plug-in mean is bounded above by $x_mh/(h-1)$;
clipping attains this bound at its atom. Neither operation preserves
unbiasedness or establishes mean-squared-error superiority.

Use [`pareto_exponent_sampling`](incerto-api-reference.md#api-pareto-exponent-sampling)
to expose the selection mass instead of silently omitting estimates:

```python
from xmath.incerto.extremes import pareto_exponent_sampling

law = pareto_exponent_sampling(1.2, 20, debiased=True)
for h in (1, 1.1, 1.3):
    print(f"h={h:g}: retained={law.sf(h):.4g}, excluded or clipped={law.cdf(h):.4g}")
```

**Output**

```text
h=1: retained=0.7495, excluded or clipped=0.2505
h=1.1: retained=0.5929, excluded or clipped=0.4071
h=1.3: retained=0.3088, excluded or clipped=0.6912
```

Appendix E, pp. 189–191, especially E.1, is the SCoFT v4 source context
[@taleb2025scoftv4]. Its finite-sample calibration must be restricted to
an exact Pareto model, with known cutoff or the fixed-$k$ Hill experiment.
A general slowly varying factor need not be constant. The earlier
singularity proof still rules out finite unconditional MSE for the
unregularized plug-in mean, even when central error quantiles look small.

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
  @resnick2007heavy.
- Taleb, *Statistical Consequences of Fat Tails*: motivation for estimating
  functionals through fitted tail parameters, v4 Appendix E, pp. 189–191
  [@taleb2025scoftv4]. See the [SCoFT reading guide](./incerto-scoft-intro.md).

## Backlinks

- Depends on: [Pareto Distribution](./incerto-pareto.md),
  [Regular Variation](./incerto-regular-variation.md),
  [Hill Estimator](./incerto-hill-estimator.md),
  [Tail Threshold Selection](./incerto-tail-threshold-selection.md),
  [Mean Excess Function](./incerto-mean-excess-function.md), and
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md).
- Reading guides: [SCoFT](./incerto-scoft-intro.md) and
  [external references](./incerto-external-reading.md).
