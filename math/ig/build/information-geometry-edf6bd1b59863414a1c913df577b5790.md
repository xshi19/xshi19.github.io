---
title: Exponential families and sufficient statistics
---

An exponential family separates the data-dependent quantities from their
parameter-dependent coefficients in the log density. This structure makes
sufficiency, likelihood, and the local geometry accessible through one function.

## Begin with a sufficient statistic

Let $X_1,\ldots,X_n$ be independent Bernoulli variables with success probability
$q\in(0,1)$. Their joint mass function is

```{math}
\prod_{i=1}^n q^{x_i}(1-q)^{1-x_i}
=\exp\!\left\{
  \log\frac{q}{1-q}\sum_{i=1}^n x_i+n\log(1-q)
\right\}.
```

The sample enters the parameter-dependent part only through the count
$S_n=\sum_i X_i$. Conditional on $S_n=s$, all binary sequences with $s$
successes have equal probability, independent of $q$. This directly shows
that $S_n$ is sufficient: the conditional distribution of the full data given
the statistic contains no further dependence on the parameter.

Set $\theta=\log(q/(1-q))$. For a single observation,

```{math}
:label: ig-bernoulli-exponential
p_\theta(x)=\exp\{\theta x-\psi(\theta)\},\qquad
\psi(\theta)=\log(1+e^\theta),\qquad x\in\{0,1\}.
```

The subtraction of $\psi$ normalizes the two masses. This calculation motivates
the general form; it does not derive every exponential family from an arbitrary
sufficient statistic.

## The general form and its domain

Fix a measure $\nu$, a nonnegative carrier $h$, and a vector-valued statistic
$t:\mathcal X\to\mathbb R^d$. Define

```{math}
:label: ig-exponential-family
\begin{aligned}
p_\theta(x)&=h(x)\exp\{\theta^\mathsf{T}t(x)-\psi(\theta)\},\\
\psi(\theta)&=\log\int_{\mathcal X}
  h(x)e^{\theta^\mathsf{T}t(x)}\,d\nu(x).
\end{aligned}
```

The **natural parameter domain** consists of parameters where the integral is
positive and finite. We work on its nonempty open interior $\Theta$ and assume
the support $\{h>0\}$ is fixed. On this interior the integral admits local
differentiation of all orders; exponential integrability in a neighborhood
controls the moments of $t(X)$. This is the regular setting used below.

The domain is convex: Hölder's inequality bounds the normalizing integral at
a convex combination of two parameters by the corresponding geometric mean of
their integrals. In particular, $\psi$ is convex.

For independent observations from this family, the density factors as

```{math}
\left(\prod_{i=1}^n h(x_i)\right)
\exp\!\left\{\theta^\mathsf{T}\sum_{i=1}^n t(x_i)-n\psi(\theta)\right\}.
```

The factorization criterion for dominated models therefore makes
$\sum_i t(X_i)$ sufficient. Its dimension stays fixed as $n$ grows. Sufficiency
alone is a broader concept: the full data are always sufficient, and converse
characterizations of exponential families require additional hypotheses.

## Differentiating the normalizer

Differentiating the logarithm and then the normalizing integral gives

```{math}
\begin{aligned}
\partial_i\psi(\theta)
&=e^{-\psi(\theta)}\int_{\mathcal X}
  t_i(x)h(x)e^{\theta^\mathsf{T}t(x)}\,d\nu(x)\\
&=\int_{\mathcal X}t_i(x)p_\theta(x)\,d\nu(x)
=\mathbb E_\theta[t_i(X)].
\end{aligned}
```

Define the expectation parameter $\eta(\theta)=\mathbb E_\theta[t(X)]$.
Since $\partial_jp_\theta(x)=p_\theta(x)(t_j(x)-\eta_j(\theta))$,
a second derivative gives

```{math}
:label: ig-log-partition-moments
\eta=\nabla\psi,\qquad
\partial_i\partial_j\psi
=\mathbb E_\theta[t_i(X)t_j(X)]
 -\mathbb E_\theta[t_i(X)]\mathbb E_\theta[t_j(X)].
```

Thus $\nabla^2\psi=\operatorname{Cov}_\theta(t(X))$ is positive semidefinite.
For any vector $a$,

```{math}
a^\mathsf{T}\nabla^2\psi\,a
=\operatorname{Var}_\theta(a^\mathsf{T}t(X)).
```

The representation is **minimal** if no nonzero $a$ makes $a^\mathsf{T}t(x)$
constant almost everywhere on the support. Under minimality the variance above
is positive for every $a\ne0$, so $\psi$ is strictly convex and its gradient
has an invertible derivative. Natural and expectation parameters are then
smooth local coordinates for the same family. Strict convexity also makes
$\nabla\psi$ injective on $\Theta$; its image must still be distinguished
from boundary moments or the entire set of conceivable moment vectors.

The score, the derivative of the log density, is

```{math}
s_\theta(x)=\nabla_\theta\log p_\theta(x)=t(x)-\eta(\theta).
```

Its covariance is the Fisher information matrix in natural coordinates:
$I(\theta)=\nabla^2\psi$. The
[Fisher note](./information-geometry-fisher-vs-l2.md) derives this metric for
general regular models.

## Two examples

For Bernoulli observations, direct differentiation gives

```{math}
\eta=\psi'(\theta)=\frac{e^\theta}{1+e^\theta}=q,\qquad
\psi''(\theta)=q(1-q).
```

The expectation coordinate is the success probability. Natural coordinates
range over $\mathbb R$, while expectation coordinates range over $(0,1)$.

For the [normal running example](./information-geometry-euclidean-to-manifold.md#a-running-example-normal-location-and-scale),
with mean $\mu\in\mathbb R$ and standard deviation $\sigma>0$, we use
$\theta$ for natural coordinates, as in that note. With Lebesgue measure and
$h(x)=1$, the representation is

```{math}
\begin{aligned}
t(x)&=(x,x^2)^\mathsf{T},\\
\theta_1&=\frac{\mu}{\sigma^2},\qquad
\theta_2=-\frac{1}{2\sigma^2}<0,\\
\psi(\theta)&=-\frac{\theta_1^2}{4\theta_2}
 +\frac12\log\frac{\pi}{-\theta_2}.
\end{aligned}
```

Completing the square in the integral verifies this normalizer. Differentiating
recovers $\eta=(\mu,\mu^2+\sigma^2)^\mathsf{T}$. The expectation domain is
$\eta_2>\eta_1^2$; equality would describe zero variance, outside the family.

This is a full two-dimensional family: the natural coordinates vary over the
open half-plane $\theta_2<0$. Restricting, for example, to
$\sigma^2=1+\mu^2$ produces a one-dimensional curved subfamily in that natural
coordinate space. The ambient potential still exists, but the restricted
model does not automatically inherit the full family's affine coordinates.

## Likelihood becomes moment matching

For observed data, let $\widehat\eta=n^{-1}\sum_i t(x_i)$ be the empirical
sufficient-statistic average. The average log likelihood is

```{math}
\frac1n\sum_{i=1}^n\log p_\theta(x_i)
=\theta^\mathsf{T}\widehat\eta-\psi(\theta)+\text{constant},
```

where the constant is independent of $\theta$.
An interior optimum of a full family must satisfy

```{math}
:label: ig-moment-matching
\nabla\psi(\widehat\theta)=\widehat\eta.
```

The average log likelihood has Hessian $-\nabla^2\psi$, so in a minimal
family an interior solution is the unique maximum. Existence is a separate
question. If all Bernoulli observations are successes, then $\widehat\eta=1$
lies outside $(0,1)$ and
the likelihood supremum is approached as $\theta\to+\infty$; no finite natural
parameter attains it.

For a constrained parameterization $\theta=\theta(\phi)$ with Jacobian
$J=\partial\theta/\partial\phi$, the stationary equation is instead
$J^\mathsf{T}(\widehat\eta-\eta)=0$. Matching every ambient sufficient-statistic
moment is generally too strong. This distinction will matter when an
[EM M-step](./information-geometry-latent-variables-em.md) fits a constrained
joint model.

For further reading, Michael I. Jordan's
[The Exponential Family: Basics](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf),
§§8.3–8.7, treats moment derivatives, sufficiency, and likelihood. That reference
uses different symbols for natural and expectation parameters.

Continue with [latent variables and EM](./information-geometry-latent-variables-em.md),
or take the geometry route to [Fisher information](./information-geometry-fisher-vs-l2.md).
The [hub](./ig/index.md) lists both routes.
