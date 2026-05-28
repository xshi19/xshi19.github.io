---
kernelspec:
  name: python3
  display_name: Python 3
---

# Variance Gamma Distribution

## Statement

In the repository's parameterization, let

$$
\alpha>|\beta|,\qquad \lambda>0,\qquad
\theta=\frac{2}{\alpha^2-\beta^2}.
$$

Let $G\sim\operatorname{Gamma}(\lambda,\theta)$ use shape $\lambda$ and scale
$\theta$, and let $Z\sim N(0,1)$ be independent.  The univariate
variance-gamma variable is

$$
X=\beta G+\sqrt{G}\,Z.
$$

For $x\ne0$, its density is

$$
f(x)
=
\frac{(\alpha^2-\beta^2)^\lambda |x|^{\lambda-1/2}
K_{\lambda-1/2}(\alpha |x|)}
{\sqrt{\pi}\,\Gamma(\lambda)(2\alpha)^{\lambda-1/2}}
e^{\beta x},
$$

where $K_\nu$ is the modified Bessel function of the second kind.  Its mean and
variance are

$$
\mathbb E[X]=\beta\lambda\theta,\qquad
\operatorname{Var}(X)=\lambda\theta(1+\beta^2\theta).
$$

When $\beta=0$, the distribution is symmetric.  Its tails decay exponentially,
not as a power law.

## Intuition

Variance-gamma is a continuous normal variance mixture.  The observation is
normal after conditioning on a random gamma clock, but the random clock creates
more central mass and heavier shoulders than a single Gaussian.  The parameter
$\lambda$ controls the gamma clock's shape, $\alpha$ controls tail decay, and
$\beta$ tilts the distribution to the right or left.

This is a useful Chapter 4 bridge: stochastic variance creates visibly fatter
tails than the Gaussian baseline, but it is still not the same object as a
Pareto tail.  The model is "semi-heavy": heavier than Gaussian in many finite
sample diagnostics, yet all ordinary moments remain finite under this
parameterization.

## Proof

Condition on $G$.  Since $X\mid G=g$ is normal with mean $\beta g$ and variance
$g$,

$$
\mathbb E[e^{tX}\mid G]
=
\exp\left((\beta t+t^2/2)G\right).
$$

The moment generating function of a gamma variable with shape $\lambda$ and
scale $\theta$ is $(1-\theta s)^{-\lambda}$ where it is finite.  Therefore

$$
\mathbb E[e^{tX}]
=
\left(1-\theta\beta t-\frac{\theta t^2}{2}\right)^{-\lambda}.
$$

Differentiating at $t=0$ gives

$$
\mathbb E[X]=\beta\lambda\theta.
$$

The variance can also be obtained from conditional variance:

$$
\operatorname{Var}(X)
=
\mathbb E[\operatorname{Var}(X\mid G)]
+
\operatorname{Var}(\mathbb E[X\mid G])
=
\mathbb E[G]+\operatorname{Var}(\beta G)
=
\lambda\theta+\beta^2\lambda\theta^2.
$$

The Bessel density is the closed form obtained by integrating the conditional
normal density over the gamma mixing density.  For tail behavior, use

$$
K_\nu(z)\sim \sqrt{\frac{\pi}{2z}}e^{-z}
$$

as $z\to\infty$.  Hence the right tail density has exponential rate
$\alpha-\beta$, and the left tail density has exponential rate $\alpha+\beta$.
This proves the distribution is not regularly varying.

## Python

The implementation lives in `incerto.distributions.univariate_variance_gamma`.

```{code-cell} python
:label: variance-gamma-python-check

import numpy as np

from incerto.distributions import univariate_variance_gamma

alpha = 1.2
beta = -0.5
lam = 2.0

mean, variance = univariate_variance_gamma.stats(
    alpha, beta, lam, moments="mv"
)

rng = np.random.default_rng(20260523)
sample = univariate_variance_gamma.rvs(
    alpha, beta, lam, size=200_000, random_state=rng
)
density_points = np.array([-3.0, 0.0, 3.0])
density_values = univariate_variance_gamma.pdf(
    density_points, alpha, beta, lam
)

print(
    f"Theoretical mean and variance: {float(mean):.4f}, "
    f"{float(variance):.4f}"
)
print(f"Simulated mean and variance: {np.mean(sample):.4f}, {np.var(sample):.4f}")
print(f"PDF at x={density_points.tolist()}: {np.round(density_values, 6)}")
```

The simulated mean and variance should be close to the theoretical values.  The
asymmetric density values show the effect of $\beta<0$.

## Caveats

- Parameterizations vary across books and libraries.  This page uses the
  parameterization implemented in `incerto.distributions`.
- The Bessel density needs a limiting value at $x=0$ when $\lambda>1/2$; the
  implementation handles that numerical boundary explicitly.
- Variance-gamma has heavier shoulders than a Gaussian, but it is not a
  Pareto-type model.  Hill estimates and moment-existence claims designed for
  regularly varying tails do not apply directly.
- In financial modeling, variance-gamma is often used as a process with
  independent increments.  This page only states the one-step distribution.
- Dependence, volatility clustering, and time aggregation are separate modeling
  assumptions.

## References

- Madan and Seneta, "The Variance Gamma (V.G.) Model for Share Market Returns"
  [@madan1990variance].
- Barndorff-Nielsen, Kent, and Sorensen, "Normal Variance-Mean Mixtures and
  z Distributions" [@barndorff1982normal].
- Taleb, *Statistical Consequences of Fat Tails*, Chapter 4
  [@taleb2020scoft].

## Backlinks

- Depends on: [Normal Variance Mixture](normal-mixture.md) and the canonical
  density and moment notation in [Notation](../../notation/index.md).
- Used by: [Dispersion Ratio Under Fat Tails](../examples/dispersion-ratio.md)
  and the planned Chapter 4 reading guide.

<!-- incerto-provenance:start -->
:::{div} incerto-provenance
**Provenance.** Source: `content/concepts/distributions/variance-gamma.md`. Last verified: 2026-05-24. Checked against cited sources, page proof or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
