---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: variance-gamma
    type: distribution
    depends_on:
      - normal-mixture
    tags:
      - mixture
      - volatility
---

# Variance Gamma Distribution

## Statement

In one convenient parameterization, let

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

This is a normal variance-mean mixture: conditional on $G=g$,
$X\mid G=g\sim N(\beta g,g)$.  When $\beta=0$, it reduces to a pure
[normal variance mixture](normal-mixture.md).

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

Its moment generating function is

$$
M_X(t)
=
\left(1-\theta\beta t-\frac{\theta t^2}{2}\right)^{-\lambda},
\qquad
-\alpha-\beta<t<\alpha-\beta.
$$

When $\beta=0$, the distribution is symmetric.  Its tails decay exponentially,
not as a power law.

## Shape intuition

Variance-gamma is a continuous normal variance-mean mixture.  The observation
is normal after conditioning on a random gamma clock, but the random clock
changes both conditional variance and, when $\beta\ne0$, conditional mean.  The
parameter $\lambda$ controls the gamma clock's shape, $\alpha$ controls tail
decay, and $\beta$ tilts the distribution to the right or left.

Relative to its own moment-matched Gaussian, a typical variance-gamma
parameter choice has a sharper center and heavier far tails.  It is still not
the same object as a Pareto tail.  The model is "semi-heavy": heavier than
Gaussian in many finite-sample diagnostics, yet all ordinary moments remain
finite under this parameterization.

```{code-cell} python
:label: variance-gamma-density-plot
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from incerto.distributions import univariate_variance_gamma
from incerto.figures import COLORS, FIGURE_SIZES, set_theme, style_axes

set_theme()
x_grid = np.linspace(-8, 8, 600)
y_grid = np.linspace(-5, 5, 600)
alpha = 1.4
lam = 1.5
betas = (-0.4, 0.0, 0.4)
colors = [COLORS["green"], COLORS["accent"], COLORS["teal"]]

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"])
moment_rows = []

for beta, color in zip(betas, colors):
    mean, variance = univariate_variance_gamma.stats(
        alpha,
        beta,
        lam,
        moments="mv",
    )
    mean = float(mean)
    variance = float(variance)
    sd = np.sqrt(variance)
    moment_rows.append((beta, mean, variance))

    axes[0].plot(
        x_grid,
        univariate_variance_gamma.pdf(x_grid, alpha, beta, lam),
        color=color,
        label=fr"VG $\beta={beta}$",
    )
    axes[0].plot(
        x_grid,
        norm.pdf(x_grid, loc=mean, scale=sd),
        color=color,
        linestyle="--",
        linewidth=1.0,
        label=fr"matched normal $\beta={beta}$",
    )

    standardized_pdf = sd * univariate_variance_gamma.pdf(
        mean + sd * y_grid,
        alpha,
        beta,
        lam,
    )
    axes[1].plot(y_grid, standardized_pdf, color=color, label=fr"$\beta={beta}$")

axes[1].plot(
    y_grid,
    norm.pdf(y_grid),
    color="black",
    linestyle="--",
    linewidth=1.0,
    label="standard normal",
)

axes[0].set_xlabel("x")
axes[0].set_ylabel("density")
axes[0].set_title("VG and moment-matched Gaussians")
axes[0].legend(fontsize=8)

axes[1].set_xlabel("standardized x")
axes[1].set_ylabel("density")
axes[1].set_title("Standardized shape comparison")
axes[1].legend()

style_axes(axes)
plt.show()

print("Moment-matched Gaussian parameters")
print(f"{'beta':>6} {'mean':>10} {'variance':>10}")
for beta, mean, variance in moment_rows:
    print(f"{beta:6.2f} {mean:10.3f} {variance:10.3f}")
```

**What to notice.** The comparison uses each variance-gamma curve's own mean
and variance.  The left panel keeps the skew and location visible; the right
panel removes mean and variance effects so the sharper center and heavier far
tails are shape features rather than artifacts of a mismatched benchmark.

## Gamma-normal mixture derivation

We derive the moment generating function, moment formulas, and exponential
tail rates from the gamma-normal mixture representation.  The Bessel density
uses the displayed standard integral identity, and the origin behavior is cited
as a standard consequence of Bessel-function asymptotics.

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

Because $\theta=2/(\alpha^2-\beta^2)$, this MGF is finite exactly on

$$
-\alpha-\beta<t<\alpha-\beta.
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

The Bessel density follows by integrating the conditional normal density over
the gamma mixing density and using

$$
\int_0^\infty
g^{\nu-1}e^{-a/g-bg}\,dg
=
2\left(\frac ab\right)^{\nu/2}
K_\nu(2\sqrt{ab}),
$$

with

$$
\nu=\lambda-\frac12,\qquad
a=\frac{x^2}{2},\qquad
b=\frac{\alpha^2}{2}.
$$

The special case $\beta=0,\lambda=1$ gives the Laplace density
$(\alpha/2)e^{-\alpha|x|}$.

For tail behavior, use

$$
K_\nu(z)\sim \sqrt{\frac{\pi}{2z}}e^{-z}
$$

as $z\to\infty$.  Hence

$$
f(x)\sim C_+x^{\lambda-1}e^{-(\alpha-\beta)x},
\qquad x\to\infty,
$$

and

$$
f(-x)\sim C_-x^{\lambda-1}e^{-(\alpha+\beta)x},
\qquad x\to\infty,
$$

for positive constants $C_+$ and $C_-$.  The right exponential rate is
$\alpha-\beta$, and the left exponential rate is $\alpha+\beta$.  This proves
the distribution is not [regularly varying](../theorems/regular-variation.md).

At the origin, the finite-density case is

$$
\lambda>\frac12:
\quad
f(0)=
\frac{(\alpha^2-\beta^2)^\lambda
\Gamma(\lambda-\tfrac12)}
{2\sqrt\pi\,\Gamma(\lambda)\alpha^{2\lambda-1}}.
$$

When $\lambda=1/2$, the density has a logarithmic singularity at zero.  When
$0<\lambda<1/2$, it has an integrable power singularity.

## Tail comparison and simulation check

The implementation lives in `incerto.distributions.univariate_variance_gamma`.

```{code-cell} python
:tags: [hide-input]
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
sample_size = 200_000
sample = univariate_variance_gamma.rvs(
    alpha, beta, lam, size=sample_size, random_state=rng
)
density_points = np.array([-3.0, 0.0, 3.0])
density_values = univariate_variance_gamma.pdf(
    density_points, alpha, beta, lam
)
sample_mean = float(np.mean(sample))
sample_variance = float(np.var(sample, ddof=1))
mean_se = float(np.std(sample, ddof=1) / np.sqrt(sample_size))
variance_mc_variance = (
    np.mean((sample - sample_mean) ** 4)
    - ((sample_size - 1) / sample_size * sample_variance) ** 2
)
variance_se = float(np.sqrt(max(variance_mc_variance, 0.0)) / np.sqrt(sample_size))

tail_x = np.linspace(8.0, 18.0, 80)
right_log_density = np.log(
    univariate_variance_gamma.pdf(tail_x, alpha, beta, lam)
)
left_log_density = np.log(
    univariate_variance_gamma.pdf(-tail_x, alpha, beta, lam)
)
right_slope = np.polyfit(tail_x[-30:], right_log_density[-30:], 1)[0]
left_slope = np.polyfit(tail_x[-30:], left_log_density[-30:], 1)[0]

print(
    f"Theoretical mean and variance: {float(mean):.4f}, "
    f"{float(variance):.4f}"
)
print(
    f"Simulated mean: {sample_mean:.4f} +/- {1.96 * mean_se:.4f} "
    "(approx. 95% MC)"
)
print(
    f"Simulated variance: {sample_variance:.4f} +/- {1.96 * variance_se:.4f} "
    "(approx. 95% MC)"
)
print(f"PDF at x={density_points.tolist()}: {np.round(density_values, 6)}")
print(
    "Log-density tail slopes, right/left: "
    f"{right_slope:.3f}, {left_slope:.3f}; "
    f"targets {- (alpha - beta):.3f}, {- (alpha + beta):.3f}"
)
```

The simulated mean and variance should be close to the theoretical values
within Monte Carlo error.  The asymmetric density values show the effect of
$\beta<0$, and the log-density slopes approach the right and left exponential
tail rates.

## Caveats

- Parameterizations vary across books and libraries.  We use the
  parameterization implemented in `incerto.distributions`.
- The Bessel density needs a limiting value at $x=0$ when $\lambda>1/2$; at
  and below $\lambda=1/2$, the density is singular but still integrable under
  the stated parameter restrictions.
- Variance-gamma can have a sharper center and heavier far tails than its
  moment-matched Gaussian, but it is not a
  [Pareto-type model](pareto.md).  Hill estimates and moment-existence claims
  designed for regularly varying tails do not apply directly.
- In financial modeling, variance-gamma is often used as a process with
  independent increments.  We state only the one-step distribution.
- Dependence, volatility clustering, and time aggregation are separate modeling
  assumptions.

## References

- Madan and Seneta, "The Variance Gamma (V.G.) Model for Share Market Returns"
  [@madan1990variance].
- Barndorff-Nielsen, Kent, and Sorensen, "Normal Variance-Mean Mixtures and
  z Distributions" [@barndorff1982normal].

## Backlinks

- Depends on: [Normal Variance Mixture](normal-mixture.md) and the canonical
  density and moment notation in [Notation](../../notation/index.md).
- Used by: [Dispersion Ratio Under Fat Tails](../examples/dispersion-ratio.md)
  and future stochastic-variance examples.

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/distributions/variance-gamma.md`. Last verified: 2026-06-25. Checked against cited sources, scoped mixture derivation, and executable examples.
:::
<!-- incerto-provenance:end -->
