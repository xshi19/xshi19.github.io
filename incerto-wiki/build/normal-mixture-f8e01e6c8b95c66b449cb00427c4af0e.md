---
kernelspec:
  name: python3
  display_name: Python 3
options:
  concept:
    id: normal-mixture
    type: distribution
    depends_on: []
    tags:
      - mixture
      - volatility
---

# Normal Variance Mixture

## Statement

A normal variance mixture is a random variable of the form

$$
X=\sqrt{V}\,Z,
$$

where $Z\sim N(0,1)$ and $V>0$ is an independent random variance factor.
Conditional on $V=v$, the distribution is normal with variance $v$ and
standard deviation $\sqrt v$.

In general, its density and characteristic function are

$$
f_X(x)
=
\mathbb E\left[
\frac{1}{\sqrt{2\pi V}}
\exp\left(-\frac{x^2}{2V}\right)
\right],
\qquad
\varphi_X(t)=\mathbb E\left[e^{-t^2V/2}\right].
$$

Whenever the moment exists,

$$
\mathbb E[X^{2m}]
=
(2m-1)!!\,\mathbb E[V^m].
$$

If $\mathbb E[V]<\infty$, the excess kurtosis is

$$
\operatorname{ExKurt}(X)
=
3\frac{\operatorname{Var}(V)}{\mathbb E[V]^2}.
$$

A simple two-component mixture takes

$$
V_a=
\begin{cases}
1-a, & \text{with probability }1/2,\\
1+a, & \text{with probability }1/2,
\end{cases}
\qquad 0\le a<1.
$$

Its density is

$$
f_a(x)
=
\frac12\phi_{\sqrt{1-a}}(x)
+
\frac12\phi_{\sqrt{1+a}}(x),
$$

where $\phi_\sigma$ is the density of $N(0,\sigma^2)$.  This distribution has

$$
\mathbb E[X]=0,\qquad \operatorname{Var}(X)=1,
$$

but its fourth moment is

$$
\mathbb E[X^4]=3(1+a^2).
$$

Thus the excess kurtosis is $3a^2$ even though the variance is held fixed.

The symbols $f$, $\mathbb E$, and $\operatorname{Var}$ follow the shared
[notation table](../../notation/index.md).  The mixing variable $V$ and the
spread parameter $a$ are local to this page; $\phi_\sigma$ denotes the
$N(0,\sigma^2)$ density.

## Examples of mixing distributions

- If $a=0$, then $V_a=1$ always, so the mixture collapses to the standard
  normal distribution.
- If $a=0.8$, the variance is still $1$, but the fourth moment is
  $3(1+0.8^2)=4.92$ rather than the Gaussian value $3$.
- This finite normal mixture sharpens the center, thins an intermediate
  [shoulder](../methods/body-shoulder-tail.md) band, and thickens the far tails
  relative to a standard Gaussian.  It is still not
  [regularly varying](../theorems/regular-variation.md) and not a
  [Pareto tail](pareto.md).

## Why mixtures create heavy-looking tails

Mixing variances is the smallest way to leave the single-Gaussian world without
leaving normal conditional behavior.  Some observations come from narrower
Gaussian components and some from wider components.  With the mean and variance
held fixed, the combined distribution sharpens the central peak, removes
density from an intermediate shoulder band, and adds density in sufficiently
far tails.

This is Gaussian-mixture tail thickening, not a strict heavy-tail mechanism.
It can make Gaussian diagnostics look too optimistic, but a finite normal
mixture is not a [Pareto](pareto.md) or
[regularly varying](../theorems/regular-variation.md) tail, and its moment
generating function is finite for every real argument.  The far tail is
eventually controlled by the largest Gaussian variance in the mixture.

The mixing law determines the tail class:

| Mixing law for $V$ | Resulting tail behavior |
| --- | --- |
| Finite support | Gaussian-type far tail controlled by the largest variance. |
| Inverse-gamma | Student-$t$ type, regularly varying tail. |
| Gamma, with no mean shift | Symmetric variance-gamma type, exponential tail. |

```{code-cell} python
:label: normal-mixture-density-plot
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

from incerto.distributions import simple_normal_mixture
from incerto.figures import FIGURE_SIZES, set_theme, style_axes

set_theme()
x_grid = np.linspace(-6, 6, 600)
normal_pdf = norm.pdf(x_grid)
normal_logpdf = norm.logpdf(x_grid)


def density_difference(z):
    return simple_normal_mixture.pdf(z, a=0.8) - norm.pdf(z)


crossings = np.array(
    [
        brentq(density_difference, 0.05, 1.0),
        brentq(density_difference, 1.0, 3.5),
    ]
)

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"])
axes[0].plot(x_grid, normal_pdf, label="standard normal")
for a in (0.4, 0.8):
    mixture_pdf = simple_normal_mixture.pdf(x_grid, a)
    axes[0].plot(x_grid, mixture_pdf, label=fr"mixture $a={a}$")
    log_ratio = np.log(mixture_pdf) - normal_logpdf
    axes[1].plot(x_grid, np.exp(log_ratio), label=fr"$a={a}$")

for crossing in crossings:
    axes[0].axvline(crossing, color="black", linestyle=":", linewidth=0.9)
    axes[0].axvline(-crossing, color="black", linestyle=":", linewidth=0.9)
    axes[1].axvline(crossing, color="black", linestyle=":", linewidth=0.9)
    axes[1].axvline(-crossing, color="black", linestyle=":", linewidth=0.9)

axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
axes[1].set_yscale("log")

axes[0].set_xlabel("x")
axes[0].set_ylabel("density")
axes[0].set_title("Fixed-variance mixture")
axes[0].legend()

axes[1].set_xlabel("x")
axes[1].set_ylabel(r"$f_a(x)/\phi(x)$")
axes[1].set_title("Density ratio to standard normal")
axes[1].legend()

style_axes(axes, grid_axis="both")
plt.show()

print(f"Density crossings for a=0.8: +/-{np.round(crossings, 4)}")
```

**What to notice.** Increasing $a$ sharpens the center, removes density from an
intermediate shoulder band, and adds density in the far tails while preserving
variance one.  This is variance heterogeneity, not a proof of a Pareto tail.

## Mixture derivation

We derive the displayed two-component mixture formulas explicitly.  The general
conditional-density, characteristic-function, and even-moment identities in the
statement follow from the same conditioning argument.  For the far tail, we
show why the largest Gaussian variance controls this finite mixture.

The density formula follows from conditioning on $V_a$:

$$
f_a(x)
=
\mathbb E[f_{X\mid V_a}(x\mid V_a)]
=
\frac12\phi_{\sqrt{1-a}}(x)
+
\frac12\phi_{\sqrt{1+a}}(x).
$$

Because $Z$ has mean zero and is independent of $V_a$,

$$
\mathbb E[X]=\mathbb E[\sqrt{V_a}]\,\mathbb E[Z]=0.
$$

The variance is

$$
\operatorname{Var}(X)
=
\mathbb E[V_a Z^2]
=
\mathbb E[V_a]\mathbb E[Z^2]
=
\frac{(1-a)+(1+a)}2
=1.
$$

For the fourth moment, use $\mathbb E[Z^4]=3$:

$$
\mathbb E[X^4]
=
\mathbb E[V_a^2]\mathbb E[Z^4]
=
3\frac{(1-a)^2+(1+a)^2}{2}
=3(1+a^2).
$$

Finally, for fixed $a>0$, the component with variance $1+a$ dominates the
absolute far tail.  Relative to the standard normal density $\phi_1$,

$$
\frac{f_a(x)}{\phi_1(x)}
\sim
\frac{1}{2\sqrt{1+a}}
\exp\left(\frac{a x^2}{2(1+a)}\right)
$$

as $|x|\to\infty$.  The ratio diverges, but the density still decays like
$\exp[-x^2/(2(1+a))]$ up to a constant factor, so it is lighter than a
power-law tail.

## Tail comparison and numerical check

The implementation lives in `incerto.distributions.simple_normal_mixture`.

```{code-cell} python
:tags: [hide-input]
:label: normal-mixture-python-check

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from incerto.distributions import simple_normal_mixture

a = 0.8
x = np.array([0.0, 2.0, 4.0, 6.0])

mixture_pdf = simple_normal_mixture.pdf(x, a)
log_density_ratio = np.log(mixture_pdf) - norm.logpdf(x)
density_ratio = np.round(np.exp(log_density_ratio), 3)
mean = float(simple_normal_mixture.mean(a))
variance = float(simple_normal_mixture.var(a))
fourth_moment = 3 * (1 + a**2)

normalization, _ = quad(lambda z: simple_normal_mixture.pdf(z, a), -np.inf, np.inf)
second_moment, _ = quad(
    lambda z: z**2 * simple_normal_mixture.pdf(z, a),
    -np.inf,
    np.inf,
)
fourth_moment_numeric, _ = quad(
    lambda z: z**4 * simple_normal_mixture.pdf(z, a),
    -np.inf,
    np.inf,
)

np.testing.assert_allclose(simple_normal_mixture.pdf(x, 0.0), norm.pdf(x))
np.testing.assert_allclose(normalization, 1.0, rtol=1e-10)
np.testing.assert_allclose(second_moment, variance, rtol=1e-10)
np.testing.assert_allclose(fourth_moment_numeric, fourth_moment, rtol=1e-10)

print(f"Density ratio f_a(x)/phi(x) at x={x.tolist()}: {density_ratio}")
print(f"Mean: {mean:.1f}")
print(f"Variance: {variance:.1f}")
print(f"Fourth moment: {fourth_moment:.3f}")
```

The density ratio is above one near the center, below one in the intermediate
shoulder region, and then grows rapidly in the far tail.  The mean and
variance remain fixed while the fourth moment rises with the variance-mixing
parameter.

## Caveats

- A finite normal mixture has light tails in the strict asymptotic sense.  It is
  heavier than a reference Gaussian in the far-tail density ratio, not
  [regularly varying](../theorems/regular-variation.md).
- The simple two-component mixture requires $0\le a<1$.  At $a=1$, one
  component degenerates at zero and the ordinary density description changes.
- High kurtosis in a normal mixture can come from variance heterogeneity rather
  than from a Pareto tail.  Do not infer a power law from kurtosis alone.
- Real stochastic-volatility models usually use a continuous mixing
  distribution and dependence across time.  We isolate the iid one-step
  distribution.

## References

- Barndorff-Nielsen, Kent, and Sorensen, "Normal Variance-Mean Mixtures and
  z Distributions" [@barndorff1982normal].

## Backlinks

- Depends on: the canonical density and moment notation in
  [Notation](../../notation/index.md).
- Used by: [Variance Gamma Distribution](variance-gamma.md),
  [Body, Shoulders, and Tails](../methods/body-shoulder-tail.md), and future
  variance-mixture examples.

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/distributions/normal-mixture.md`. Last verified: 2026-06-25. Checked against cited sources, scoped mixture derivations, and executable density-ratio diagnostics.
:::
<!-- incerto-provenance:end -->
