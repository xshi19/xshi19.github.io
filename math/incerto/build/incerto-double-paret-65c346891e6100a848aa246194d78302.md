---
title: Symmetric Shifted Double Pareto Distribution
options:
  concept:
    id: double-pareto
    type: distribution
    depends_on:
      - pareto
      - regular-variation
    tags:
      - fat-tails
      - two-sided
---

## Statement

We use the symmetric shifted convention defined below; the
name "double Pareto" is overloaded in the literature and can also refer to
positive-valued size distributions with different left and right exponents.
Let $Y$ have a [Pareto Type I distribution](./incerto-pareto.md) with lower cutoff $1$ and
tail exponent $\alpha>0$.  Let $B$ be independent of $Y$ with
$\mathbb P(B=1)=\mathbb P(B=-1)=1/2$.  The shifted symmetric double Pareto
variable is

$$
X = B(Y-1).
$$

Its density on the real line is

$$
f(x;\alpha)=\frac{\alpha}{2}(1+|x|)^{-(\alpha+1)},\qquad x\in\mathbb R.
$$

The CDF is

$$
F(x)=
\begin{cases}
\frac12(1-x)^{-\alpha}, & x<0,\\
1-\frac12(1+x)^{-\alpha}, & x\ge 0.
\end{cases}
$$

For $x\ge0$, each one-sided tail has probability

$$
\mathbb P(X>x)=\frac12(1+x)^{-\alpha},
\qquad
\mathbb P(X<-x)=\frac12(1+x)^{-\alpha}.
$$

Consequently,

$$
\mathbb P(|X|>x)=(1+x)^{-\alpha}.
$$

More generally, a location-scale version $Z=\mu+sX$ with $s>0$ has density

$$
f_Z(z)=\frac{\alpha}{2s}
\left(1+\frac{|z-\mu|}{s}\right)^{-(\alpha+1)}.
$$

The ordinary mean exists and equals $0$ exactly when $\alpha>1$.  For
$\alpha\le1$, the positive and negative parts are both infinite, so symmetry
gives a center but not an ordinary expectation.  The variance is finite exactly
when $\alpha>2$, in which case

$$
\operatorname{Var}(X)=\frac{2}{(\alpha-1)(\alpha-2)}.
$$

The absolute moments unify these thresholds:

$$
\mathbb E|X|^p
=\frac{\Gamma(p+1)\Gamma(\alpha-p)}{\Gamma(\alpha)},
\qquad 0<p<\alpha.
$$

Here $F$ and $f$ denote the CDF and density; $B$ is an independent sign.
$\Gamma$ denotes the gamma function. Absolute moments diverge for
$p\ge\alpha$; the displayed gamma expression only applies for $0<p<\alpha$.

## Shape and two-sided tail intuition

The ordinary [Pareto distribution](./incerto-pareto.md) is one-sided: the rare extreme is always on
the right.  The double Pareto keeps the same power-law magnitude but gives the
shock a sign.  This makes it useful for toy return models where both large gains
and large losses are possible, while preserving a transparent tail exponent.

(double-pareto-density-plot)=
The density is symmetric, with peak $f(0)=\alpha/2$ and

$$
\frac{f(x)}{f(0)}=(1+|x|)^{-(\alpha+1)}.
$$

Smaller $\alpha$ lowers the central density and puts more probability far
from zero on both sides. The chance of a large positive value and the chance
of an equally large negative value decay at the same power rate.

## Derivation from signed Pareto magnitude

We derive the symmetric shifted construction used here from the signed
Pareto magnitude $X=B(Y-1)$.  The tail, density, and CDF formulas follow
directly from symmetry.  The moment thresholds use the same Pareto moment
logic as [Pareto Moment Existence](./incerto-pareto-moment-existence.md).

For $x\ge0$,

$$
\mathbb P(X>x)
=\mathbb P(B=1, Y-1>x)
=\frac12\mathbb P(Y>1+x)
=\frac12(1+x)^{-\alpha}.
$$

The negative tail is identical by symmetry.  Differentiating the CDF on either
side gives the density.  The expectation is zero only when the first absolute
moment exists.  When $\alpha\le1$,
$\mathbb E[X^+]=\mathbb E[X^-]=\infty$, so the ordinary mean is undefined even
though the distribution is symmetric.  Since $X^2=(Y-1)^2$,

$$
\mathbb E[X^2]
=\mathbb E[Y^2]-2\mathbb E[Y]+1
=\frac{\alpha}{\alpha-2}-2\frac{\alpha}{\alpha-1}+1
=\frac{2}{(\alpha-1)(\alpha-2)}
$$

for $\alpha>2$. For $1<\alpha\le2$, the mean exists but the variance
is infinite. For $\alpha\le1$, the ordinary mean, and hence variance
about that mean, is undefined; the second raw moment is still infinite.

The library helper `xmath.incerto.distributions.double_pareto` reports the
mean as `NaN` and the variance as $+\infty$ for $0<\alpha\le1$.
The variance result is a reporting convention reflecting the infinite raw
second moment; it does not define variance about an undefined mean.
For $1<\alpha\le2$ it reports mean $0$ and variance $+\infty$;
for $\alpha>2$ it reports mean $0$ and the finite variance above.

More generally, integration of the absolute-value density gives the beta
integral

$$
\mathbb E|X|^p=\alpha\int_0^\infty t^p(1+t)^{-\alpha-1}\,dt
=\frac{\Gamma(p+1)\Gamma(\alpha-p)}{\Gamma(\alpha)},
\qquad 0<p<\alpha.
$$

## Two-sided tail calculations

(double-pareto-tail-simulation)=
The exact two-sided survival $(1+x)^{-\alpha}$ is regularly varying with
index $-\alpha$. At $\alpha=1.5$, its values at $x=0,3,8$ are respectively
$1$, $1/8$, and $1/27$. Each one-sided tail has half of that probability.

For a fixed multiplier $t>0$,

$$
\frac{\mathbb P(|X|>tx)}{\mathbb P(|X|>x)}
=\left(\frac{1+tx}{1+x}\right)^{-\alpha}\to t^{-\alpha}.
$$

The shift means this is a limiting power ratio, rather than the exact
threshold scaling of an unshifted Pareto law.

(double-pareto-python-check)=
At $\alpha=3$, direct evaluation gives the following exact values.

| $x$ | Density $f(x)$ | CDF $F(x)$ |
| --- | --- | --- |
| $-4$ | $3/1250$ | $1/250$ |
| $0$ | $3/2$ | $1/2$ |
| $4$ | $3/1250$ | $249/250$ |

The mean is zero and the variance is $2/[(3-1)(3-2)]=1$. These values are
formula evaluations, without simulation or an imported distribution package.
For [max-to-sum diagnostics](./incerto-max-to-sum-ratio.md), use $|X_i|$
or another explicitly defined nonnegative quantity; signed sums can cancel.

(double-pareto-student-matching)=
## Matching a Student tail or a variance

SCoFT v4 §5.3, p. 101 [@taleb2025scoftv4], compares a symmetric shifted
power law with a Student law. We keep the normalized density $\alpha/2$
above on each side and derive both matching scales. Equal exponents do
not imply equal moments or equal leading tail constants.

Let $T_\nu$ have the unit-scale Student density, with $\nu=\alpha>0$,
and let $Z=sT_\alpha$, $s>0$. Its right tail satisfies

$$
P(Z>x)\sim C_\alpha s^\alpha x^{-\alpha},\qquad
C_\alpha=\frac{\Gamma((\alpha+1)/2)\alpha^{\alpha/2-1}}
                  {\sqrt\pi\,\Gamma(\alpha/2)}.
$$

We prove the constant from the density. Factoring $x^2/(\alpha s^2)$
out of $1+x^2/(\alpha s^2)$ in the Student density gives
$f_Z(x)\sim\alpha C_\alpha s^\alpha x^{-\alpha-1}$.
For each $\varepsilon>0$, this density lies between $1\pm\varepsilon$
times its leading power beyond a fixed threshold. Integrating those bounds
from $x$ to infinity and letting $\varepsilon\downarrow0$ gives the
displayed survival asymptotic. For positive orders $p$, the same comparison
proves $\mathbb E|Z|^p<\infty$ exactly when $p<\alpha$; the density is
bounded near zero.

The double-Pareto right-tail constant is $1/2$ at scale one. Matching it
requires

$$
s_{\rm tail}=(2C_\alpha)^{-1/\alpha}.
$$

For $\alpha>2$, the Student variance is
$s^2\alpha/(\alpha-2)$ (the beta-integral moment calculation is also
given in the [mixture note](incerto-mixture-uncertainty.md#mixture-variance-precision)).
Equating that to $2/[(\alpha-1)(\alpha-2)]$ instead gives

$$
s_{\rm var}=\sqrt{\frac{2}{\alpha(\alpha-1)}}.
$$

At $\alpha=3$, $C_3=2\sqrt3/\pi$. Variance matching uses
$s_{\rm var}=1/\sqrt3$ and gives Student right-tail constant
$2/(3\pi)$, versus $1/2$ for double Pareto. Their asymptotic survival
ratio is $4/(3\pi)\approx0.424413$ even though both variances are one.
Tail matching uses $s_{\rm tail}=(\pi/(4\sqrt3))^{1/3}$ and gives
Student variance about $1.770683$, versus one for double Pareto.
Matching tails thus leaves a substantial variance difference in this example.

```{figure} ../assets/figures/incerto/power-law-matching.svg
:alt: Three vertical panels compare the density, survival ratio and threshold multiplier ratio of a double Pareto law and two scaled Student laws, with legends below the axes so the central spike is visible.

Exact model curves at exponent $3$. Top: densities near the center, including the double-Pareto spike at zero;
the legend is below the axes so it covers no density curve.
Middle: Student survival divided by the double-Pareto survival, with
the two asymptotic targets marked. Bottom: multiplier ratios at multiplier
$2$, approaching $1/8$. Matching variance and matching a tail constant
choose different Student scales; neither makes the distributions identical.
```

Run `uv run python demos/incerto/power_laws.py`. The figure uses 501 density
points on $[-4,4]$ and 241 thresholds from $1$ to $10^4$ for each tail
panel. All values are dimensionless, exact-model evaluations. This comparison
does not change the helper's retained `NaN`/`+inf` reporting convention
when $\alpha\le1$, or define variance about an undefined mean.

## Caveats

- We describe a symmetric signed construction whose magnitude $|X|$ has
  a Lomax law, not every distribution called
  "double Pareto" in the literature.
- For $\alpha\le1$, even the absolute first moment is infinite.  Symmetry can
  make a formal location or principal-value calculation look harmless while the
  ordinary expectation is undefined and absolute exposure remains uncontrolled.
- The distribution is a pedagogical model.  Empirical returns usually need
  skew, truncation, volatility clustering, dependence, or threshold modeling
  before a tail fit is credible.

## References

- @taleb2025scoftv4, §5.3, p. 101: comparison context. The normalization,
  tail constants and matching formulas above are derived for our explicit convention.
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).

## Backlinks

- Depends on: [Pareto Distribution](./incerto-pareto.md) and
  [Regular Variation](./incerto-regular-variation.md).
- Used by: [Chapter 5](incerto-scoft-ch5.md) and its Student comparison.

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/distributions/double-pareto.md`, revision `9717c9c`
(2026-09-13 Batch 2 import). Copyright (c) 2023 xshi19. Licensed under MIT.
Links, notation, and qualifications were adapted for this site; executable
figures and simulations were replaced with static calculations. No upstream
execution or formal-proof verification is claimed for this adaptation.

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
