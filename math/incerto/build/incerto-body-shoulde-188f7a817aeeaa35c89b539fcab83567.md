---
title: Body, Shoulders, and Tails
options:
  concept:
    id: body-shoulder-tail
    type: method
    related:
      - mixture-uncertainty
      - moment-estimation
    tags:
      - diagnostics
      - tail-risk
---

## Statement

For a centered normal density parameterized by variance $v>0$

$$
g_v(x)=
\frac{1}{\sqrt{2\pi v}}
\exp\left(-\frac{x^2}{2v}\right),
$$

the body-shoulder-tail diagnostic asks where a small, mean-preserving
randomization of $v$ adds density and where it removes density.  A local
version is the sign of

$$
\frac{\partial^2 g_v(x)}{\partial v^2}.
$$

For the standard normal density, this second derivative changes sign at

$$
\pm\sqrt{3-\sqrt6},
\qquad
\pm\sqrt{3+\sqrt6}.
$$

These four points split the line into left tail, left shoulder, peak, right
shoulder, and right tail regions.  Positive variance curvature means a local
variance mixture raises density there; negative variance curvature means it
lowers density there.

The notation $g_v$ denotes the density at fixed variance $v$; a mixture
averages these densities over a positive random variance $Y$. The
[Normix normal-mixtures note](https://xshi19.github.io/math/normix-theory/normix-normal-mixtures/) supplies the
complementary conditional construction. See the [shared notation](https://xshi19.github.io/math/notation/) canon.

## Examples

- For the standard normal density under variance perturbation, the inner
  shoulder boundaries are about $\pm0.742$, and the outer tail boundaries are
  about $\pm2.334$.
- If one perturbs the normal scale $\sigma$ rather than the variance $v$, the
  corresponding boundaries are different: about $\pm0.662$ and $\pm2.136$ for
  the standard normal.
- For Student-t densities, a scale-curvature version applies, but the boundary
  values depend on the degrees of freedom.
- These boundaries are not Pareto thresholds; threshold selection for
  [regular variation](./incerto-regular-variation.md) is a separate problem.

## Diagnostic intuition

If a Gaussian variance is randomized while keeping the center fixed, the
resulting mixture does not simply "spread out" everywhere.  It tends to add mass near the
peak and in the far tails, while taking mass from the shoulders.  The shoulders
are the moderate-deviation region that looks ordinary under a single scale but
is depleted when variance uncertainty is introduced.

This diagnoses the geometry of a specified population density, not its tail index.
It explains why stochastic volatility can create both a sharper center and
fatter-looking tails without producing a [Pareto tail](./incerto-pareto.md).
Choosing a threshold for tail estimation remains the separate judgment handled
by [Tail Threshold Selection](./incerto-tail-threshold-selection.md).

## Derivation of the variance-curvature boundaries

We derive the normal variance-curvature boundaries and the pointwise local
mixture interpretation.

:::{prf:proposition} Normal variance curvature and bounded perturbations
:label: prop-body-shoulder-tail-variance-curvature

For $v>0$ and $x\in\mathbb R$, let

$$
g_v(x)=
\frac{1}{\sqrt{2\pi v}}
\exp\left(-\frac{x^2}{2v}\right).
$$

Then

$$
\frac{\partial^2 g_v(x)}{\partial v^2}
=\frac{g_v(x)}{4v^2}\left(\frac{x^4}{v^2}-\frac{6x^2}{v}+3\right).
$$

The four sign-change boundaries are $x=\pm\sqrt{v(3-\sqrt6)}$ and
$x=\pm\sqrt{v(3+\sqrt6)}$. Curvature is positive inside the inner pair
and beyond the outer pair, and negative between them.

Fix $v_0>0$ and a bounded random variable $H$ with $\mathbb EH=0$ and
$\mathbb EH^2=1$. Put $Y_\tau=v_0+\tau H$, taking $\tau>0$ small
enough that $Y_\tau>0$ almost surely. For each fixed $x$, as $\tau\downarrow0$,

$$
\mathbb E[g_{Y_\tau}(x)]-g_{v_0}(x)
=\frac{\tau^2}{2}
\frac{\partial^2g_v(x)}{\partial v^2}\bigg|_{v=v_0}
+O(\tau^3).
$$

Away from the curvature zeros, this difference has the curvature's sign for
all sufficiently small $\tau>0$.
:::

:::{prf:proof}
Differentiating at fixed $x$ gives

$$
\frac{\partial g_v(x)}{\partial v}
=
\frac{g_v(x)}{2v}\left(\frac{x^2}{v}-1\right),
$$

and a second differentiation gives

$$
\frac{\partial^2 g_v(x)}{\partial v^2}
=
\frac{g_v(x)}{4v^2}\left(\frac{x^4}{v^2}-\frac{6x^2}{v}+3\right).
$$

At $v=1$, the sign changes where

$$
x^4-6x^2+3=0.
$$

Solving the quadratic in $x^2$ gives

$$
x^2=3\pm\sqrt6.
$$

For general $v$, replace $x^2$ by $x^2/v$. The quadratic has positive
leading coefficient, giving the stated signs. Boundedness of $H$ keeps
$Y_\tau$ in a compact interval around $v_0$ for small $\tau$. At each fixed
$x$, the third variance derivative is bounded there. Taylor's theorem and
$\mathbb EH=0$, $\mathbb EH^2=1$ therefore give

$$
\mathbb E[g_{Y_\tau}(x)]-g_{v_0}(x)
=\frac{\tau^2}{2}
\frac{\partial^2g_v(x)}{\partial v^2}\bigg|_{v=v_0}
+O(\tau^3).
$$

Thus the curvature sign determines the leading local density change away
from its zeros.
:::

**Remarks.** Small variance of a perturbation alone would not justify
this remainder without control of the perturbation family. The expansion
is pointwise in $x$, not uniform arbitrarily far into the tails.
The scale calculation below uses a different derivative; none of these
boundaries is a rule for selecting a Pareto threshold.

## Static region and mixture calculations

(body-shoulder-tail-region-plot)=
At $v=1$, put $a=\sqrt{3-\sqrt6}\approx0.742$ and
$b=\sqrt{3+\sqrt6}\approx2.334$.

| Region | Variance curvature of density | Leading change under small mean-preserving variance mixing |
| --- | --- | --- |
| $\lvert x\rvert<a$ (peak) | Positive | Density increases |
| $a<\lvert x\rvert<b$ (shoulders) | Negative | Density decreases |
| $\lvert x\rvert>b$ (tails) | Positive | Density increases |

For a concrete finite mixture, define

$$
f_{\mathrm{mix}}(x)=\tfrac12 g_{0.2}(x)+\tfrac12 g_{1.8}(x).
$$

Its mean is zero and its variance is one. The ratio to the unit normal is

$$
\frac{f_{\mathrm{mix}}(x)}{g_1(x)}
=\frac{\sqrt5}{2}e^{-2x^2}
+\frac{\sqrt5}{6}e^{2x^2/9}.
$$

At $x=0,1,3$, the ratio is approximately $1.491$, $0.617$, and $2.754$.
This exhibits increased center and far-tail density with depleted shoulders.
The exact crossings solve the ratio-equals-one equation; they need not be
the local curvature boundaries $a,b$. This finite Gaussian mixture has a
finite moment generating function at every real argument and is not
regularly varying, despite having more far-tail mass than the unit normal.

### Density, survival, and scale are different diagnostics

We differentiate survival and density separately to compare their boundaries.

:::{prf:proposition} Normal survival and scale curvature
:label: prop-body-shoulder-tail-survival-scale-curvature

For $x>0$, the two-sided normal survival at variance $v$ is
$T_v(x)=2\bar\Phi(x/\sqrt v)$, where $\bar\Phi$ is standard normal survival
and $\phi$ is its density. For $v>0$,

$$
\frac{\partial^2 T_v(x)}{\partial v^2}
=\frac{x\phi(x/\sqrt v)}{2v^{5/2}}\left(\frac{x^2}{v}-3\right).
$$

Its curvature is negative for $0<x<\sqrt{3v}$ and positive for
$x>\sqrt{3v}$.

For $x\in\mathbb R$ and $\sigma>0$, let
$g_{\sigma^2}(x)=(\sqrt{2\pi}\sigma)^{-1}e^{-x^2/(2\sigma^2)}$.
Then

$$
\frac{\partial^2}{\partial\sigma^2}g_{\sigma^2}(x)
=\frac{g_{\sigma^2}(x)}{\sigma^2}
\left(\frac{x^4}{\sigma^4}-\frac{5x^2}{\sigma^2}+2\right).
$$

Its positive scale-curvature zeros are
$\sigma\sqrt{(5-\sqrt{17})/2}$ and
$\sigma\sqrt{(5+\sqrt{17})/2}$. Curvature is positive inside the inner
pair of symmetric zeros and beyond the outer pair, and negative between them.
:::

:::{prf:proof}
Differentiating survival once gives
$\partial_vT_v(x)=xv^{-3/2}\phi(x/\sqrt v)$. Differentiating again gives
the displayed formula, whose sign is that of $x^2/v-3$.
For the density,
$\partial_\sigma g_{\sigma^2}(x)=g_{\sigma^2}(x)(x^2/\sigma^2-1)/\sigma$.
A second differentiation gives the displayed scale derivative. Its sign is
that of the quadratic $t^2-5t+2$, with $t=x^2/\sigma^2$; solving that
quadratic gives the zeros and signs.
:::

**Remarks.** The survival boundary is about $1.732$ when $v=1$, differing
from both variance-curvature density boundaries. At $\sigma=1$, the
positive scale-curvature zeros are
$\sqrt{(5-\sqrt{17})/2}\approx0.662$ and
$\sqrt{(5+\sqrt{17})/2}\approx2.136$. Mean-preserving scale mixing is a
different operation from mean-preserving variance mixing. Student scale-curvature comparisons depend on degrees of freedom, as derived below.

(body-shoulder-student)=
## Student scale curvature

The scale calculation also has a closed form for a Student density.

:::{prf:proposition} Student scale-curvature boundaries
:label: prop-body-shoulder-tail-student-curvature

Let
$h_s(x)=C_\nu s^{-1}(1+x^2/(\nu s^2))^{-(\nu+1)/2}$, where $s>0$ is
scale, $\nu>1$ is degrees of freedom, $x\in\mathbb R$, and the positive
constant $C_\nu$ normalizes the density.
Writing $t=x^2/(\nu s^2)$, the curvature is

$$
\frac{s^2\partial_s^2h_s}{h_s}
=\frac{\nu(\nu-1)t^2-(5\nu+1)t+2}{(1+t)^2}.
$$

Its four sign-change boundaries are

$$
x=\pm s\sqrt{
\frac{5\nu+1\pm\sqrt{(\nu+1)(17\nu+1)}}{2(\nu-1)}}.
$$

Curvature is positive inside the inner pair and beyond the outer pair, and
negative between them. At fixed $s$, the positive boundaries tend to
$s\sqrt{(5-\sqrt{17})/2}$ and $s\sqrt{(5+\sqrt{17})/2}$ as
$\nu\to\infty$.
:::

:::{prf:proof}
Logarithmic differentiation gives

$$
s\,\partial_s\log h_s=\frac{\nu t-1}{1+t},\qquad
\frac{s^2\partial_s^2h_s}{h_s}
=\frac{\nu(\nu-1)t^2-(5\nu+1)t+2}{(1+t)^2}.
$$

For the second formula, differentiate the first, using
$\partial_st=-2t/s$, and add the square of the log derivative.
The numerator is a quadratic with two positive roots when $\nu>1$.
Solving for $t$ and using $x=\pm s\sqrt{\nu t}$ gives the four
density-curvature boundaries

$$
x=\pm s\sqrt{
\frac{5\nu+1\pm\sqrt{(\nu+1)(17\nu+1)}}{2(\nu-1)}}.
$$

The leading coefficient and the value at zero are positive, so curvature
has the stated signs. Dividing numerator and denominator inside the root
formula by $\nu$ gives the two Gaussian scale-curvature limits.
:::

**Remarks.** At $\nu=3,s=1$, the positive boundaries are
$\sqrt{4-\sqrt{13}}\approx0.628$ and
$\sqrt{4+\sqrt{13}}\approx2.758$. The standard deviation is $\sqrt3$;
these numbers are in **scale units**. For $\nu\le2$, the variance is not
finite, although the four-boundary formula still applies when $1<\nu\le2$.
At $\nu=1$ the quadratic loses its leading term, so that formula's domain
must not be extended through the zero denominator.

The Gaussian limit is algebraic, not a fitted threshold rule.
The [gamma-precision construction](incerto-mixture-uncertainty.md#mixture-variance-precision)
explains the Student tail exponent and moment conditions separately.

(body-shoulder-worked-figure)=
## Curvature and exact density differences

```{figure} ../assets/figures/incerto/finite-moment-curvature.svg
:alt: Two stacked plots compare the Gaussian variance curvature divided by two with exact symmetric mixture density differences divided by squared perturbation size, and show different variance, scale and Student scale boundaries.

The upper panel compares the pointwise quadratic approximation with exact
mixtures at perturbations 0.1 and 0.6. Vertical lines mark the variance
curvature zeros; finite-mixture crossings need not stay on those lines.
The lower panel places normal variance, normal scale, and Student-three
scale boundaries on the same dimensionless axis. Student scale is 1,
so its standard deviation is the square root of 3.
```

Use [`normal_peak_shoulder_tail`](incerto-api-reference.md#api-normal-peak-shoulder-tail), [`normal_variance_peak_shoulder_tail`](incerto-api-reference.md#api-normal-variance-peak-shoulder-tail), [`t_peak_shoulder_tail`](incerto-api-reference.md#api-t-peak-shoulder-tail).

```python
from xmath.incerto.stats import (
    normal_peak_shoulder_tail, normal_variance_peak_shoulder_tail,
    t_peak_shoulder_tail,
)

for label, values in (("Normal, scale", normal_peak_shoulder_tail()),
                      ("Normal, variance", normal_variance_peak_shoulder_tail()),
                      ("Student 3, scale", t_peak_shoulder_tail(3))):
    print(f"{label}: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
Normal, scale: [-2.136, -0.6622, 0.6622, 2.136]
Normal, variance: [-2.334, -0.742, 0.742, 2.334]
Student 3, scale: [-2.758, -0.6281, 0.6281, 2.758]
```

These are local perturbation boundaries at unit scale, not fitted tail thresholds.

Figure regeneration: `uv run python demos/incerto/finite_moments.py`. These are exact density
calculations on 801 fixed points, with no random sampling. The script
reports the finite-mixture crossings separately from the local boundaries.

Jensen's inequality gives a survival comparison when curvature has one sign
throughout the mixing interval.

:::{prf:corollary} Survival comparisons under restricted variance mixing
:label: cor-body-shoulder-tail-survival-jensen

Fix $x>0$ and let $Y>0$ almost surely with $\mathbb EY<\infty$.
Put $T_v(x)=2\bar\Phi(x/\sqrt v)$ for $v>0$, where $\bar\Phi$ is
standard normal survival.
If $Y\le v_{\max}<\infty$ almost surely and $x^2\ge3v_{\max}$, then
$\mathbb ET_Y(x)\ge T_{\mathbb EY}(x)$.
If instead $Y\ge v_{\min}>0$ almost surely and $x^2\le3v_{\min}$, then
$\mathbb ET_Y(x)\le T_{\mathbb EY}(x)$.
:::

:::{prf:proof}
By [](#prop-body-shoulder-tail-survival-scale-curvature), $v\mapsto T_v(x)$
is convex on $(0,v_{\max}]$ in the first case and concave on
$[v_{\min},\infty)$ in the second. Its supporting-line inequality at
$\mathbb EY$ gives the respective Jensen inequality after taking
expectations. Survival is bounded, and the mixing mean is finite.
:::

**Remark.** These sufficient conditions show why a
local tail-curvature statement cannot justify global Jensen claims at
every threshold under unrestricted mixing.

## Caveats

- The diagnostic uses centered normal components and a small mean-preserving
  perturbation of their positive variance.  Skewed or
  multimodal distributions need a different interpretation.
- The region labels are not universal definitions of "body" or "tail".  They
  are tied here to local variance perturbations.  Perturbing scale instead of
  variance changes the numerical boundaries.
- A shoulder/tail boundary is not a threshold for Pareto estimation.  It
  describes density geometry, not regular variation.
- For a large finite mixture, exact density crossings must be calculated
  from that mixture; the curvature formulas describe a local limit.

## References

- @taleb2025scoftv4, *Statistical Consequences of Fat Tails*, third edition,
  2025, ch. 4, §§4.3–4.3.1, printed pp. 75–78. The source uses scale
  curvature; our variance and survival comparisons are derived separately.
  The calculations above prove the stated normal and Student cases, not a
  universal crossover theorem for all unimodal densities.

## Backlinks

- Depends on: [Normix normal mixtures](https://xshi19.github.io/math/normix-theory/normix-normal-mixtures/)
  and the canonical density notation in [Shared notation](https://xshi19.github.io/math/notation/).
- Used by: [Dispersion Ratio Under Fat Tails](./incerto-dispersion-ratio.md).
