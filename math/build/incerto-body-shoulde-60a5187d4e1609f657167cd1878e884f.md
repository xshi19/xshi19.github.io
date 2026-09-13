---
title: Body, Shoulders, and Tails
options:
  concept:
    id: body-shoulder-tail
    type: method
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
averages these densities over a positive random variance $V$. The
[Normix normal-mixtures note](./normix-normal-mixtures.md) supplies the
complementary conditional construction. A shared notation page is planned.

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

This is a diagnostic for finite-sample geometry, not a tail-index estimator.
It explains why stochastic volatility can create both a sharper center and
fatter-looking tails without producing a [Pareto tail](./incerto-pareto.md).
Choosing a threshold for tail estimation remains the separate judgment handled
by [Tail Threshold Selection](./incerto-tail-threshold-selection.md).

## Derivation of the variance-curvature boundaries

We derive the standard-normal variance-curvature boundaries by differentiating
the density with respect to variance and solving the resulting quadratic in
$x^2$.  The Taylor expansion explains the local mixture interpretation.  The
comparison with scale perturbation below uses a different derivative; none
of these boundaries is a rule for selecting a Pareto threshold.

For the normal density with variance parameter $v$, write

$$
g_v(x)=
\frac{1}{\sqrt{2\pi v}}
\exp\left(-\frac{x^2}{2v}\right).
$$

Set $y=x/\sqrt v$.  A direct differentiation gives

$$
\frac{\partial g_v(x)}{\partial v}
=
\frac{g_v(x)}{2v}(y^2-1),
$$

and a second differentiation gives

$$
\frac{\partial^2 g_v(x)}{\partial v^2}
=
\frac{g_v(x)}{4v^2}(y^4-6y^2+3).
$$

At $v=1$, the sign changes where

$$
x^4-6x^2+3=0.
$$

Solving the quadratic in $x^2$ gives

$$
x^2=3\pm\sqrt6.
$$

This is a local Taylor diagnostic. A sufficient small-perturbation model is

$$
V_\tau=v_0+\tau Z,\qquad v_0>0,\quad
\mathbb E[Z]=0,\quad \mathbb E[Z^2]=1,
$$

with bounded $Z$ and $\tau$ small enough that $V_\tau>0$. For each fixed $x$,
Taylor's theorem, with bounded third derivative near $v_0$, gives

$$
\mathbb E[g_{V_\tau}(x)]-g_{v_0}(x)
=\frac{\tau^2}{2}
\frac{\partial^2g_v(x)}{\partial v^2}\bigg|_{v=v_0}
+O(\tau^3).
$$

Thus the curvature sign determines the leading local density change away
from its zeros. Small variance of a perturbation alone would not justify
this remainder without control of the perturbation family. The expansion
is pointwise in $x$, not uniform arbitrarily far into the tails.

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

For $x>0$, the two-sided normal survival at variance $v$ is
$T_v(x)=2\bar\Phi(x/\sqrt v)$, where $\bar\Phi$ is standard normal survival
and $\phi$ is its density. Differentiation gives

$$
\frac{\partial^2 T_v(x)}{\partial v^2}
=\frac{x\phi(x/\sqrt v)}{2v^{5/2}}\left(\frac{x^2}{v}-3\right).
$$

Its positive-threshold curvature boundary is $x=\sqrt{3v}$, about $1.732$
when $v=1$. It differs from both density boundaries.

For scale perturbation, write $h_\sigma(x)=g_{\sigma^2}(x)$ and
$z=x/\sigma$. Then

$$
\frac{\partial^2h_\sigma(x)}{\partial\sigma^2}
=\frac{h_\sigma(x)}{\sigma^2}(z^4-5z^2+2).
$$

At $\sigma=1$, its positive zeros are
$\sqrt{(5-\sqrt{17})/2}\approx0.662$ and
$\sqrt{(5+\sqrt{17})/2}\approx2.136$. Mean-preserving scale mixing is a
different operation from mean-preserving variance mixing. Student-t
scale-curvature comparisons depend on degrees of freedom; a separate
worked comparison is planned.

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

- Taleb, *Statistical Consequences of Fat Tails* [2020](https://arxiv.org/abs/2001.10488).

## Backlinks

- Depends on: [Normix normal mixtures](./normix-normal-mixtures.md)
  and the canonical density notation in Notation (planned).
- Used by: Dispersion Ratio Under Fat Tails (planned).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/methods/body-shoulder-tail.md`, revision `9717c9c`
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
