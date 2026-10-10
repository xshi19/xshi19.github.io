---
title: Karamata's Theorem
options:
  concept:
    id: karamata
    type: theorem
    depends_on:
      - regular-variation
    tags:
      - fat-tails
      - moments
---

## Statement

Let $L$ be eventually positive, measurable, locally bounded, and
[slowly varying](./incerto-regular-variation.md) at infinity.  If $\rho>-1$, then

$$
\int_a^x t^\rho L(t)\,dt
\sim
\frac{x^{\rho+1}L(x)}{\rho+1},
\qquad x\to\infty.
$$

If $\rho<-1$, then

$$
\int_x^\infty t^\rho L(t)\,dt
\sim
\frac{x^{\rho+1}L(x)}{-\rho-1},
\qquad x\to\infty.
$$

Here $a>0$ is fixed, and $A(x)\sim B(x)$ means $A(x)/B(x)\to1$.

For a nonnegative random variable $X$ with
[regularly varying](./incerto-regular-variation.md) survival function
$\bar F(x)=x^{-\alpha}L(x)$ and $\alpha>0$, integration by parts gives the
moment-test consequences

$$
\mathbb E[X^p\mathbf 1_{\{X>x\}}]
\sim
\frac{\alpha}{\alpha-p}x^p\bar F(x),
\qquad 0<p<\alpha,
$$

and

$$
\mathbb E[X^p\mathbf 1_{\{X\le x\}}]
\sim
\frac{\alpha}{p-\alpha}x^p\bar F(x),
\qquad p>\alpha.
$$

Thus a [Pareto-type tail](./incerto-pareto.md) has finite $p$th moment
for $p<\alpha$ and infinite $p$th moment for $p>\alpha$.  The boundary case
$p=\alpha$ is not decided by this theorem alone; it depends on the slowly
varying factor $L$.

Here $\bar F(x)=\mathbb P(X>x)$, $\alpha$ is the positive tail exponent,
and $p$ is the moment order. The integration parameters $a$ and $\rho$
belong to the integral theorem.

## Intuition

Karamata's theorem says that, for regularly varying functions, integrals are
asymptotically governed by the endpoint where the mass accumulates.  When
$\rho>-1$, the integral up to $x$ is controlled by the upper endpoint.  When
$\rho<-1$, the remaining tail integral beyond $x$ is controlled by the lower
endpoint.

This is the bridge between tail shape and
[moment existence](./incerto-pareto-moment-existence.md).  Once the survival tail behaves
like $x^{-\alpha}L(x)$, multiplying by $x^p$ tests whether the $p$th moment is
still dominated by ordinary observations or by the far tail.

## Examples

- For an exact [Pareto tail](./incerto-pareto.md), $L$ is constant, so
  Karamata reduces the moment test to integrating a pure power.
- For $L(x)=\log x$, the same power boundary applies, but the endpoint
  asymptotic gains the slowly varying logarithmic factor.
- The boundary case $p=\alpha$ needs separate analysis.  Exact Pareto tails
  diverge logarithmically, as shown in
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md).  More generally,

  $$
  \mathbb E[X^\alpha]<\infty
  \Longleftrightarrow
  \int^\infty \frac{L(t)}{t}\,dt<\infty,
  $$

  when $\bar F(t)=t^{-\alpha}L(t)$ in the tail.  Thus $L(t)=1$ gives
  logarithmic divergence, $L(t)=1/\log t$ gives $\log\log t$ divergence,
  and $L(t)=1/(\log t)^2$ gives a finite boundary moment.

## Deriving the moment consequences

We use Karamata's integral asymptotics as a cited theorem, then derive the two
moment consequences by integration by parts.  The boundary case $p=\alpha$ is
not decided by Karamata alone; it depends on the slowly varying factor and is
handled through examples and caveats.

The integral asymptotics are the classical form of Karamata's theorem for
regularly varying functions.  The full proof uses uniform convergence of slowly
varying functions on compact multiplier intervals and a split of the integral
into a near-endpoint part and a negligible remainder; this page cites the
standard theorem rather than reproducing that argument here.

The moment consequences follow from integration by parts.  For $0<p<\alpha$,

$$
\mathbb E[X^p\mathbf 1_{\{X>x\}}]
=x^p\bar F(x)+p\int_x^\infty t^{p-1}\bar F(t)\,dt.
$$

Since $t^{p-1}\bar F(t)=t^{p-\alpha-1}L(t)$ and
$p-\alpha-1<-1$, Karamata's tail-integral form gives

$$
\int_x^\infty t^{p-1}\bar F(t)\,dt
\sim
\frac{x^{p-\alpha}L(x)}{\alpha-p}
=
\frac{x^p\bar F(x)}{\alpha-p}.
$$

Therefore

$$
\mathbb E[X^p\mathbf 1_{\{X>x\}}]
\sim
\left(1+\frac{p}{\alpha-p}\right)x^p\bar F(x)
=
\frac{\alpha}{\alpha-p}x^p\bar F(x).
$$

For $p>\alpha$,

$$
\mathbb E[X^p\mathbf 1_{\{X\le x\}}]
=p\int_0^x t^{p-1}\bar F(t)\,dt-x^p\bar F(x).
$$

The part of the integral over any fixed bounded interval is negligible relative
to $x^p\bar F(x)$.  Applying Karamata to
$t^{p-\alpha-1}L(t)$ gives

$$
p\int_0^x t^{p-1}\bar F(t)\,dt
\sim
\frac{p}{p-\alpha}x^p\bar F(x),
$$

so subtracting $x^p\bar F(x)$ leaves

$$
\mathbb E[X^p\mathbf 1_{\{X\le x\}}]
\sim
\frac{\alpha}{p-\alpha}x^p\bar F(x).
$$

For the exact Pareto special case, $L$ is constant and the same formulas are
obtained by direct power integration.

## Moment-test calculation

(karamata-moment-test-computation)=
For a slowly varying example take $L(t)=\log t$ on $t>1$ and
$\rho>-1$. Integration by parts gives the exact identity

$$
\int_1^x t^\rho\log t\,dt
=\frac{x^{\rho+1}\log x}{\rho+1}
-\frac{x^{\rho+1}-1}{(\rho+1)^2}.
$$

Dividing by Karamata's leading term yields

$$
\frac{\int_1^x t^\rho\log t\,dt}
     {x^{\rho+1}\log x/(\rho+1)}
=1-\frac{1-x^{-(\rho+1)}}{(\rho+1)\log x}\longrightarrow1.
$$

For exact Pareto with $x_m=1$ and $p>\alpha$, the ratio of the truncated
moment to its leading term is $1-x^{-(p-\alpha)}\to1$.
At $p=\alpha$, the truncated moment is $\alpha\log x$ instead.
These are exact calculations illustrating the cited theorem.

## Caveats

- The assumptions on $L$ matter.  Slow variation is an asymptotic regularity
  condition, not a finite-sample diagnostic.
- Karamata does not settle the boundary moment $p=\alpha$.  Exact Pareto tails
  diverge logarithmically there, while other slowly varying factors can change
  the boundary behavior.
- Moment existence is a tail statement.  Estimating a moment from data also
  depends on sample size, dependence, threshold choice, and whether the fitted
  tail model is credible.
- We treat nonnegative right tails.  Two-sided models need the same
  reasoning applied to $|X|$ or to each side separately.

## References

- Bingham, Goldie, and Teugels, *Regular Variation* [1987](https://doi.org/10.1017/CBO9780511721434).
- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  (1971, 2nd ed., Wiley).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).

## Backlinks

- Depends on: [Regular Variation](./incerto-regular-variation.md) and the canonical tail
  notation in Notation (planned).
- Used by: [Pareto Distribution](./incerto-pareto.md),
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md),
  [LLN Failure Under Infinite Mean](./incerto-lln-failure.md),
  Max-to-Sum Ratio (planned), and
  [Hill Estimator](./incerto-hill-estimator.md).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/theorems/karamata.md`, revision `9717c9c`
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
