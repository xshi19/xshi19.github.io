---
title: Pareto Moment Existence
options:
  concept:
    id: pareto-moment-existence
    type: theorem
    prerequisites: [pareto]
    related: [karamata]
    tags:
      - moments
      - pareto
---

(pareto-moment-theorem)=
## Statement

Let $X$ have a [Pareto Type I distribution](./incerto-pareto.md) with
lower cutoff $x_m>0$ and tail exponent $\alpha>0$:

$$
\bar F(x)=\mathbb P(X>x)=\left(\frac{x_m}{x}\right)^\alpha,
\qquad x\ge x_m.
$$

For any moment order $p>0$,

$$
\mathbb E[X^p] < \infty
\quad\Longleftrightarrow\quad
p<\alpha.
$$

When $p<\alpha$,

$$
\mathbb E[X^p]=\frac{\alpha x_m^p}{\alpha-p}.
$$

At $p=\alpha$, the truncated moment diverges logarithmically.  For
$p>\alpha$, it diverges as a power of the upper cutoff.  In particular, the
Pareto mean exists only for $\alpha>1$, and the variance exists only for
$\alpha>2$.

Here $\bar F$ denotes survival probability and $\mathbb E$ denotes
expectation; $p$ is a positive moment order.

## Proof by direct integration

We prove the Pareto Type I moment boundary and the moment formula by direct
integration.  The same calculation also explains the logarithmic boundary and
power-divergent regimes in the truncated-moment section.  The broader
regularly varying moment test is handled by [Karamata's Theorem](./incerto-karamata.md).

For $x\ge x_m$, the Pareto density is

$$
f(x)=\alpha x_m^\alpha x^{-(\alpha+1)}.
$$

For $p>0$,

$$
\mathbb E[X^p]
=
\int_{x_m}^{\infty}x^p\alpha x_m^\alpha x^{-(\alpha+1)}\,dx
=
\alpha x_m^\alpha
\int_{x_m}^{\infty}x^{p-\alpha-1}\,dx.
$$

(moment-exponent-threshold)=
The integral converges exactly when $p-\alpha-1<-1$, equivalently
$p<\alpha$.  Evaluating the convergent case gives

$$
\alpha x_m^\alpha
\cdot\frac{x_m^{p-\alpha}}{\alpha-p}
=
\frac{\alpha x_m^p}{\alpha-p}.
$$

Adding $\alpha+1$ to both sides proves the algebraic equivalence
$p-\alpha-1<-1 \Longleftrightarrow p<\alpha$. Lean formalization is
planned; the integral criterion and evaluation here are an ordinary proof.

This exact Pareto calculation is the simplest instance of the
[Karamata](./incerto-karamata.md) moment test for regularly varying tails.

## Truncated-moment behavior

For a finite upper cutoff $b\ge x_m$, define

$$
M_p(b)=\mathbb E[X^p\mathbf 1_{\{X\le b\}}].
$$

Direct integration gives

$$
M_p(b)=
\begin{cases}
\dfrac{\alpha x_m^\alpha}{\alpha-p}
\left(x_m^{p-\alpha}-b^{p-\alpha}\right), & p<\alpha,\\[1.1em]
\alpha x_m^\alpha\log(b/x_m), & p=\alpha,\\[0.8em]
\dfrac{\alpha x_m^\alpha}{p-\alpha}
\left(b^{p-\alpha}-x_m^{p-\alpha}\right), & p>\alpha.
\end{cases}
$$

Thus the same theorem has three operational regimes: convergence to a finite
moment, logarithmic boundary growth, and power growth.

(pareto-moment-existence-truncated-plot)=
For $\alpha=1.5$ and $x_m=1$, the three regimes reduce to:

| Moment order $p$ | Truncated moment $M_p(b)$, $b\ge1$ | Limit as $b\to\infty$ |
| --- | --- | --- |
| $1$ | $3(1-b^{-1/2})$ | $3$ |
| $1.5$ | $1.5\log b$ | Infinite, logarithmic growth |
| $2$ | $3(\sqrt b-1)$ | Infinite, power growth |

**What to notice.** For $p<\alpha$, the truncated moment levels off.  At
$p=\alpha$, it keeps growing, but only logarithmically.  For $p>\alpha$, the
upper tail contributes a visible power-law rise.

## Examples

- If $\alpha=0.8$, the mean is infinite because $1\ge\alpha$.
- If $\alpha=1.5$, the mean exists, but the variance is infinite because
  $2\ge\alpha$.
- If $\alpha=3$, the mean and variance exist, while the third raw moment is at
  the logarithmic boundary and diverges.

The theorem is exact, but sample estimates can still look misleading.  With
$\alpha$ close to a boundary, a finite run can appear calm until a new large
observation changes the empirical moment.

(pareto-moment-existence-sample-check)=
For $\alpha=1.2$ and $x_m=1$, the population mean is $6$ while the
second raw moment is infinite. Any finite sample of finite observations still
has a finite sample second moment. Its finiteness does not settle the
population question.

## Caveats

- Moment existence is a property of the generating distribution, not proof that
  a finite sample estimate will be accurate.
- When $\alpha$ is close to a boundary, convergence can be so slow that the
  formal moment is a poor operational summary.
- For two-sided heavy-tailed variables, check absolute moments or one-sided
  tails explicitly.  Symmetry can make a location parameter look finite while
  absolute exposure is infinite.
- Empirical Pareto fits require threshold checks.  A moment calculation using a
  fitted $\widehat\alpha$ inherits the uncertainty and bias of the tail fit.

## References

- Bingham, Goldie, and Teugels, *Regular Variation*
  [1987](https://doi.org/10.1017/CBO9780511721434).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).

## Backlinks

- Depends on: [Pareto Distribution](./incerto-pareto.md) and
  [Karamata's Theorem](./incerto-karamata.md).
- Used by: [LLN Failure Under Infinite Mean](./incerto-lln-failure.md),
  Pre-Asymptotic LLN Behavior (planned), and
  Max-to-Sum Ratio (planned).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/theorems/pareto-moment-existence.md`, revision `9717c9c`
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
