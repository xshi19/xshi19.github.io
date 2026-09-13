---
title: Cramér Exponential-Moment Condition
options:
  concept:
    id: cramer-condition
    type: theorem
    depends_on:
      - regular-variation
    tags:
      - thin-tails
      - large-deviations
      - exponential-moments
---

## Statement

A random variable $X$ satisfies a right-tail Cramér condition when there is
some $\theta>0$ such that

$$
\mathbb E[e^{\theta X}]<\infty.
$$

For two-sided large-deviation statements, one usually asks for the moment
generating function $M(\theta)=\mathbb E[e^{\theta X}]$ to be finite in an
open neighborhood of $0$.  Equivalently, the cumulant generating function
$\Lambda(\theta)=\log \mathbb E[e^{\theta X}]$ is finite near zero, and
the Cramér rate function is

$$
I(x)=\sup_\theta\{\theta x-\Lambda(\theta)\}.
$$

We use the one-sided version when the question is upper-tail
concentration; the classical two-sided Cramér large-deviation theorem requires
stronger mgf control.

The condition gives the elementary Chernoff bound

$$
\mathbb P(X>x)\le e^{-\theta x}\mathbb E[e^{\theta X}],
$$

so the upper-tail probability is bounded above by an exponentially decaying
function.  Equivalently, $\bar F(x)=O(e^{-\theta x})$ for that value of
$\theta$.

Here $\bar F(x)=\mathbb P(X>x)$ is the survival function, $\mathbb E$
is expectation, and $\theta$ is the exponential-tilting parameter. Shared
notation is planned; all symbols needed here are defined locally.

## Thin-tail intuition

The Cramér condition is a thin-tail gate.  If exponential moments exist, then
exponential tilting, Chernoff bounds, and classical large-deviation rates have
room to operate.  If no positive exponential moment exists, those tools may
give a false sense of security.

This is the clean contrast with [Pareto-type](./incerto-pareto.md) fat
tails.  A [regularly varying](./incerto-regular-variation.md) tail can have many finite
ordinary moments, but multiplying by $e^{\theta X}$ eventually overwhelms
every power-law decay. Failure of the Cramér condition does not require
infinite variance: it can also occur when both the mean and variance are finite.

## Chernoff bound and examples

We prove the displayed Chernoff bound from Markov's inequality.  The right-tail
and two-sided Cramér conditions are definitions or standard large-deviation
hypotheses; the larger Cramér theorem is cited rather than proved here.  The
normal, exponential, and Pareto cases below are direct checks of the condition.

The Chernoff bound follows from Markov's inequality applied to the nonnegative
random variable $e^{\theta X}$:

$$
\mathbb P(X>x)
=
\mathbb P(e^{\theta X}>e^{\theta x})
\le
e^{-\theta x}\mathbb E[e^{\theta X}].
$$

A standard normal random variable satisfies the two-sided version because
$\mathbb E[e^{\theta X}]=e^{\theta^2/2}$ for all real $\theta$.  An exponential
random variable with rate $\lambda>0$ has mgf
$\lambda/(\lambda-\theta)$ for $\theta<\lambda$, so it also satisfies
the two-sided neighborhood condition. Its positive exponential moments exist
exactly for $0<\theta<\lambda$.

For a [Pareto Type I](./incerto-pareto.md) random variable,

$$
\mathbb E[e^{\theta X}]
=
\int_{x_m}^{\infty} e^{\theta x}\alpha x_m^\alpha x^{-(\alpha+1)}\,dx.
$$

For every $\theta>0$, the exponential factor dominates the polynomial decay,
so the integral diverges.  Thus Pareto tails fail the right-tail Cramér
condition for every positive $\theta$.

## Truncated exponential moment check

(cramer-condition-truncated-moment-check)=
For an exponential variable of rate $\lambda$ and $c\ge0$,

$$
\mathbb E[e^{\theta X}\mathbf 1_{\{X\le c\}}]
=\frac{\lambda}{\lambda-\theta}
  \left(1-e^{-(\lambda-\theta)c}\right),
\qquad 0<\theta<\lambda.
$$

For example, $\lambda=1$ and $\theta=0.12$ give a finite limit
$1/0.88\approx1.13636$ as $c\to\infty$.

For Pareto Type I with $\alpha>0$, every $\theta>0$ instead gives a
lower bound that diverges. For $c\ge2x_m$, retain only the interval $(c/2,c]$:

$$
\begin{aligned}
\mathbb E[e^{\theta X}\mathbf 1_{\{X\le c\}}]
&\ge e^{\theta c/2}\mathbb P(c/2<X\le c)\\
&=(2^\alpha-1)\left(\frac{x_m}{c}\right)^\alpha
  e^{\theta c/2}\longrightarrow\infty.
\end{aligned}
$$

These are population calculations, independent of simulation. Even a Pareto
law with $\alpha=2.5$, finite mean, and finite variance has no positive
exponential moment.

## Caveats

- The condition is direction-specific unless the moment generating function is
  finite around both sides of zero.
- Failure of the Cramér condition does not by itself prove regular variation
  or subexponentiality.  A lognormal distribution has every positive polynomial
  moment finite but no positive exponential moment, so "all moments finite" is
  not enough for the Cramér condition.
- Empirical samples cannot prove that an exponential moment exists; a finite
  sample always has a finite empirical exponential average.
- Large-deviation results need more than the one-line condition here.  This
  page records the gate and the thin-tail contrast, not the full theorem.

## References

- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [1971, 2nd ed., Wiley](https://www.wiley-vch.de/de/fachgebiete/mathematik-und-statistik/an-introduction-to-probability-theory-and-its-applications-volume-2-978-0-471-25709-7).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Taleb, *Statistical Consequences of Fat Tails* [2020](https://arxiv.org/abs/2001.10488).

## Backlinks

- Depends on: [Regular Variation](./incerto-regular-variation.md).
- Used by: [Subexponentiality](./incerto-subexponentiality.md) and thin-tail contrast
  pages.

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/theorems/cramer-condition.md`, revision `9717c9c`
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
