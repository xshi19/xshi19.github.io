---
title: Tail Class Catalog
options:
  concept:
    id: tail-class-catalog
    type: distribution
    depends_on:
      - subexponentiality
      - regular-variation
    tags:
      - fat-tails
      - asymptotics
      - distributions
---

## Main facts

We catalog common right-tail examples by two asymptotic properties:
[subexponentiality](./incerto-subexponentiality.md) and
[regular variation](./incerto-regular-variation.md). The classifications are
cited results; the appendix proves the elementary convolution decomposition.
Standard references are Bingham, Goldie, and Teugels for regular variation
[1987](https://doi.org/10.1017/CBO9780511721434), and Embrechts, Klueppelberg,
and Mikosch [1997](https://doi.org/10.1007/978-3-642-33483-2) plus Foss,
Korshunov, and Zachary [2013, 2nd ed.](https://doi.org/10.1007/978-1-4614-7101-1)
for subexponential tails.

Let $\bar F(x)=\mathbb P(X>x)$ be the survival function.  In the table,
"regularly varying" means $\bar F\in RV_{-\alpha}$ for some $\alpha>0$, so
$\bar F(tx)/\bar F(x)\to t^{-\alpha}$ for fixed $t>0$. Here $F$ is the CDF,
$x$ is the threshold, and $\alpha$ is a positive tail exponent. All examples
below have nonnegative support, with positive scale parameters, positive gamma
shape and rate, and nonzero lognormal log-variance. Shared notation is planned.

| Distribution | Subexponential? | Regularly varying? | Diagnostic tail behavior |
| --- | --- | --- | --- |
| [Pareto](./incerto-pareto.md), $\bar F(x)=(x_m/x)^\alpha$ | Yes | Yes | Exact power tail. |
| Lognormal | Yes | No | Heavier than any exponential, lighter than any power. |
| Weibull, $\bar F(x)=\exp(-x^\beta)$ with $0<\beta<1$ | Yes | No | Stretched exponential tail. |
| Exponential | No | No | Memoryless light tail. |
| Gamma | No | No | Exponential tail with a polynomial factor. |

For the lognormal row, "heavier than any exponential, lighter than any power"
means the survival function satisfies

$$
e^{cx}\bar F(x)\to\infty
\quad\text{for every }c>0,
\qquad
x^p\bar F(x)\to0
\quad\text{for every }p>0.
$$

The inclusion direction to remember is:

$$
\bar F\in RV_{-\alpha},\ \alpha>0
\quad\Longrightarrow\quad
F\text{ is subexponential},
$$

for distributions supported on $[0,\infty)$, or under the standard
corresponding right-tail assumptions.  The converse fails: lognormal and
stretched-Weibull tails are subexponential without being regularly varying.

## Reusable diagnostics

(tail-class-catalog-diagnostics)=
For a nonnegative continuous law with unbounded support, compare

$$
A_t(x)=\frac{\bar F(tx)}{\bar F(x)},\qquad
R(x)=\frac{\mathbb P(X_1+X_2>x)}{2\bar F(x)},
$$

where $X_1,X_2$ are iid and $t>1$ is fixed. The first ratio checks multiplicative
tail scaling; the second is the defining two-summand subexponential ratio.
These are population quantities. Finite evaluations of them do not certify
an asymptotic class from data.

| Distribution and parameters | Multiplier ratio $A_t(x)$ | Limit of $R(x)$ |
| --- | --- | --- |
| Pareto Type I, $x\ge x_m$ | $t^{-\alpha}$ | $1$ |
| Lognormal, $\log X\sim N(0,1)$ | $\bar\Phi(\log x+\log t)/\bar\Phi(\log x)\to0$ | $1$ |
| Weibull, $0<\beta<1$ | $\exp[-(t^\beta-1)x^\beta]\to0$ | $1$ |
| Exponential, rate $1$ | $e^{-(t-1)x}\to0$ | $\infty$ |
| Gamma, shape $2$, rate $1$ | $e^{-(t-1)x}(1+tx)/(1+x)\to0$ | $\infty$ |

Here $\bar\Phi$ is the standard normal survival function. For the lognormal
row, the normal-tail asymptotic $\bar\Phi(z)\sim\phi(z)/z$, with $\phi$ the
standard normal density, yields

$$
A_t(x)\sim
\frac{\log x}{\log x+\log t}
\exp\left[-(\log t)\log x-\frac{(\log t)^2}{2}\right]
\longrightarrow0.
$$

The asymptotic for $\bar\Phi$ follows from the complementary-error-function
expansion in [NIST DLMF, Section 7.12](https://dlmf.nist.gov/7.12).
Thus the multiplier diagnostic separates power tails from all four other
examples, but cannot distinguish subexponential lognormal and stretched-Weibull
tails from the two light-tailed examples. The classifications of lognormal and
stretched Weibull are cited results; a ratio table is not a proof of their
subexponentiality.

## Exact light-tail checks

Exponential and integer-shape gamma sums give elementary checks of $R$.
With shape–rate notation, $\operatorname{Gamma}(k,1)$ has survival
$e^{-x}\sum_{j=0}^{k-1}x^j/j!$ for $x\ge0$. Independence adds the shapes.
Consequently,

$$
R_{\operatorname{Exp}(1)}(x)=\frac{1+x}{2},
\qquad
R_{\operatorname{Gamma}(2,1)}(x)
=\frac{1+x+x^2/2+x^3/6}{2(1+x)}.
$$

| Threshold $x$ | Exponential sum ratio | Gamma shape 2 sum ratio |
| --- | --- | --- |
| $2$ | $1.5$ | $19/18\approx1.056$ |
| $10$ | $5.5$ | $683/66\approx10.348$ |
| $100$ | $50.5$ | $515303/606\approx850.335$ |

Both ratios diverge, although the gamma ratio happens to be near one at
$x=2$. A single moderate-threshold observation is insufficient to infer a
limiting tail class. The [Cramér condition](./incerto-cramer-condition.md)
provides the exponential-moment contrast for these two light-tailed laws.

## Caveats

- Regular variation and subexponentiality concern limits, not the shape of
  a fitted curve on a finite range.
- Very small survival probabilities can underflow in numerical calculations.
  Logarithmic ratios help preserve information; rounded zeros do not establish
  exact zero probability.
- The subexponential definition used here is the right-tail iid version for
  nonnegative summands. Two-sided and dependent settings need additional
  assumptions.
- The finite exact examples above check formulas, not the accuracy of an
  upstream sampler, survival-function implementation, or quadrature routine.

## Appendix: convolution calculation

This appendix proves the elementary two-summand convolution decomposition used
by the subexponential diagnostic.  We work with independent continuous
variables, then specialize to an iid lower-bounded distribution.  Discrete,
dependent, and two-sided variants require separate assumptions.

For independent continuous random variables $X_1$ and $X_2$ with densities
$f_1$, $f_2$ and survival functions $\bar F_1$, $\bar F_2$,

$$
\mathbb P(X_1+X_2>x)
=
\int_{-\infty}^{\infty} f_1(y)\bar F_2(x-y)\,dy.
$$

To see the iid reduction, take a common density $f$, survival function
$\bar F$, and lower support endpoint $a$, with $x>2a$.  Split the event
$\{X_1+X_2>x\}$ into three cases, ignoring probability-zero boundary points:
first $X_2\le x/2$, second $X_1\le x/2$, and third both variables exceed
$x/2$.  In the first case, conditioning on $X_2=y$ with $a\le y\le x/2$
leaves the requirement $X_1>x-y$, so this part contributes
$\int_a^{x/2} f(y)\bar F(x-y)\,dy$.  The second case contributes the same
quantity by iid symmetry.  The remaining upper-right square has probability
$\bar F(x/2)^2$ by independence.

(tail-class-catalog-convolution-split)=
For $a=0$, the partition can be recorded as disjoint regions rather than a
contour plot. Boundaries have zero probability under the continuous-law
assumption.

| Region within $\{X_1+X_2>x\}$ | Probability contribution |
| --- | --- |
| $0\le X_2\le x/2$, $X_1>x-X_2$ | $\int_0^{x/2}f(y)\bar F(x-y)\,dy$ |
| $0\le X_1\le x/2$, $X_2>x-X_1$ | The same integral by iid symmetry |
| $X_1>x/2$, $X_2>x/2$ | $\bar F(x/2)^2$ |

For iid variables with lower support endpoint $a$, symmetry gives the more
useful half-line formula

$$
\mathbb P(X_1+X_2>x)
=
2\int_a^{x/2} f(y)\bar F(x-y)\,dy+\bar F(x/2)^2.
$$

Dividing by $2\bar F(x)$ yields a normalized integral:

$$
\begin{aligned}
R(x)
&=\int_a^{x/2}\frac{f(y)\bar F(x-y)}{\bar F(x)}\,dy
  +\frac{\bar F(x/2)^2}{2\bar F(x)}\\
&=\int_a^{x/2}
\exp\{\log f(y)+\log\bar F(x-y)-\log\bar F(x)\}\,dy\\
&\quad+\frac12\exp\{2\log\bar F(x/2)-\log\bar F(x)\}.
\end{aligned}
$$

The logarithmic form applies where the factors are positive, with zero terms
understood by limits. It avoids forming tiny unnormalized probabilities before
taking their ratio. The decomposition is an ordinary conditioning argument;
no numerical integration or executable package helper is needed for it.

## References

- Bingham, Goldie, and Teugels, *Regular Variation*
  [1987](https://doi.org/10.1017/CBO9780511721434).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Foss, Korshunov, and Zachary, *An Introduction to Heavy-Tailed and
  Subexponential Distributions* [2013, 2nd ed.](https://doi.org/10.1007/978-1-4614-7101-1).

## Backlinks

- Depends on: [Subexponentiality](./incerto-subexponentiality.md) and
  [Regular Variation](./incerto-regular-variation.md).
- Example distribution: [Pareto Distribution](./incerto-pareto.md).
- Related diagnostic: [Max-to-Sum Ratio](./incerto-max-to-sum-ratio.md).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/distributions/tail-class-catalog.md`, revision `9717c9c`
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
