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

The [SCoFT Prologue guide](incerto-scoft-ch1.md) separates Gaussian-relative
kurtosis from asymptotic tail classes. This catalog compares nonnegative
right-tail models by [subexponentiality](incerto-subexponentiality.md) and
[regular variation](incerto-regular-variation.md), then checks which moments
exist. The definitions concern a specified population law.

The positive classifications are cited, not proved here: Foss, Korshunov,
and Zachary, second edition (2013), §3.5, Theorem 3.29 and its following
paragraph, p. 57 (regular variation, including Pareto); Theorem 3.30 and
its following examples, p. 58 (lognormal and Weibull with shape below one).
See the [book](https://doi.org/10.1007/978-1-4614-7101-1).
We derive the elementary tail ratios and moment contrasts below; the
appendix proves the convolution decomposition.

Let $\bar F(x)=\mathbb P(X>x)$ be the survival function.  In the table,
"regularly varying" means $\bar F\in RV_{-\alpha}$ for some $\alpha>0$, so
$\bar F(tx)/\bar F(x)\to t^{-\alpha}$ for fixed $t>0$. Here $F$ is the CDF,
$x$ is the threshold, and $\alpha$ is a positive tail exponent. All examples
below have nonnegative support, with positive scale parameters, positive gamma
shape and rate, and nonzero lognormal log-variance. See [Shared notation](https://xshi19.github.io/math/notation/).

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

for distributions supported on $[0,\infty)$.  The converse fails: lognormal and
stretched-Weibull tails are subexponential without being regularly varying.

(tail-class-catalog-moment-gates)=
## Moment gates are different questions

The [Cramér condition](incerto-cramer-condition.md) asks whether
$\mathbb E e^{\theta X}<\infty$ for at least one $\theta>0$. Ordinary moments
ask about $\mathbb E X^q$, with $q>0$. For the following nonnegative models:

| Model | Finite positive power moments | Finite positive exponential moments |
| --- | --- | --- |
| Exponential, rate $1$ | Every $q>0$ | Exactly $0<\theta<1$ |
| Lognormal, $\log X\sim N(0,1)$ | Every $q>0$ | None |
| Pareto Type I, $\alpha>0$, $x_m>0$ | Exactly $0<q<\alpha$ | None |

The [Pareto moment proof](incerto-pareto-moment-existence.md) gives the last
row's power-moment boundary, including divergence at $q=\alpha$; the
[Cramér note](incerto-cramer-condition.md) proves its exponential-moment
failure. For the exponential row, direct integration gives
$\int_0^\infty x^q e^{-x}\,dx=\Gamma(q+1)<\infty$ and
$\int_0^\infty e^{-(1-\theta)x}\,dx=1/(1-\theta)$ when $\theta<1$;
the latter integral diverges when $\theta\ge1$. Here $\Gamma$ denotes
the gamma integral.

### Proof of the lognormal moment contrast

We prove both entries in the lognormal row. Let $Z\sim N(0,1)$ and
$X=e^Z$. Completing the square gives, for every real $q$,

$$
\mathbb E X^q
=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}e^{qz-z^2/2}\,dz
=e^{q^2/2}\frac{1}{\sqrt{2\pi}}
  \int_{-\infty}^{\infty}e^{-(z-q)^2/2}\,dz
=e^{q^2/2}.
$$

In contrast, for any fixed $\theta>0$,

$$
\mathbb E e^{\theta X}
=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}
  \exp(\theta e^z-z^2/2)\,dz=\infty.
$$

Indeed, $e^z/z^2\to\infty$ as $z\to\infty$, so the integrand is at
least $1$ for all sufficiently large $z$. The integral over that half-line
diverges. This establishes failure of the Cramér condition despite all
positive power moments being finite. Subexponentiality is a separate,
cited result; it does not follow just from this failure.

For a Pareto model with $\alpha=5/2$ and $x_m=1$, the mean is $5/3$
and the variance is $20/9$, but the fourth moment diverges. The formulas
follow from the linked integral: $\mathbb E X^2=5$ and
$5-(5/3)^2=20/9$. Pearson kurtosis is therefore not a finite summary for
this model. Having a finite variance does not imply a finite fourth or
positive exponential moment.

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
expansion in [NIST DLMF, Eq. 7.12.1](https://dlmf.nist.gov/7.12#E1).
It also gives
$\log\bar F(x)=-\tfrac12(\log x)^2-\log\log x-\tfrac12\log(2\pi)+o(1)$
for this standard lognormal as $x\to\infty$. Adding $cx$ or $p\log x$
proves the two exponential-versus-power comparisons stated above. For
$\log X\sim N(\mu,\sigma^2)$ with $\sigma>0$, replace $\log x$ in
the normal-tail calculation by $(\log x-\mu)/\sigma$; the same comparisons
and zero multiplier-ratio limit follow.
Thus the multiplier diagnostic separates power tails from all four other
examples, but cannot distinguish subexponential lognormal and stretched-Weibull
tails from the two light-tailed examples. The classifications of lognormal and
stretched Weibull are cited results; a ratio table is not a proof of their
subexponentiality.

## Exact light-tail checks

We prove the two negative classifications in the table. Write
$\operatorname{Gamma}(k,\lambda)$ in shape–rate notation, with $k,\lambda>0$.
At rate one, convolution of the densities with shapes $k$ and $\ell>0$
gives

$$
\frac{e^{-x}}{\Gamma(k)\Gamma(\ell)}
\int_0^x y^{k-1}(x-y)^{\ell-1}\,dy
=\frac{e^{-x}x^{k+\ell-1}}{\Gamma(k+\ell)},\qquad x>0,
$$

using the beta integral ([NIST DLMF, Eq. 5.12.1](https://dlmf.nist.gov/5.12#E1)).
Thus independent shapes add. Scaling by
$1/\lambda$ gives the same rule at a common rate $\lambda$.
The survival integral, compared with its leading integrand by l'Hôpital's
rule, gives

$$
\bar F(x)\sim\frac{(\lambda x)^{k-1}e^{-\lambda x}}{\Gamma(k)},
\qquad x\to\infty.
$$

For completeness, at rate one the derivatives of
$\int_x^\infty y^{k-1}e^{-y}\,dy$ and $x^{k-1}e^{-x}$ have ratio
$[1-(k-1)/x]^{-1}\to1$; both functions tend to zero. Consequently,
for fixed $t>1$,

$$
A_t(x)\sim t^{k-1}e^{-\lambda(t-1)x}\to0,\qquad
R(x)\sim\frac{\Gamma(k)}{2\Gamma(2k)}(\lambda x)^k\to\infty.
$$

This proves that every positive-shape gamma law, including the exponential
case $k=1$, is neither regularly varying with a finite positive exponent
nor subexponential.

For integer $k\ge1$ and rate one, repeated integration by parts gives the
exact survival $e^{-x}\sum_{j=0}^{k-1}x^j/j!$ for $x\ge0$. Hence

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

(tail-class-catalog-demo)=
## A finite model comparison

Run `uv run python demos/incerto/prologue_tail_classes.py` from the
repository root. It evaluates $\log A_2(u)$ for Pareto
$(\alpha=5/2,x_m=1)$, standard lognormal, Weibull $(\beta=1/2)$,
exponential (rate $1$), and gamma (shape $2$, rate $1$).
The fixed grid has 241 logarithmically spaced thresholds from $1$ to $10^4$;
all quantities are dimensionless population calculations. There is no
sampling, data fitting, truncation of a distribution, or uncertainty interval.

```{figure} ../assets/figures/incerto/prologue-tail-ratios.svg
:alt: Negative log multiplier ratios on logarithmic axes. Pareto is constant while the lognormal, stretched Weibull, exponential and gamma curves increase.

Theoretical survival functions evaluated on a finite grid. The ordinate is
$-\log[\bar F(2u)/\bar F(u)]$: an increasing curve means a decreasing
multiplier ratio. The lognormal and stretched-Weibull curves share their
zero-ratio limit with the two light-tailed models, although their convolution
tails belong to a different class.
```

The script also prints exact exponential and gamma two-sum checks at
$u=2,10,100$. The helper uses log survival values for the multiplier figure;
non-finite values cause an error. In particular, it does not divide zeros
when an ordinary survival function underflows. The figure is numerical
illustration of the formulas above, not evidence of an observed tail class.

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
  Subexponential Distributions* [2013, 2nd ed.](https://doi.org/10.1007/978-1-4614-7101-1),
  §3.5, Theorem 3.29 and following paragraph, p. 57; Theorem 3.30 and
  following examples, p. 58. These support the cited positive classifications.
- Taleb, *Statistical Consequences of Fat Tails*, third edition (2025),
  [arXiv:2001.10488v4](https://arxiv.org/abs/2001.10488v4), ch. 1,
  terminology discussion, pp. 4–5: context for the companion Prologue guide,
  not a theorem source for this table.

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
