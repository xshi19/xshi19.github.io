---
title: Entropy, varentropy, and Rényi entropy
---

This note derives formulas for exponential-family components and joint normal
variance–mean mixtures. Entropy is the mean of the surprisal, and varentropy
is its variance. For a wider structural account, the upstream source cites
Stankyavichyus (2026),
[*Varentropy: Overview, Computational Routes, and Structural Decomposition*](https://anatolyvitold.com/preprints/varentropy_decomposition.pdf),
a preprint. The derivations below concern the stated model families.

Read the [exponential-family core](./normix-exponential-family-core.md),
[Fisher geometry](https://xshi19.github.io/math/ig/information-geometry-fisher-vs-l2/), and the
[joint/marginal distinction](./normix-mixture-architecture.md) first.
Monte Carlo examples stay in the
[upstream varentropy tutorial](https://xshi19.github.io/normix/tutorials/stats/03_varentropy.html).

## Definitions

Let $X$ have density $p(x)$ with respect to a reference measure
$\mu(dx)$. The **information content** (surprisal) is
$\mathcal{I}(X) = -\log p(X)$. Its mean is the **entropy**
and its variance is the **varentropy**:

```{math}
:label: ve-defs

H = \mathbb{E}[-\log p(X)], \qquad
V_H = \operatorname{Var}[-\log p(X)].
```

Entropy measures the average surprisal; varentropy measures how much the
surprisal fluctuates around that average. Unlike the kurtosis, the varentropy
requires only $\mathcal{I}(X) \in L^2$, so it stays finite for many
heavy-tailed laws whose fourth moment diverges.

The **Rényi entropy** of order $\alpha > 0$, $\alpha \neq 1$
(Rényi, [1961, *On Measures of Entropy and Information*](https://digicoll.lib.berkeley.edu/record/112906)), is

```{math}
:label: ve-renyi

H_\alpha = \frac{1}{1-\alpha} \log \int p(x)^\alpha \, \mu(dx),
```

For Lebesgue measure $H$ is differential entropy; for a general reference
measure it is entropy relative to that measure. Logs are natural. Fix the
reference measure throughout. Where the density-power integral is finite
in an open interval around $1$ and differentiation under the integral is
justified, $H_\alpha\to H$ as $\alpha\to1$. These regularity assumptions
also justify the derivatives and local expansion below.

## Density-Power Route

All three quantities are governed by the **log density-power integral**

```{math}
:label: ve-R

R(\alpha) = \log \int p(x)^\alpha \, \mu(dx), \qquad R(1) = 0.
```

Writing $\mathcal{I} = -\log p(X)$, the function $R(1-s)$ is the
cumulant-generating function of $\mathcal{I}$, so differentiating at
$\alpha = 1$ gives

```{math}
:label: ve-cumulants

H = -R'(1), \qquad V_H = R''(1),
```

while the Rényi entropy is $H_\alpha = R(\alpha)/(1-\alpha)$. The
first-order expansion

```{math}
H_\alpha = H - \tfrac{1}{2} V_H\,(\alpha - 1) + \mathcal{O}\!\left((\alpha-1)^2\right)
```

shows that the varentropy is (twice the negative of) the slope of the Rényi
spectrum at $\alpha = 1$.

## Exponential Family

For an exponential family
$p(x\mid\theta) = h(x)\exp\{\theta^\top t(x) - \psi(\theta)\}$
whose carrier is **constant on its support**, $\log h(x) \equiv b_0$,
the normalized density power stays in the family with natural parameter
$\alpha\theta$, provided this parameter belongs to the natural domain:

```{math}
:label: ve-R-ef

R(\alpha) = (\alpha - 1)\,b_0 + \psi(\alpha\theta) - \alpha\,\psi(\theta).
```

Substituting into [](#ve-cumulants) and using $\eta = \nabla\psi(\theta)$
and the Fisher information $I(\theta) = \nabla^2\psi(\theta)$ gives closed
forms in terms of the log-partition triad:

```{math}
:label: ve-ef-formulas

\begin{aligned}
H &= \psi(\theta) - \theta^\top \eta - b_0, \\
V_H &= \theta^\top I(\theta)\,\theta, \\
H_\alpha &= \frac{(\alpha-1)\,b_0 + \psi(\alpha\theta) - \alpha\,\psi(\theta)}{1-\alpha}.
\end{aligned}
```

The varentropy identity $V_H = \theta^\top I(\theta)\,\theta$ holds because
the centered information content
$\mathcal{I}(X) - H = -\theta^\top\{t(X) - \eta\}$ lies exactly in the
span of the score. This covers gamma, inverse gamma,
[GIG](./normix-generalized-inverse-gaussian.md), and multivariate normal
families when their carrier is chosen as $h=1$ on their fixed supports.
For a $d$-dimensional normal law, $V_H=d/2$.

When $\log h$ is not constant, the general variance formula is

```{math}
V_H=\theta^\top I(\theta)\theta
+2\theta^\top\operatorname{Cov}(t(X),\log h(X))
+\operatorname{Var}(\log h(X)).
```

For example, the two-parameter inverse-Gaussian representation has
$\log h(y)=-\tfrac12\log(2\pi)-\tfrac32\log y$. One can instead use its
exact embedding $\operatorname{GIG}(-1/2,\lambda/m^2,\lambda)$ with $h=1$
and statistic $\log y$. Along the density-power path, the GIG order then
varies; holding it at $-1/2$ would omit part of the surprisal.

## Varentropy of the GIG Distribution

Let $Y \sim \mathrm{GIG}(p, a, b)$ with density [](./normix-generalized-inverse-gaussian.md#gig-pdf) and natural
parameters $\theta = [p-1,\,-b/2,\,-a/2]$. Raising the density to the
power $\alpha$ keeps it proportional to a GIG density with parameters
$(1 + \alpha(p-1),\, \alpha a,\, \alpha b)$ — the escort-closure property.
Using the Bessel integral
$\int_0^\infty y^{q-1} e^{-(uy + v/y)/2}\,dy = 2\,(v/u)^{q/2} K_q(\sqrt{uv})$
([NIST DLMF, 10.32.10](https://dlmf.nist.gov/10.32.E10), after a change of variable)
gives, with $z = \sqrt{ab}$,

```{math}
:label: ve-gig-R

R(\alpha) = (1-\alpha)\log 2 + \frac{1-\alpha}{2}\log\!\frac{b}{a}
+ \log K_{1 + \alpha(p-1)}(\alpha z) - \alpha \log K_p(z).
```

Define $F(p, z) = \log K_p(z)$ and the first-order differential operator

```{math}
:label: ve-L

L = (p-1)\,\partial_p + z\,\partial_z.
```

Along the escort path $\alpha \mapsto (1 + \alpha(p-1),\, \alpha z)$ the derivatives at $\alpha=1$ are
$\frac{d}{d\alpha}F=LF$ and
$\frac{d^2}{d\alpha^2}F=(L^2-L)F$. The subtraction accounts for
the variable coefficients of $L$. All remaining
terms of [](#ve-gig-R) are linear in $\alpha$, so from
[](#ve-cumulants),

```{math}
:label: ve-gig

\boxed{\;
V_H\{\mathrm{GIG}(p,a,b)\} = (L^2 - L)\log K_p(z),
\qquad z = \sqrt{ab}.
\;}
```

Expanded, this is the pure second-order form

```{math}
V_H = (p-1)^2 F_{pp} + 2(p-1)\,z\,F_{pz} + z^2 F_{zz},
```

which coincides with the Fisher quadratic form
$\theta^\top I(\theta)\,\theta$ of [](#ve-ef-formulas). The entropy
follows from $H = -R'(1)$:

```{math}
:label: ve-gig-entropy

H\{\mathrm{GIG}(p,a,b)\} = \log\{2 K_p(z)\} + \tfrac12\log\!\frac{b}{a}
- L\log K_p(z).
```

For $a,b>0$, the density decays exponentially in $y$ at infinity and
in $1/y$ near zero, so the required logarithmic and power statistics have
finite second moments. Thus $V_H$ is finite for every interior GIG law.
The proper inverse-gamma boundary $a=0$, $p=-k<0$, $b>0$ also has finite
varentropy for every $k>0$, even when its fourth moment fails ($k\leq4$).
That boundary fact follows from the inverse-gamma density, not from
substitution of $z=0$ in an interior Bessel expression. It is not a uniform
bound as the shape approaches zero.

## Joint Varentropy of Normal Variance-Mean Mixtures

Consider the joint law of [](./normix-generalized-hyperbolic.md#gh-joint) for a positive mixing variable,

```{math}
X \mid Y = y \sim \mathcal{N}_d(\mu + \gamma y,\ \Sigma y), \qquad
Y \sim g_\vartheta.
```

Conditionally on $Y$, the quadratic form
$Q = (X - \mu - \gamma Y)^\top \Sigma^{-1}(X - \mu - \gamma Y)/Y$ is
$\chi^2_d$ and independent of $Y$, so

```{math}
-\log p(X \mid Y) = C_\Sigma + \tfrac{d}{2}\log Y + \tfrac12 Q, \qquad
C_\Sigma = \tfrac{d}{2}\log(2\pi) + \tfrac12\log|\Sigma|.
```

Writing $\mathcal{I}_Y = -\log g_\vartheta(Y)$ for the mixing-law
surprisal, the joint information content is
$\mathcal{I}_{X,Y} = C_\Sigma + \mathcal{I}_Y + \tfrac{d}{2}\log Y +
\tfrac12 Q$. Since $\operatorname{Var}(\tfrac12 Q) = d/2$ and
$Q \perp Y$,

```{math}
:label: ve-joint

\boxed{\;
V_H(X, Y) = \frac{d}{2}
+ \operatorname{Var}\!\left[\mathcal{I}_Y + \frac{d}{2}\log Y\right].
\;}
```

The representation is $X=\mu+\gamma Y+\sqrt Y LZ$, with
$LL^\top=\Sigma$, $Z\sim\mathcal N_d(0,I_d)$ independent of $Y$, and
$\Sigma$ positive definite. Thus $Q=Z^\top Z$ is independent of $Y$.
The notation maps to $W=Y$, $\beta=\gamma$, $\sigma^2=\Sigma$ in the scalar
[conditioning note](./normix-conditioning-a-mixture.md).

The Gaussian layer contributes exactly $d/2$ to joint varentropy.
For a fixed mixing law, $\mu$, $\gamma$, and $\Sigma$ all drop out of this
variance. The joint entropy is

```{math}
H(X,Y)=H(Y)+\frac d2\log(2\pi e)+\frac12\log|\Sigma|
+\frac d2\mathbb E[\log Y],
```

when the terms are finite. In particular, $\mu$ and $\gamma$ also drop out
of joint entropy; only $\Sigma$ contributes among the normal parameters.
More generally, Gaussian integration gives the joint density-power route

```{math}
\begin{aligned}
R_{X,Y}(\alpha)
&=(1-\alpha)C_\Sigma-\frac d2\log\alpha\\
&\quad+\log\int_0^\infty g_\vartheta(y)^\alpha
y^{(1-\alpha)d/2}\,dy.
\end{aligned}
```

The integral must be finite at the requested Rényi order. These are
**joint** information quantities. They do not give the entropy or varentropy
of the GH marginal $X$ by deleting the hidden variable; the marginal's
surprisal is $-\log\int p(x,y)\,dy$.

For GIG mixing, $\mathcal{I}_Y = \psi_{\mathrm{GIG}}(\theta) -
\theta^\top t(Y)$ with $t(Y) = [\log Y,\, Y^{-1},\, Y]$, so adding
$\tfrac{d}{2}\log Y$ shifts only the coefficient of $\log Y$:

```{math}
\mathcal{I}_Y + \tfrac{d}{2}\log Y
= \psi_{\mathrm{GIG}}(\theta) - (\theta - \delta_d)^\top t(Y),
\qquad \delta_d = (\tfrac{d}{2},\, 0,\, 0)^\top.
```

Its variance is the shifted Fisher quadratic form, and translating to classical
coordinates gives the operator $L_d$ of [](#ve-L) with the order
shifted by $-d/2$:

```{math}
:label: ve-joint-gig

\boxed{\;
V_H(X, Y) = \frac{d}{2} + (L_d^2 - L_d)\log K_p(z),
\qquad
L_d = \left(p - 1 - \tfrac{d}{2}\right)\partial_p + z\,\partial_z.
\;}
```

For $d = 0$ this reduces to the GIG varentropy [](#ve-gig). The
operator shift $p - 1 \mapsto p - 1 - d/2$ is exactly the effect of the
conditional Gaussian volume factor $Y^{-d/2}$, and mirrors the natural
parameter $\theta_1 = p - 1 - d/2$ of the joint exponential family
[](./normix-generalized-hyperbolic.md#gh-natural-params). The variance-gamma and normal-inverse-gamma joints arise at the proper
boundaries $b=0$, $p>0$ and $a=0$, $p<0$, respectively; they require their
own log-partition formulas. Normal-inverse-Gaussian mixing has $p=-1/2$
and can be evaluated through the GH embedding so that the order varies
faithfully along the density-power path.

## Coordinate and moment limits

Under an invertible affine change $U=AX+c$ of a continuous vector,
$H(U)=H(X)+\log|\det A|$ and $V_H(U)=V_H(X)$, because the Jacobian
contributes a constant. A nonlinear transformation generally adds a random
log-Jacobian and can change varentropy. All formulas above use the stated
Lebesgue coordinates, rather than a coordinate-free information measure.

Finiteness of surprisal variance is different from finiteness of $X$'s
fourth moment. It can make varentropy useful for heavy-tailed families,
but does not by itself order all distributions by tail weight or guarantee
accurate finite-sample estimation. The [GH family tour](./normix-gh-family-tour.md)
identifies the mixing boundaries; [online EM](./normix-online-em.md) and
[shrinkage](./normix-shrinkage.md) concern parameter estimation rather than
an estimator or convergence guarantee for these information quantities.

## Source and adaptation

Adapted from `xshi19/normix`, `docs/theory/varentropy.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/theory/varentropy.md)
records the original version. Notation, mathematical qualifications, and links
were adapted for this site. Package interfaces, fitter recipes, and executable
cells are omitted; no upstream benchmark or formal-proof verification is claimed.

:::{dropdown} MIT permission notice

MIT License

Copyright (c) 2020 xshi19

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
