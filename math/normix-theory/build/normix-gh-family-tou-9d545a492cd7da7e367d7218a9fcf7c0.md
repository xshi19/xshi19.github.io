---
title: A theory tour of the GH family
---

Choosing a positive mixing law produces several familiar relatives of the
generalized hyperbolic distribution. This note follows their mathematical
nesting; the [upstream tutorial](https://xshi19.github.io/normix/tutorials/core/02_gh_family_tour.html)
retains executable constructions and plots.

## One mixing mechanism

Let $Y>0$ and $Z\sim\mathcal N_d(0,I_d)$ be independent, and put

```{math}
X=\mu+\gamma Y+\sqrt Y\,LZ,\qquad LL^\top=\Sigma\succ0.
```

Conditional on $Y=y$, the mean is $\mu+\gamma y$ and the covariance is
$y\Sigma$. The [conditioning introduction](./normix-conditioning-a-mixture.md)
uses $W=Y$, $\beta=\gamma$, and $\sigma^2=\Sigma$ in one dimension.
Drawing $Y$ then the conditional normal gives the joint law; keeping only
$X$ gives its marginal.

The source calls $Y$ a *subordinator*. Here it is just a positive random
variable. A stochastic-process subordinator is an increasing Lévy process;
using a one-time mixing law does not by itself specify a process, time scale,
or independent increments.

## Four mixing laws

The general [GIG kernel](./normix-generalized-inverse-gaussian.md) is

```{math}
g(y)\propto y^{p-1}\exp\!\left[-\tfrac12(ay+b/y)\right],\qquad y>0.
```

Its interior has $p\in\mathbb R$ and $a,b>0$. Gamma and inverse gamma arise
as normalized boundary limits, while inverse Gaussian is an exact subfamily:

| Mixing law | GIG relation | Parameters used here |
| --- | --- | --- |
| Gamma | $b\downarrow0$, $p>0$ | Shape $\alpha=p$, rate $\rho=a/2$ |
| Inverse gamma | $a\downarrow0$, $p<0$ | Shape $\alpha=-p$, scale $\rho=b/2$ |
| Inverse Gaussian | $p=-1/2$ | Mean $m=\sqrt{b/a}$, shape $\lambda=b$ |
| Generalized inverse Gaussian | Interior $a,b>0$ | $(p,a,b)$ |

For clarity, the two boundary densities are

```{math}
\begin{aligned}
g_{\mathrm{Gamma}}(y)&=\frac{\rho^\alpha}{\Gamma(\alpha)}
y^{\alpha-1}e^{-\rho y},\\
g_{\mathrm{InvGamma}}(y)&=\frac{\rho^\alpha}{\Gamma(\alpha)}
y^{-\alpha-1}e^{-\rho/y},\qquad \alpha,\rho>0.
\end{aligned}
```

The inverse-Gaussian density is

```{math}
g_{\mathrm{IG}}(y)=\sqrt{\frac{\lambda}{2\pi y^3}}
\exp\!\left[-\frac{\lambda(y-m)^2}{2m^2y}\right].
```

Expanding its exponent shows the GIG embedding
$(-1/2,\lambda/m^2,\lambda)$. Unlike the gamma limits, this equality holds
at positive finite parameters without a limiting operation.

## The induced marginal families

| Positive mixing law | Distribution of $X$ |
| --- | --- |
| Gamma | Variance gamma (VG) |
| Inverse gamma | Normal-inverse gamma |
| Inverse Gaussian | Normal-inverse Gaussian (NIG) |
| Generalized inverse Gaussian | Generalized hyperbolic (GH) |

The historical references retained from the source are Barndorff-Nielsen
(1977), *Exponentially decreasing distributions for the logarithm of particle
size*; Madan and Seneta (1990), *The Variance Gamma (V.G.) model for share market
returns*; and Barndorff-Nielsen (1997), *Normal inverse Gaussian distributions
and stochastic volatility modelling*. These identify the distribution families;
no finance-model derivation is imported here.

The phrase *GH family* often includes its gamma and inverse-gamma boundary
limits. An arbitrary positive mixing law gives a normal variance–mean mixture,
but need not be GH. The [GH density note](./normix-generalized-hyperbolic.md)
is specifically the result of integrating a GIG-normal joint law.

A familiar boundary example is Student $t$. If
$Y\sim\operatorname{InvGamma}(\nu/2,\nu/2)$ and $\gamma=0$, integration gives

```{math}
f_X(x)=\frac{\Gamma((\nu+d)/2)}
{\Gamma(\nu/2)(\nu\pi)^{d/2}|\Sigma|^{1/2}}
\left(1+\frac{(x-\mu)^\top\Sigma^{-1}(x-\mu)}{\nu}\right)^{-(\nu+d)/2}.
```

Here $\nu>0$, and $\Sigma$ is the scale matrix; the covariance is
$\nu\Sigma/(\nu-2)$ only when $\nu>2$.

## Symmetry, tails, and comparable scales

When $\gamma=0$, the marginal is symmetric about $\mu$. With finite second
mixing moment, every coordinate has excess kurtosis

```{math}
\gamma_2=3\frac{\operatorname{Var}(Y)}{\mathbb E[Y]^2}.
```

This follows from the conditional fourth normal moment and is strictly
positive for nonconstant $Y$. It becomes zero for constant mixing, which gives
a Gaussian. When $\gamma\ne0$, the conditional mean also varies with $Y$;
the [GH third and fourth moments](./normix-generalized-hyperbolic.md#gh-skew-kurt)
quantify the resulting asymmetry whenever those moments exist.

Interior GIG mixing and its inverse-Gaussian subfamily have finite power
moments of every order. Inverse-gamma mixing has positive moments only below
its shape parameter. Consequently a symmetric normal-inverse-gamma marginal
can have a finite fourth moment even when the mixing variable has no fourth
moment: symmetry requires $\mathbb E[Y^2]$, whereas a nonzero mean coupling
can require $\mathbb E[Y^4]$. A plot alone cannot establish these boundaries.

For scalar comparisons with $\mathbb E[Y^2]<\infty$, centering and scaling use

```{math}
\frac{X-\mu-\gamma\mathbb E[Y]}
{\sqrt{\Sigma\mathbb E[Y]+\gamma^2\operatorname{Var}(Y)}}.
```

Comparing distributions at equal conditional $\Sigma$ does not generally put
them at equal marginal variance. The original tutorial's plotting cells and
notebook outputs are omitted here; no numerical plot is offered as a proof of
tail behavior.

Continue with [normal-mixture moments](./normix-normal-mixtures.md),
[joint and marginal structure](./normix-mixture-architecture.md), or
[EM for GH](./normix-em-algorithm.md). The
[upstream tutorial](https://xshi19.github.io/normix/tutorials/core/02_gh_family_tour.html)
and [package API](https://xshi19.github.io/normix/api/index.html) own construction
and sampling examples.

## Source and adaptation

Rewritten as a mathematical note from `xshi19/normix`, `docs/tutorials/core/02_gh_family_tour.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/tutorials/core/02_gh_family_tour.md)
records the original version. Notation, mathematical qualifications, and links
were adapted for this site; API recipes, executable package cells, and notebook
plots are omitted. No upstream benchmark execution or formal-proof verification
is claimed for this adaptation.

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
