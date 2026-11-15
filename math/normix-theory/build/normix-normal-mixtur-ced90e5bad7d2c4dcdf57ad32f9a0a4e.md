---
title: Normal variance–mean mixtures
---

Variance gamma, normal-inverse gamma, normal-inverse Gaussian, and generalized
hyperbolic distributions share a conditional Gaussian construction. This note
extracts its moment and probability identities from the upstream tutorial.
Begin with [conditioning a mixture](./normix-conditioning-a-mixture.md) for
the scalar example, or the [family tour](./normix-gh-family-tour.md) for the
choice of positive mixing law.

## The model and notation

Let $Z\sim\mathcal N_d(0,I_d)$ be independent of a nonnegative random variable
$Y$. Given $\mu,\gamma\in\mathbb R^d$ and $LL^\top=\Sigma\succ0$, define

```{math}
:label: normix-normal-mixture-model
X=\mu+\gamma Y+\sqrt Y\,LZ.
```

For $d=1$, use $Y=W$, $\gamma=\beta$, and $L=\sigma$, so this is exactly
$X=\mu+\beta W+\sigma\sqrt W\,Z$ from the conditioning note. Literature
that writes a noise vector with covariance $\Sigma$ has absorbed $L$ into it.
For $y>0$, the conditional law is normal with mean $\mu+\gamma y$ and
covariance $y\Sigma$; at $y=0$ it is a point mass at $\mu$.

The four named families in the source use strictly positive mixing variables.
They are not the entire set of normal mixtures. In particular, taking $Y$
constant or discrete also fits [](#normix-normal-mixture-model).

## Mean and covariance by conditioning

Assume $\mathbb E[Y^2]<\infty$, a sufficient condition for all terms below.
The laws of total expectation and total covariance give

```{math}
:label: normix-normal-mixture-moments
\begin{aligned}
\mathbb E[X]&=\mathbb E[\mu+\gamma Y]=\mu+\gamma\mathbb E[Y],\\
\operatorname{Cov}(X)
&=\mathbb E[\operatorname{Cov}(X\mid Y)]
 +\operatorname{Cov}(\mathbb E[X\mid Y])\\
&=\mathbb E[Y]\Sigma+\operatorname{Var}(Y)\gamma\gamma^\top.
\end{aligned}
```

The first term averages variation within conditional normals. The second is
a positive-semidefinite rank-at-most-one contribution from variation in their
means. Thus $\mu$ is a location parameter and $\Sigma$ a conditional scale
matrix; neither is generally the corresponding marginal moment.

If $\gamma=0$, only $\mathbb E[Y]<\infty$ is needed for the covariance
formula $\operatorname{Cov}(X)=\mathbb E[Y]\Sigma$. The stronger second-moment
assumption is convenient for the general mean-coupled construction, not part
of the definition of a mixture.

For constant mixing $Y=c>0$, the law reduces to
$\mathcal N_d(\mu+c\gamma,c\Sigma)$. This also shows why a model with almost
constant mixing can have difficulty separating location from mean coupling.

## An exact two-dimensional calculation

Take

```{math}
\mu=\begin{pmatrix}0\\0\end{pmatrix},\quad
\gamma=\begin{pmatrix}3/10\\-2/5\end{pmatrix},\quad
\Sigma=\begin{pmatrix}1&3/10\\3/10&1\end{pmatrix}.
```

For gamma mixing with shape and rate both $3/2$,
$\mathbb E[Y]=1$ and $\operatorname{Var}(Y)=2/3$. Therefore

```{math}
\mathbb E[X]=\begin{pmatrix}3/10\\-2/5\end{pmatrix},\qquad
\operatorname{Cov}(X)=
\begin{pmatrix}53/50&11/50\\11/50&83/75\end{pmatrix}.
```

The positive conditional off-diagonal entry decreases from $3/10$ to $11/50$
because $\gamma_1\gamma_2<0$. The additional covariance matrix is still
positive semidefinite: its off-diagonal entries need not be positive.
This is an algebraic example, not an empirical fit or simulation result.

## Joint draws, marginal density, and the posterior

If the positive mixing law has density $g$, then

```{math}
f(x,y)=\varphi_d(x;\mu+\gamma y,y\Sigma)g(y),\qquad
f_X(x)=\int_0^\infty f(x,y)\,dy.
```

Here $\varphi_d$ is a multivariate normal density. Sampling $(X,Y)$ and
retaining $X$ has exactly the marginal law. Observing $X$ alone, however,
does not reveal the latent draw; its uncertainty is represented by

```{math}
f(y\mid x)=\frac{\varphi_d(x;\mu+\gamma y,y\Sigma)g(y)}{f_X(x)}
```

where the denominator is positive and finite. For GIG mixing the posterior
is another GIG, as derived in [EM](./normix-em-algorithm.md#normix-em-posterior).
The [mixture architecture](./normix-mixture-architecture.md) explains why the
joint law provides useful sufficient statistics for estimation of the marginal.

These density formulas assume $Y>0$. If $Y$ has an atom at zero, the marginal
has the corresponding atom at $\mu$ in addition to its positive-mixing part;
a Lebesgue density alone does not describe the entire law.

## Scalar distribution functions and quantiles

In one dimension, for strictly positive $Y$, conditioning gives

```{math}
F_X(x)=\mathbb E\!\left[
\Phi\!\left(\frac{x-\mu-\gamma Y}{\sigma\sqrt Y}\right)\right],
\qquad \sigma^2=\Sigma,
```

where $\Phi$ is the standard normal CDF. This identity requires no finite
moment of $Y$, since the integrand is bounded. A quantile is the generalized
inverse

```{math}
q_\tau=\inf\{x:F_X(x)\geq\tau\},\qquad 0<\tau<1.
```

An atom at $Y=0$ would add
$\Pr(Y=0)\mathbf1_{x\geq\mu}$ to the positive-mixing contribution.
The formula explains the scalar probability calculation without prescribing
an integration algorithm or package interface.

## Estimation and further reading

For the four named mixing families, [GH theory](./normix-generalized-hyperbolic.md)
and its limits give density and higher-moment formulas.
[EM](./normix-em-algorithm.md) replaces missing mixing statistics by posterior
expectations before maximizing the joint likelihood. Its monotonicity and
limitations follow the [ordinary EM entry](https://xshi19.github.io/math/ig/information-geometry-latent-variables-em/).
A fitted location or scale should still be interpreted through
[](#normix-normal-mixture-moments), not as an automatic estimate of a raw
marginal moment.

The [upstream normal-mixtures tutorial](https://xshi19.github.io/normix/tutorials/distributions/04_normal_mixtures.html)
retains constructors, plots, sampling, CDF/quantile operations, and fitted
examples. The [package API](https://xshi19.github.io/normix/api/index.html)
remains authoritative for implementation. No fitter, executable cell, or
notebook image is imported into this note.

## Source and adaptation

Rewritten as a mathematical note from `xshi19/normix`, `docs/tutorials/distributions/04_normal_mixtures.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/tutorials/distributions/04_normal_mixtures.md)
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
