---
title: Generalized hyperbolic distribution
---

The generalized hyperbolic (GH) distribution combines a conditional Gaussian
law with a [GIG mixing variable](./normix-generalized-inverse-gaussian.md).
Barndorff-Nielsen introduced the family in *Exponentially decreasing
distributions for the logarithm of particle size* (1977),
*Proceedings of the Royal Society A* 353, 401–419.

## Definition as a normal mixture

Let $Y\sim\operatorname{GIG}(p,a,b)$ with $a,b>0$, and let
$Z\sim\mathcal N_d(0,I_d)$ be independent of $Y$. For
$\mu,\gamma\in\mathbb R^d$ and $LL^\top=\Sigma\succ0$, define

```{math}
:label: gh-def
\begin{aligned}
X&=\mu+\gamma Y+\sqrt Y\,LZ,\\
X\mid Y=y&\sim\mathcal N_d(\mu+\gamma y,y\Sigma).
\end{aligned}
```

This uses **standard** normal noise, as in
[conditioning a mixture](./normix-conditioning-a-mixture.md): in one dimension
$Y=W$, $\gamma=\beta$, $L=\sigma$, and $\Sigma=\sigma^2$.
The upstream convention absorbs $L$ into the noise and writes
$Z_{\mathrm{up}}=LZ\sim\mathcal N_d(0,\Sigma)$.

The location $\mu$ need not be the marginal mean, and $\Sigma$ need not be
the marginal covariance. The vector $\gamma$ couples the conditional mean
to the mixing variable. The parameters $(p,a,b)$ control the mixing law.
A general positive mixing distribution gives a
[normal variance–mean mixture](./normix-normal-mixtures.md); the GIG choice
selects GH. This is distinct from restricting attention to finite mixtures
with a discrete component label.

## Joint density and marginal density

Multiplying the normal conditional density by the GIG density gives

```{math}
:label: gh-joint
\begin{aligned}
f(x,y)&=\frac{(a/b)^{p/2}y^{p-1-d/2}}
{(2\pi)^{d/2}|\Sigma|^{1/2}2K_p(\sqrt{ab})}\\
&\quad\times\exp\!\left[-\frac{(x-\mu-\gamma y)^\top\Sigma^{-1}
(x-\mu-\gamma y)}{2y}-\frac{ay+b/y}{2}\right],\quad y>0.
\end{aligned}
```

Define the squared Mahalanobis distance $q(x)$ and three auxiliary quantities:

```{math}
q(x)=(x-\mu)^\top\Sigma^{-1}(x-\mu),\quad
A=a+\gamma^\top\Sigma^{-1}\gamma,\quad B(x)=b+q(x),\quad r=p-d/2.
```

Expansion of the exponent leaves a factor
$e^{(x-\mu)^\top\Sigma^{-1}\gamma}$ independent of $y$. The remaining integral
has the GIG normalizing form, with parameters $(r,A,B(x))$. Hence

```{math}
:label: gh-marginal
f_X(x)=\frac{(a/b)^{p/2}}{(2\pi)^{d/2}|\Sigma|^{1/2}K_p(\sqrt{ab})}
\left(\frac{B(x)}{A}\right)^{r/2}
K_r(\sqrt{AB(x)})\,e^{(x-\mu)^\top\Sigma^{-1}\gamma}.
```

This also proves the GIG posterior closure used in the
[EM derivation](./normix-em-algorithm.md). The
[mixture architecture note](./normix-mixture-architecture.md) distinguishes
the statistical roles of the joint and marginal laws.

## Alternative parameterization

Let $\delta=\sqrt{b/a}$ and $\omega=\sqrt{ab}$, using $\omega$ for the
source's scalar $\eta$ to reserve $\eta$ for expectation coordinates.
Then $Y=\delta V$ with $V\sim\operatorname{GIG}(p,\omega,\omega)$.
Set $S=\delta\Sigma$ and $g=\delta\gamma$. The same model is

```{math}
X=\mu+gV+\sqrt V\,S^{1/2}Z.
```

In [](#gh-marginal), replace $(\gamma,\Sigma,a,b)$ by
$(g,S,\omega,\omega)$. More explicitly, put
$A_\delta=\omega+g^\top S^{-1}g$ and
$B_\delta(x)=\omega+(x-\mu)^\top S^{-1}(x-\mu)$. Then

```{math}
f_X(x)=\frac{(B_\delta(x)/A_\delta)^{r/2}
K_r(\sqrt{A_\delta B_\delta(x)})}
{(2\pi)^{d/2}|S|^{1/2}K_p(\omega)}
e^{(x-\mu)^\top S^{-1}g}.
```

## Model identifiability

For any $c>0$, replace the latent variable by $Y'=cY$. Since
$Y'\sim\operatorname{GIG}(p,a/c,bc)$, the transformation

```{math}
:label: normix-gh-scale
T_c:(\mu,\gamma,\Sigma,p,a,b)
\longmapsto(\mu,\gamma/c,\Sigma/c,p,a/c,bc)
```

leaves the distribution of $X$ unchanged. Equivalently,
$(\mu,\gamma/c,\Sigma/c,p,c\delta,\omega)$ has the same marginal law.
The joint law on a *fixed* coordinate pair $(x,y)$ changes: its latent
coordinate has been rescaled.

The unconstrained marginal parameterization is therefore nonidentifiable.
Where the score and Fisher information exist, differentiating along this
constant-density curve gives a null score direction and a singular Fisher
matrix. This argument identifies a scale redundancy; it does not compute
marginal curvature or prove that every remaining parameterization is regular.

One can choose a representative by imposing $\delta=1$, $b=1$, or
$|\Sigma|=1$ in the interior. For the determinant convention, take
$c=|\Sigma|^{1/d}$ in [](#normix-gh-scale). Such a normalization must transform
**all** coupled parameters. Rescaling $\Sigma$ alone changes the model.
Fixing the determinant also does not control the matrix condition number:
$\operatorname{diag}(\varepsilon,\varepsilon^{-1})$ has determinant one
and becomes arbitrarily ill-conditioned.

Protassov (2004), *EM-based maximum likelihood parameter estimation for
multivariate generalized hyperbolic distributions*, and Hu (2005),
*Calibration of multivariate generalized hyperbolic distributions using the
EM algorithm*, discuss constrained estimation. The normalization here follows
directly from the displayed latent-variable transformation.

## Moments, skewness, and kurtosis

When the required mixing moments are finite, conditioning gives

```{math}
:label: gh-mean-cov
\begin{aligned}
\mathbb E[X]&=\mu+\gamma\mathbb E[Y],\\
\operatorname{Cov}(X)&=\mathbb E[Y]\Sigma
+\operatorname{Var}(Y)\gamma\gamma^\top.
\end{aligned}
```

For a coordinate $i$, write $m_k=\mathbb E[Y^k]$ and
$U_i=X_i-\mathbb E[X_i]$. Conditional on $Y$, its mean is
$\gamma_i(Y-m_1)$ and its variance is $\Sigma_{ii}Y$. The third and fourth
Gaussian moment formulas yield

```{math}
:label: gh-skew-kurt
\begin{aligned}
\mu_3(X_i)&=\gamma_i^3\mu_3(Y)
 +3\gamma_i\Sigma_{ii}\operatorname{Var}(Y),\\
\mu_4(X_i)&=\gamma_i^4\mu_4(Y)
 +6\gamma_i^2\Sigma_{ii}\mathbb E[(Y-m_1)^2Y]
 +3\Sigma_{ii}^2\mathbb E[Y^2].
\end{aligned}
```

Here $\mu_k$ denotes a central moment, and
$\mathbb E[(Y-m_1)^2Y]=m_3-2m_1m_2+m_1^3$.
Divide by $\operatorname{Var}(X_i)^{3/2}$ for skewness and by
$\operatorname{Var}(X_i)^2$, then subtract three, for excess kurtosis.
All these moments exist for interior GIG mixing.

For $\gamma=0$, symmetry gives zero skewness, and excess kurtosis is
$3\mathbb E[Y^2]/\mathbb E[Y]^2-3$. This is $3/\alpha$ for gamma mixing
with shape $\alpha$, and $3/(\alpha-2)$ for inverse-gamma mixing when
$\alpha>2$. For a coordinate with $\gamma_i\ne0$, a fourth moment of the
inverse-gamma mixing variable is needed, requiring $\alpha>4$.
Do not apply interior moment claims blindly at the family boundaries.

## Joint exponential-family form

The complete-data density [](#gh-joint) has the form
$f_\theta(x,y)=h(x,y)\exp\{\langle\theta,t(x,y)\rangle-\psi(\theta)\}$, with

```{math}
t(x,y)=(\log y,\ y^{-1},\ y,\ x,\ x/y,\ xx^\top/y).
```

The first three blocks are scalars, the next two are vectors, and the last is
a symmetric matrix. Use the trace inner product for matrix blocks. Symmetric
matrices have $d(d+1)/2$ independent coordinates; treating duplicate off-diagonal
entries as independent would introduce an artificial nonminimal representation.

Expanding the quadratic in [](#gh-joint) identifies

```{math}
:label: gh-natural-params
\begin{aligned}
\theta_1&=p-1-d/2,&
\theta_2&=-\tfrac12(b+\mu^\top\Sigma^{-1}\mu),\\
\theta_3&=-\tfrac12(a+\gamma^\top\Sigma^{-1}\gamma),&
\theta_4&=\Sigma^{-1}\gamma,\\
\theta_5&=\Sigma^{-1}\mu,&
\theta_6&=-\tfrac12\Sigma^{-1}.
\end{aligned}
```

With $h(x,y)=(2\pi)^{-d/2}\mathbf1_{y>0}$ the log-partition is

```{math}
:label: gh-log-partition
\psi(\theta)=\tfrac12\log|\Sigma|+\log2+\log K_p(\sqrt{ab})
+\tfrac p2\log(b/a)+\mu^\top\Sigma^{-1}\gamma.
```

This parameterization is defined where the recovered $\Sigma\succ0$ and
$a,b>0$. The natural parameters determine the classical parameters uniquely
for this joint family. The observable GH family, with all parameters free,
does not inherit this complete-data exponential-family representation after
integration; see the [core note](./normix-exponential-family-core.md).

## Expectation coordinates and their inverse

Taking conditional Gaussian moments gives

```{math}
:label: gh-expectation
\begin{aligned}
\eta_1&=\mathbb E[\log Y],& \eta_2&=\mathbb E[Y^{-1}],& \eta_3&=\mathbb E[Y],\\
\eta_4&=\mathbb E[X]=\mu+\gamma\eta_3,\\
\eta_5&=\mathbb E[X/Y]=\mu\eta_2+\gamma,\\
\eta_6&=\mathbb E[XX^\top/Y]
=\Sigma+\mu\mu^\top\eta_2+\gamma\gamma^\top\eta_3
+\mu\gamma^\top+\gamma\mu^\top.
\end{aligned}
```

The first three entries are the GIG expectation coordinates, in the order
$(\log Y,Y^{-1},Y)$. The two vector equations form a linear system for
$\mu,\gamma$. Solving it, then rearranging the matrix equation, gives

```{math}
:label: gh-m-step
\begin{aligned}
\mu&=\frac{\eta_4-\eta_3\eta_5}{1-\eta_2\eta_3},\\
\gamma&=\frac{\eta_5-\eta_2\eta_4}{1-\eta_2\eta_3},\\
\Sigma&=\eta_6-\eta_5\mu^\top-\mu\eta_5^\top
 +\eta_2\mu\mu^\top-\eta_3\gamma\gamma^\top,\\
(p,a,b)&=\operatorname*{arg\,max}_{p,a,b}
 L_{\mathrm{GIG}}(p,a,b\mid\eta_1,\eta_2,\eta_3).
\end{aligned}
```

Cauchy–Schwarz gives $\eta_2\eta_3\geq1$, with equality only for a constant
positive $Y$. Interior GIG mixing is nonconstant, so the denominator is
nonzero. Near that limit it can be small. These inverse formulas require
attainable moments, a positive-definite recovered matrix, and an attained GIG
optimum. In [EM](./normix-em-algorithm.md), posterior averages replace the
population expectations.

## Hellinger distance for the joint law

For two joint densities, take the convention
$H^2=1-\int\sqrt{f_1f_2}\,dx\,dy$. The exponential-family representation
gives the exact expression

```{math}
1-H_{\mathrm{JGH}}^2=
\exp\!\left\{\psi\!\left(\frac{\theta^{(1)}+\theta^{(2)}}2\right)
-\frac{\psi(\theta^{(1)})+\psi(\theta^{(2)})}{2}\right\}.
```

The average is in **natural coordinates**. For comparison with classical
parameters, let $\bar\Sigma=(\Sigma_1+\Sigma_2)/2$,
$\Delta\mu=\mu_1-\mu_2$, $\Delta\gamma=\gamma_1-\gamma_2$, and let
$\bar p,\bar a,\bar b$ be arithmetic means. Define

```{math}
\bar a'=\bar a+\tfrac14\Delta\gamma^\top\bar\Sigma^{-1}\Delta\gamma,
\qquad
\bar b'=\bar b+\tfrac14\Delta\mu^\top\bar\Sigma^{-1}\Delta\mu.
```

Integrating the conditional Gaussian affinity and then the GIG kernel yields

```{math}
\begin{aligned}
1-H_{\mathrm{JGH}}^2
&=\frac{|\Sigma_1|^{1/4}|\Sigma_2|^{1/4}}{|\bar\Sigma|^{1/2}}
\frac{(a_1/b_1)^{p_1/4}(a_2/b_2)^{p_2/4}}
{\sqrt{K_{p_1}(\sqrt{a_1b_1})K_{p_2}(\sqrt{a_2b_2})}}\\
&\quad\times(\bar b'/\bar a')^{\bar p/2}K_{\bar p}(\sqrt{\bar a'\bar b'})
e^{-\Delta\mu^\top\bar\Sigma^{-1}\Delta\gamma/4}.
\end{aligned}
```

When the conditional Gaussian parameters agree, this reduces to the GIG
Hellinger distance. The upstream note attributes the joint-distance application
to Shi (2016), *Generalized Hyperbolic Distributions and Related Topics*.

Marginalization gives $H_X^2\leq H_{\mathrm{JGH}}^2$: for each $x$,
Cauchy–Schwarz bounds the joint affinity integral over $y$ by
$\sqrt{f_{X,1}(x)f_{X,2}(x)}$, and integration over $x$ proves the claim.
The joint distance depends on the latent representation; equivalent marginal
models can have different joint laws. No general closed expression for the
marginal GH distance is asserted here.

## Numerical limits and special cases

The upstream source reports small distributional errors despite larger errors
in some recovered GIG parameters. Those experiments are not rerun here, and do
not guarantee stability for every parameter regime. Moment inversion can suffer
from cancellation, Bessel overflow, and poorly conditioned covariance matrices.
The [optimization note](./normix-why-not-gradient-descent.md) separates these
issues from likelihood monotonicity.

| Choice of mixing parameters | Marginal family |
| --- | --- |
| $p=-1/2$, $a,b>0$ | Normal-inverse Gaussian (NIG) |
| $b\downarrow0$, $p>0$ | Variance gamma (VG) |
| $a\downarrow0$, $p<0$ | Normal-inverse gamma |
| $p=-\nu/2$, $a=0$, $b=\nu$, $\gamma=0$ | Student $t_\nu$ with location $\mu$ and scale matrix $\Sigma$ |
| $p=1$, $d=1$ | Univariate hyperbolic |

In the conventional multivariate hyperbolic density, the corresponding order
is $p=(d+1)/2$. The [family tour](./normix-gh-family-tour.md) develops the
nesting and moment boundaries. For supported operations, see the
[upstream package API](https://xshi19.github.io/normix/api/index.html).

## Source and adaptation

Adapted from `xshi19/normix`, `docs/theory/gh.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/theory/gh.md)
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
