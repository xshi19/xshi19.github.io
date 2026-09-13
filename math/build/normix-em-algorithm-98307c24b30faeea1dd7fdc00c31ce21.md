---
title: EM for generalized hyperbolic distributions
---

Expectation–maximization fits a latent-variable model by alternating posterior
expectations and complete-data likelihood maximization. The general method is
due to Dempster, Laird, and Rubin,
[1977, *Maximum likelihood from incomplete data via the EM algorithm*](https://academic.oup.com/jrsssb/article-abstract/39/1/1/7027539).
This GH specialization follows the upstream derivation, which credits
Hu (2005), *Calibration of multivariate generalized hyperbolic distributions
using the EM algorithm*.

Read [ordinary EM](./information-geometry-latent-variables-em.md) for the
likelihood inequality, and [GH](./normix-generalized-hyperbolic.md) for the
joint density and parameter conventions. We treat independent observations
$x_1,\ldots,x_n\in\mathbb R^d$, with positive-definite $\Sigma$ and interior
GIG parameters $a,b>0$, unless a boundary case is stated.

## Complete and observed data

Write the classical parameter tuple as $\vartheta=(\mu,\gamma,\Sigma,p,a,b)$;
reserve $\theta$ for the joint family's natural coordinates. The model is

```{math}
\begin{aligned}
X&=\mu+\gamma Y+\sqrt Y\,LZ,\\
Y&\sim\operatorname{GIG}(p,a,b),\qquad Z\sim\mathcal N_d(0,I_d),\\
Y&\perp Z,\qquad LL^\top=\Sigma.
\end{aligned}
```

For $d=1$, the [conditioning note](./normix-conditioning-a-mixture.md) uses
$W=Y$, $\beta=\gamma$, and $\sigma^2=\Sigma$. The observed likelihood uses
$f_X(x;\vartheta)=\int f(x,y;\vartheta)\,dy$. The joint GH law is an
exponential family, which makes its expected log likelihood easier to optimize.

## Conditional distribution of the mixing variable

Expanding the normal quadratic in the joint density and collecting powers of
$y$ gives

```{math}
f(y\mid x;\vartheta)\propto y^{p-1-d/2}
\exp\!\left[-\tfrac12\left\{
\bigl(a+\gamma^\top\Sigma^{-1}\gamma\bigr)y
+\bigl(b+(x-\mu)^\top\Sigma^{-1}(x-\mu)\bigr)/y\right\}\right].
```

Set $r=p-d/2$, $A=a+\gamma^\top\Sigma^{-1}\gamma$, and
$B(x)=b+(x-\mu)^\top\Sigma^{-1}(x-\mu)$. All are evaluated at the current
iterate when used in an E-step. Then

```{math}
:label: normix-em-posterior
Y\mid X=x;\vartheta\sim\operatorname{GIG}(r,A,B(x)).
```

Interior assumptions guarantee $A,B(x)>0$ for every observation. The GIG
normalizing integral therefore supplies every real posterior power moment:

```{math}
:label: gig-moment-cond
\mathbb E_\vartheta[Y^s\mid X=x]
=\left(\frac{B(x)}{A}\right)^{s/2}
\frac{K_{r+s}(\sqrt{AB(x)})}{K_r(\sqrt{AB(x)})}.
```

Differentiating at $s=0$ yields

```{math}
\mathbb E_\vartheta[\log Y\mid X=x]
=\tfrac12\log(B(x)/A)
+\left.\partial_\nu\log K_\nu(\sqrt{AB(x)})\right|_{\nu=r}.
```

The three expectations of $Y$, $Y^{-1}$, and $\log Y$ must be evaluated
separately. Replacing $Y$ by its posterior mean inside nonlinear statistics
would give a different algorithm.

## E-step in one statistic order

Use the joint sufficient-statistic blocks
$t(x,y)=(\log y,y^{-1},y,x,x/y,xx^\top/y)$, as in the GIG and GH notes.
At iterate $\vartheta_k$ let

```{math}
u_j=\mathbb E_k[Y^{-1}\mid x_j],\qquad
v_j=\mathbb E_k[Y\mid x_j],\qquad
l_j=\mathbb E_k[\log Y\mid x_j].
```

The E-step averages all six blocks:

```{math}
:label: e-step
\begin{aligned}
\widehat\eta_1^{(k)}&=n^{-1}\sum_j l_j,&
\widehat\eta_2^{(k)}&=n^{-1}\sum_j u_j,&
\widehat\eta_3^{(k)}&=n^{-1}\sum_j v_j,\\
\widehat\eta_4^{(k)}&=n^{-1}\sum_j x_j,\\
\widehat\eta_5^{(k)}&=n^{-1}\sum_j x_j u_j,\\
\widehat\eta_6^{(k)}&=n^{-1}\sum_j x_jx_j^\top u_j.
\end{aligned}
```

The source EM note uses the first three slots in a different order. Here
$\widehat\eta_1$ always means the log moment, $\widehat\eta_2$ the inverse
moment, and $\widehat\eta_3$ the first moment. This order also governs the
[mixture architecture](./normix-mixture-architecture.md).

## M-step as an expectation-to-parameter map

Holding the old posterior fixed, maximize

```{math}
Q(\vartheta\mid\vartheta_k)
=\frac1n\sum_j\mathbb E_k[\log f(x_j,Y;\vartheta)\mid X=x_j].
```

In natural coordinates this equals
$\langle\theta,\widehat\eta^{(k)}\rangle-\psi(\theta)$ plus a constant.
For an attained interior maximum of the full joint family, the new model
matches the posterior sufficient-statistic averages. Suppressing $(k)$ on
$\widehat\eta$, the normal-block update is

```{math}
:label: m-step
\begin{aligned}
\mu_{k+1}&=\frac{\widehat\eta_4-\widehat\eta_3\widehat\eta_5}
{1-\widehat\eta_2\widehat\eta_3},\\
\gamma_{k+1}&=\frac{\widehat\eta_5-\widehat\eta_2\widehat\eta_4}
{1-\widehat\eta_2\widehat\eta_3},\\
\Sigma_{k+1}&=\widehat\eta_6-\widehat\eta_5\mu_{k+1}^\top
-\mu_{k+1}\widehat\eta_5^\top
+\widehat\eta_2\mu_{k+1}\mu_{k+1}^\top
-\widehat\eta_3\gamma_{k+1}\gamma_{k+1}^\top,\\
(p_{k+1},a_{k+1},b_{k+1})&=\operatorname*{arg\,max}_{p,a,b}
L_{\mathrm{GIG}}(p,a,b\mid\widehat\eta_1,\widehat\eta_2,\widehat\eta_3).
\end{aligned}
```

The two vector updates solve
$\widehat\eta_4=\mu+\gamma\widehat\eta_3$ and
$\widehat\eta_5=\mu\widehat\eta_2+\gamma$.
The covariance formula then follows from the sixth joint moment. The last
line is the [GIG likelihood problem](./normix-generalized-inverse-gaussian.md#gig-loglik);
it usually requires numerical optimization. Only the normal block is closed
form in the general GH case.

These formulas require a nonzero denominator, a positive-definite recovered
covariance, and a feasible attained mixing-law optimum. Positive-definiteness
can fail with insufficiently varied data. Convexity of the GIG subproblem
does not remove its domain constraints or numerical conditioning problems.

With exact posteriors and an M-step that increases $Q$, the
[EM likelihood inequality](./information-geometry-latent-variables-em.md#ig-em-monotonicity)
ensures

```{math}
\ell(\vartheta_{k+1})\geq\ell(\vartheta_k),\qquad
\ell(\vartheta)=n^{-1}\sum_j\log f_X(x_j;\vartheta).
```

This concerns the **observed** likelihood. It does not guarantee a global
MLE, a unique stationary point, convergence of the parameter sequence, or a
particular speed. Approximate numerical steps should be checked against their
actual objective.

## Scale normalization and equivariance

The [GH scale action](./normix-generalized-hyperbolic.md#normix-gh-scale) is

```{math}
T_c\vartheta=(\mu,\gamma/c,\Sigma/c,p,a/c,bc),\qquad c>0.
```

It represents the same observed law by replacing $Y$ with $Y'=cY$.
The posterior parameters become $(r,A/c,cB(x))$. Consequently,

```{math}
\begin{aligned}
\mathbb E_{T_c\vartheta}[Y^s\mid x]
 &=c^s\mathbb E_\vartheta[Y^s\mid x],\\
\mathbb E_{T_c\vartheta}[\log Y\mid x]
 &=\mathbb E_\vartheta[\log Y\mid x]+\log c.
\end{aligned}
```

The E-step output transforms as

```{math}
(\widehat\eta_1,\widehat\eta_2,\widehat\eta_3,
 \widehat\eta_4,\widehat\eta_5,\widehat\eta_6)
\longmapsto
(\widehat\eta_1+\log c,\widehat\eta_2/c,c\widehat\eta_3,
 \widehat\eta_4,\widehat\eta_5/c,\widehat\eta_6/c).
```

Substitution in [](#m-step), including the rescaled GIG optimum, proves
$F(T_c\vartheta)=T_cF(\vartheta)$ for the exact unconstrained EM map $F$
when the maximizers are unique (or selected consistently under scaling).

To impose $|\Sigma|=1$ after an update, choose $c=|\Sigma|^{1/d}$ and
transform all coupled parameters:

```{math}
(\mu,\gamma,\Sigma,p,a,b)\longmapsto
(\mu,|\Sigma|^{-1/d}\gamma,|\Sigma|^{-1/d}\Sigma,
 p,|\Sigma|^{-1/d}a,|\Sigma|^{1/d}b).
```

The observed density and its likelihood are unchanged. This is a choice of
scale representative, not a penalty or a guarantee of numerical stability.
Fixed bounds, shrinkage, approximate solvers, or inconsistent choices among
maximizers can break the exact equivariance argument.

## ECM and multiple cycles

Meng and Rubin's
[1993 ECM framework](https://academic.oup.com/biomet/article-abstract/80/2/267/251605)
replaces one M-step by conditional maximizations of parameter blocks.
The upstream note discusses its multi-cycle version (MCECM) for GH, also
citing McNeil, Frey, and Embrechts, *Quantitative Risk Management* (2010).
A two-cycle construction in the unconstrained redundant coordinates is:

1. Compute the posterior moments at $\vartheta_k$, then maximize the normal
   block $(\mu,\gamma,\Sigma)$ with $(p,a,b)$ fixed.
2. Recompute the posterior moments at this intermediate parameter and
   maximize the GIG block with the new normal block fixed.

Each conditional maximization must increase the corresponding current $Q$.
A full coupled scale normalization can then be applied without changing the
observed law. If instead one imposes a determinant constraint *inside* a
conditional maximization while holding the mixing parameters fixed, that is
a constrained optimization problem: simply normalizing the unconstrained
covariance is not a justification for its ascent property.

EM and valid MCECM updates share the likelihood-ascent argument. Neither has
a general guarantee of reaching the global MLE, and their numerical costs and
convergence rates depend on the problem.

## Special mixing families

Write $l=\mathbb E[\log Y]$, $u=\mathbb E[Y^{-1}]$, and $v=\mathbb E[Y]$
for the posterior averages at the relevant E-step. Denote the digamma function
by $\psi_0$ and its derivative by $\psi_1$, to distinguish them from the
log-partition $\psi$.

### Gamma mixing: variance gamma

For $Y\sim\operatorname{Gamma}(\alpha,\rho)$ with rate $\rho$,
$\mathbb E[\log Y]=\psi_0(\alpha)-\log\rho$ and
$\mathbb E[Y]=\alpha/\rho$. The mixing-law update solves

```{math}
\psi_0(\alpha)-\log\alpha=l-\log v,\qquad \rho=\alpha/v.
```

A Newton proposal is

```{math}
\alpha_{t+1}=\alpha_t-
\frac{\psi_0(\alpha_t)-\log\alpha_t-l+\log v}
{\psi_1(\alpha_t)-1/\alpha_t}.
```

Safeguards must keep $\alpha>0$ and verify improvement. The posterior is GIG
with $r=\alpha-d/2$, $A=2\rho+\gamma^\top\Sigma^{-1}\gamma$, and $B=q(x)$.
At $x=\mu$, $B=0$; posterior propriety and required inverse moments then
need separate checking. For example, this gamma posterior has finite inverse
moment only if $r>1$. The interior GIG formula does not cover that point
without a boundary analysis.

### Inverse-Gaussian mixing: NIG

For an inverse Gaussian with mean $m$ and shape $\lambda$,
$\mathbb E[Y]=m$ and $\mathbb E[Y^{-1}]=1/m+1/\lambda$.
The closed-form mixing update is

```{math}
m=v,\qquad \lambda=\frac{1}{u-1/v}.
```

It requires $uv>1$; equality is a constant-mixing limit. Its GIG embedding
has $p=-1/2$, $a=\lambda/m^2$, $b=\lambda$. The posterior is generally GIG
with order $-1/2-d/2$, rather than another inverse Gaussian.

### Inverse-gamma mixing

For $Y\sim\operatorname{InvGamma}(\alpha,\rho)$,
$\mathbb E[\log Y]=\log\rho-\psi_0(\alpha)$ and
$\mathbb E[Y^{-1}]=\alpha/\rho$. Hence

```{math}
\log\alpha-\psi_0(\alpha)=l+\log u,\qquad \rho=\alpha/u,
```

with Newton proposal

```{math}
\alpha_{t+1}=\alpha_t-
\frac{\log\alpha_t-\psi_0(\alpha_t)-l-\log u}
{1/\alpha_t-\psi_1(\alpha_t)}.
```

This is again a scalar solve, not a general closed-form shape estimate.
For $\gamma=0$ the posterior has $A=0$ and is inverse gamma with shape
$\alpha+d/2$; a finite posterior first moment requires $\alpha+d/2>1$
when that moment is needed. Symmetry-constrained updates can require fewer
statistics than the full variance–mean model.

| Marginal family | Mixing statistics to fit its free parameters | Mixing M-step |
| --- | --- | --- |
| General GH | $(\log y,y^{-1},y)$ | Three-parameter convex problem in natural coordinates |
| Variance gamma | $(\log y,y)$ | Scalar shape solve and rate recovery |
| NIG | $(y^{-1},y)$ | Closed form |
| Normal-inverse gamma | $(\log y,y^{-1})$ | Scalar shape solve and scale recovery |

The normal block still uses inverse and first posterior moments when both
location and mean coupling are free. Reducing the mixing-law statistics does
not automatically eliminate these normal-block requirements.

## Numerical considerations and implementation

Large $|p-d/2|$ or extreme $\sqrt{AB(x)}$ can overflow or underflow direct
Bessel evaluations. Ratios, log-space evaluation, and stable order derivatives
matter. The covariance update can suffer cancellation or rank deficiency;
determinant normalization does not repair its condition number. Penalties
change the optimization objective and need their own ascent analysis.

See [why not gradient descent](./normix-why-not-gradient-descent.md) for the
optimization comparison and [exponential-family core](./normix-exponential-family-core.md)
for the expectation-to-natural map. Runnable fitters remain in the
[upstream EM guide](https://xshi19.github.io/normix/user_guide/em_fitting.html)
and [package API](https://xshi19.github.io/normix/api/index.html).
Continue with [online EM](./normix-online-em.md),
[penalized shrinkage](./normix-shrinkage.md), and the
[mathematical EM framework](./normix-em-framework.md).
The [full EM design](https://xshi19.github.io/normix/design/em_framework.html)
and [Bessel/solver design](https://xshi19.github.io/normix/design/solvers_and_bessel.html)
remain upstream implementation references.

## Source and adaptation

Adapted from `xshi19/normix`, `docs/theory/em_algorithm.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/theory/em_algorithm.md)
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
