---
title: Shrinkage with penalized likelihood
---

A covariance estimate can be singular or poorly conditioned when observations
are few relative to dimension. Rescaling its determinant does not change its
condition number, and cannot repair a singular matrix. A penalty toward a
well-conditioned reference model instead changes the estimation objective.
The upstream derivation credits Shi (2016), *Generalized Hyperbolic
Distributions and Related Topics*, PhD thesis; see its
[bibliographic attribution](https://xshi19.github.io/normix/references.html#shi2016).

This note derives a complete-data KL penalty and its effect on the
[GH EM update](./normix-em-algorithm.md). Read the
[exponential-family core](./normix-exponential-family-core.md) and
[IG duality](./information-geometry-duality.md) for the coordinate maps and
divergence orientation.

## Penalized likelihood and the EM surrogate

Let $X$ be observed and $Y$ hidden in a regular exponential family

```{math}
f(x,y;\theta)=h(x,y)\exp\{\langle\theta,t(x,y)\rangle-\psi(\theta)\},
\qquad \eta=\nabla\psi(\theta).
```

Choose a reference parameter $\theta_0$ with finite expectation vector
$\eta_0$. Here the word *prior* means a target distribution in the model,
not necessarily a Bayesian prior over parameters. The joint-law divergence is

```{math}
:label: normix-shrinkage-kl
D_{\mathrm{KL}}(f_{\theta_0}(X,Y)\Vert f_\theta(X,Y))
=\psi(\theta)-\psi(\theta_0)-\langle\eta_0,\theta-\theta_0\rangle
=D_\psi(\theta\Vert\theta_0).
```

For independent observations $x_1,\ldots,x_n$, maximize

```{math}
:label: normix-shrinkage-objective
J(\theta)=\frac1n\sum_{j=1}^n\log f_X(x_j;\theta)
-\tau D_\psi(\theta\Vert\theta_0),\qquad \tau\geq0.
```

The likelihood is observed-data, while the penalty compares complete-data
laws on a fixed latent coordinate. This distinction matters for GH scale
nonidentifiability: an equivalent marginal representation can have a different
penalty if the reference model is held fixed.

At iterate $\theta_k$, define

```{math}
\begin{aligned}
\widehat\eta_k&=\frac1n\sum_j
\mathbb E_{\theta_k}[t(X,Y)\mid X=x_j],\\
Q_\tau(\theta\mid\theta_k)&=\frac1n\sum_j
\mathbb E_{\theta_k}[\log f(x_j,Y;\theta)\mid x_j]
-\tau D_\psi(\theta\Vert\theta_0).
\end{aligned}
```

The penalized EM identity is

```{math}
\begin{aligned}
J(\theta_{k+1})-J(\theta_k)
&=Q_\tau(\theta_{k+1}\mid\theta_k)-Q_\tau(\theta_k\mid\theta_k)\\
&\quad+\frac1n\sum_jD_{\mathrm{KL}}\!\left(
f_{\theta_k}(Y\mid x_j)\Vert f_{\theta_{k+1}}(Y\mid x_j)\right).
\end{aligned}
```

Consequently, an M-step that increases this surrogate increases $J$, whenever
these quantities are finite. It need not increase the unpenalized likelihood
or reach a global maximum.

## Shrunk sufficient statistics

Discarding terms independent of the candidate $\theta$, the M-step becomes

```{math}
:label: normix-shrinkage-mstep
\theta_{k+1}\in\operatorname*{arg\,max}_\theta
\{\langle\theta,\widehat\eta_k+\tau\eta_0\rangle
-(1+\tau)\psi(\theta)\}.
```

It is therefore the ordinary expectation-to-parameter problem at

```{math}
:label: normix-shrinkage-statistics
\widetilde\eta_k=\frac{\widehat\eta_k+\tau\eta_0}{1+\tau}.
```

For an attained interior maximum of a full minimal family,
$\nabla\psi(\theta_{k+1})=\widetilde\eta_k$.
For a curved family, such as [factor analysis](./normix-factor-analysis.md),
the same penalized-surrogate derivation holds with $\theta=\theta(u)$, but
the constrained maximizing map replaces full ambient moment matching.

Because $J$ uses an average likelihood, the reference has the weight of
$n_{\mathrm{prior}}=\tau n$ pseudo-observations. For a fixed prior sample
size as $n$ grows, use $\tau=n_{\mathrm{prior}}/n$. A fixed $\tau$ retains
a fixed proportion of shrinkage instead.

## A coherent GH target

Use $\vartheta_0=(\mu_0,\gamma_0,\Sigma_0,p_0,a_0,b_0)$ for the classical
reference parameters, reserving $\theta_0$ for natural coordinates.
The construction is $X=\mu_0+\gamma_0Y+\sqrt Y L_0Z$, with
$L_0L_0^\top=\Sigma_0$, $Z$ standard normal independent of $Y$, and
$Y\sim\operatorname{GIG}(p_0,a_0,b_0)$. The
[conditioning notation](./normix-conditioning-a-mixture.md) uses $W=Y$,
$\beta=\gamma$, and, in one dimension, $\sigma^2=\Sigma$.

For $a_0,b_0>0$, put $M_0(s)=\mathbb E_0[Y^s]$. The
[GIG moments](./normix-generalized-inverse-gaussian.md) give

```{math}
M_0(s)=\left(\frac{b_0}{a_0}\right)^{s/2}
\frac{K_{p_0+s}(\sqrt{a_0b_0})}{K_{p_0}(\sqrt{a_0b_0})},
\qquad l_0=M_0'(0),\quad u_0=M_0(-1),\quad v_0=M_0(1).
```

In the same order as the [batch E-step](./normix-em-algorithm.md#e-step),
the reference vector has six blocks:

```{math}
\begin{aligned}
\eta_{0,1}&=l_0,&\eta_{0,2}&=u_0,&\eta_{0,3}&=v_0,\\
\eta_{0,4}&=\mu_0+\gamma_0v_0,\\
\eta_{0,5}&=\mu_0u_0+\gamma_0,\\
\eta_{0,6}&=\Sigma_0+\mu_0\mu_0^\top u_0+\gamma_0\gamma_0^\top v_0
+\mu_0\gamma_0^\top+\gamma_0\mu_0^\top.
\end{aligned}
```

Apply [](#normix-shrinkage-statistics) to all six blocks and then use the
ordinary [GH M-step](./normix-em-algorithm.md#m-step). This is a convex
combination of whole statistic vectors, not separate fits to the six moments.
At gamma or inverse-gamma boundaries, check that every reference and posterior
moment needed by the chosen model is finite.

## What the covariance penalty does

The normal-block covariance is affine in the sixth block **when the first
five blocks are fixed**. Under uniform shrinkage, those other blocks also
change, moving the fitted location, skew vector, and mixing parameters.
Thus uniform joint-KL shrinkage is not generally just
$(\widehat\Sigma+\tau\Sigma_0)/(1+\tau)$.

To isolate the distinction, fix the first five statistics and let the sixth
block be $s_6$. Suppose its normal-block update gives dispersion $\Sigma$.
A compatible target sixth block is

```{math}
s_{6,0}=s_6-\Sigma+\Sigma_0.
```

Shrinking only this block yields

```{math}
\widetilde\Sigma=\frac{\Sigma+\tau\Sigma_0}{1+\tau}.
```

If $\Sigma$ is positive semidefinite, $\Sigma_0$ positive definite, and
$\tau>0$, then
$\lambda_{\min}(\widetilde\Sigma)\geq
\tau\lambda_{\min}(\Sigma_0)/(1+\tau)>0$.
This is a statement about a compatible normal-block update, not a proof that
arbitrary per-statistic weights correspond to one KL penalty. The latter can
also violate moment feasibility.

The [EM update framework](./normix-em-framework.md) derives composition with
running rules and separates scalar KL shrinkage from more general blockwise
regularization. [Online EM](./normix-online-em.md) discusses decreasing and
forgetting schedules. Implementation choices and supported targets remain in
the [full upstream design](https://xshi19.github.io/normix/design/em_framework.html).

## Source and adaptation

Adapted from `xshi19/normix`, `docs/theory/shrinkage.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/theory/shrinkage.md)
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
