---
title: Online EM for exponential families
---

Online expectation–maximization replaces a full-data E-step by a running
average of posterior sufficient statistics. Cappé and Moulines develop the
general method in
[2009, *On-line expectation–maximization algorithm for latent data models*](https://doi.org/10.1111/j.1467-9868.2009.00698.x).
This note specializes the update to a full exponential family, derives the
upstream regret identity, and applies the same statistic order as
[batch EM for GH](./normix-em-algorithm.md).

Read the [exponential-family core](./normix-exponential-family-core.md) for
natural and expectation coordinates and [IG duality](./information-geometry-duality.md)
for Bregman divergences. Throughout the full-family derivation, assume a
regular minimal family, finite posterior statistics, and iterates in an
interior region where the expectation-to-natural map exists. Matrix statistic
blocks use their independent symmetric coordinates.

## Posterior statistics and the sequential update

For observed $X$ and hidden $Y$, write

```{math}
f(x,y;\theta)=h(x,y)\exp\{\langle\theta,t(x,y)\rangle-\psi(\theta)\},
\qquad \eta=\nabla\psi(\theta).
```

Let $\phi(\eta)=\langle\theta,\eta\rangle-\psi(\theta)$ be the Legendre
dual, so that $\theta=\nabla\phi(\eta)$. At observation $x_t$, compute

```{math}
\bar t_t=\mathbb E_{\theta_{t-1}}[t(X,Y)\mid X=x_t].
```

Starting from $\eta_0=\nabla\psi(\theta_0)$, update

```{math}
:label: online-em-update
\begin{aligned}
\eta_t&=(1-\rho_t)\eta_{t-1}+\rho_t\bar t_t,\\
\theta_t&=\nabla\phi(\eta_t).
\end{aligned}
```

The reciprocal schedule $\rho_t=1/(\tau_0+t)$, with $\tau_0\geq0$, has
the exact aggregation identity

```{math}
:label: normix-online-aggregation
\eta_T=\frac{\tau_0\eta_0+\sum_{t=1}^T\bar t_t}{\tau_0+T}.
```

Here $\rho_t$ is the step size; $\tau_t=\tau_0+t$ is its reciprocal.
The initial statistic has weight $\tau_0$ in pseudo-observation units.
Each $\bar t_t$ uses the parameter available when that observation arrives;
it is not recomputed under $\theta_T$. Thus this average differs from a
batch E-step at the final parameter. With $\tau_0=0$, the first update can
lie on the boundary, so the assumed inverse map needs particular care.

For a mini-batch, replace $\bar t_t$ by its within-batch average. A constant
$\rho_t$ gives exponential forgetting. Decreasing schedules satisfying
$\sum_t\rho_t=\infty$ and $\sum_t\rho_t^2<\infty$ are common in stationary
stochastic approximation; for example $(t+t_0)^{-\kappa}$ with
$1/2<\kappa\leq1$. These conditions alone do not prove convergence:
stability, regularity, and a suitable M-step are also needed. The method's
cost per update uses fewer observations than batch EM, but this does not
imply the same convergence rate or likelihood ascent at each observation.

## A Bregman regret identity

Define the complete-data divergence, with its orientation explicit,

```{math}
D_\psi(\theta\Vert\theta_0)
=\psi(\theta)-\psi(\theta_0)
-\langle\eta_0,\theta-\theta_0\rangle
=D_{\mathrm{KL}}(f_{\theta_0}(X,Y)\Vert f_\theta(X,Y)).
```

For a sequence $x_1,\ldots,x_T$, compare sequential prediction with a fixed
parameter chosen using all observations and this initial penalty:

```{math}
:label: regret-def
\begin{aligned}
r_T&=-\sum_{t=1}^T\log f_X(x_t;\theta_{t-1})\\
&\quad-\min_\theta\left\{-\sum_{t=1}^T\log f_X(x_t;\theta)
+\tau_0D_\psi(\theta\Vert\theta_0)\right\}.
\end{aligned}
```

Suppose this minimum is finite and attained at an interior $\theta_*$, and
put $\eta_*=\nabla\psi(\theta_*)$. For the reciprocal schedule above,
the upstream decomposition is

```{math}
:label: regret-decomp
\begin{aligned}
r_T&=\sum_{t=1}^T\tau_tD_\psi(\theta_{t-1}\Vert\theta_t)\\
&\quad+\sum_{t=1}^TD_{\mathrm{KL}}\!\left(
f_{\theta_{t-1}}(Y\mid x_t)\Vert f_{\theta_*}(Y\mid x_t)\right)\\
&\quad-\tau_TD_\phi(\eta_T\Vert\eta_*).
\end{aligned}
```

This is an identity, not by itself a sublinear regret bound. The posterior
KL sum is nonnegative and need not be small. It disappears for fully
observed data. The penalty belongs to the comparator in [](#regret-def);
the sequential algorithm need not find its minimizer.

To verify the identity, let $g_t(\theta)=\psi(\theta)-\langle\theta,\bar t_t\rangle$.
Parameter-independent carrier terms cancel in likelihood differences.
The update gives $\bar t_t=\tau_t\eta_t-\tau_{t-1}\eta_{t-1}$, and
Legendre duality gives

```{math}
\begin{aligned}
\sum_tg_t(\theta_{t-1})
&=\sum_t\tau_tD_\phi(\eta_t\Vert\eta_{t-1})
+\tau_0\phi(\eta_0)-\tau_T\phi(\eta_T),\\
\sum_tg_t(\theta_*)+\tau_0D_\psi(\theta_*\Vert\theta_0)
&=\tau_0\phi(\eta_0)-\tau_T\phi(\eta_T)
+\tau_TD_\phi(\eta_T\Vert\eta_*).
\end{aligned}
```

Subtract these expressions. The observed-versus-complete likelihood identity
adds the posterior KL sum, while
$D_\phi(\eta_t\Vert\eta_{t-1})=D_\psi(\theta_{t-1}\Vert\theta_t)$
supplies [](#regret-decomp). The calculation is algebraic and requires all
displayed expectations and divergences to be finite.

## Application to generalized hyperbolic mixtures

Use the [GH model](./normix-generalized-hyperbolic.md)

```{math}
X=\mu+\gamma Y+\sqrt Y\,LZ,\qquad
Y\sim\operatorname{GIG}(p,a,b),\quad Z\sim\mathcal N_d(0,I_d),
\quad Y\perp Z,\quad LL^\top=\Sigma.
```

The classical tuple is $\vartheta=(\mu,\gamma,\Sigma,p,a,b)$; $\theta$
denotes natural coordinates. In the scalar
[conditioning note](./normix-conditioning-a-mixture.md), $W=Y$,
$\beta=\gamma$, and $\sigma^2=\Sigma$.
For $a,b>0$ and positive-definite $\Sigma$, the posterior is GIG and its
[power moments](./normix-em-algorithm.md#gig-moment-cond) are finite.
Set $u_t=\mathbb E[Y^{-1}\mid x_t]$, $v_t=\mathbb E[Y\mid x_t]$, and
$l_t=\mathbb E[\log Y\mid x_t]$, evaluated at $\vartheta_{t-1}$.
In the Batch 1 statistic order,

```{math}
\bar t_t=(l_t,u_t,v_t,x_t,x_tu_t,x_tx_t^\top u_t),\qquad
\eta_t=(1-\rho_t)\eta_{t-1}+\rho_t\bar t_t.
```

The first three slots are $(\log Y,Y^{-1},Y)$, a permutation of the source
online note's $(Y^{-1},Y,\log Y)$. The normal parameters are recovered as

```{math}
\begin{aligned}
\mu_t&=\frac{\eta_{4,t}-\eta_{3,t}\eta_{5,t}}
{1-\eta_{2,t}\eta_{3,t}},\\
\gamma_t&=\frac{\eta_{5,t}-\eta_{2,t}\eta_{4,t}}
{1-\eta_{2,t}\eta_{3,t}},\\
\Sigma_t&=\eta_{6,t}-\eta_{5,t}\mu_t^\top-\mu_t\eta_{5,t}^\top
+\eta_{2,t}\mu_t\mu_t^\top-\eta_{3,t}\gamma_t\gamma_t^\top.
\end{aligned}
```

The mixing-law update maximizes
$L_{\mathrm{GIG}}(p,a,b\mid\eta_{1,t},\eta_{2,t},\eta_{3,t})$.
These are the [batch M-step formulas](./normix-em-algorithm.md#m-step)
applied to running statistics. Their feasibility, nonzero-denominator, and
positive-definiteness requirements still apply; boundary mixing laws need
their own moment checks. A [shrinkage target](./normix-shrinkage.md) can
regularize the statistics, but changes the update and its objective.

## Curved families need a constrained M-step

For $f(x,y;u)=h(x,y)\exp\{\langle\theta(u),t(x,y)\rangle-\psi(\theta(u))\}$,
the ambient expectation vector need not equal the statistics being averaged.
The appropriate map is

```{math}
u_t\in\operatorname*{arg\,max}_u
\{\langle\theta(u),\eta_t\rangle-\psi(\theta(u))\}.
```

At an interior stationary point it satisfies

```{math}
D\theta(u_t)^\top[\eta_t-\nabla\psi(\theta(u_t))]=0.
```

This is weaker than ambient moment matching. Applying $\nabla\phi$ without
the constraint can leave the curved family. This limitation concerns that
unconstrained inverse map: it does **not** exclude online EM for curved
families. The Cappé–Moulines construction uses a model-specific maximizing
map. Its convergence assumptions must be checked for the particular model;
the full-family regret derivation above does not transfer automatically.

[GH factor analysis](./normix-factor-analysis.md) gives a concrete constrained
example. The [EM update framework](./normix-em-framework.md) separates
statistic aggregation, update rules, and parameter recovery. Numerical
backends and runnable streaming examples stay in the
[upstream EM framework](https://xshi19.github.io/normix/design/em_framework.html).

## Source and adaptation

Adapted from `xshi19/normix`, `docs/theory/online_em.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/theory/online_em.md)
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
