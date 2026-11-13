---
title: Online EM for exponential families
---

Online expectation–maximization replaces a full-data E-step by a running
average of posterior sufficient statistics. @cappe2009online develop the
general method in *On-line expectation–maximization algorithm for latent data models*.
This note specializes the update to a full exponential family, derives the
upstream regret identity, and applies the same statistic order as
[batch EM for GH](./normix-em-algorithm.md).

Read the [exponential-family core](./normix-exponential-family-core.md) for
natural and expectation coordinates and [IG duality](https://xshi19.github.io/math/ig/information-geometry-duality/)
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

Here $f$ is the joint density and $f_X$ its observed marginal.
Let $\psi^*(\eta)=\langle\theta,\eta\rangle-\psi(\theta)$ be the Legendre
dual, so that $\theta=\nabla\psi^*(\eta)$. At observation $x_t$, compute

```{math}
\bar t_t=\mathbb E_{\theta_{t-1}}[t(X,Y)\mid X=x_t].
```

Starting from $\eta_0=\nabla\psi(\theta_0)$, update

```{math}
:label: online-em-update
\begin{aligned}
\eta_t&=(1-\rho_t)\eta_{t-1}+\rho_t\bar t_t,\\
\theta_t&=\nabla\psi^*(\eta_t).
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
=D_{\mathrm{KL}}(f(X,Y;\theta_0)\Vert f(X,Y;\theta)).
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
f(Y\mid x_t;\theta_{t-1})\Vert f(Y\mid x_t;\theta_*)\right)\\
&\quad-\tau_TD_{\psi^*}(\eta_T\Vert\eta_*).
\end{aligned}
```

This is an identity, not by itself a sublinear regret bound. The posterior
KL sum is nonnegative and need not be small. It disappears for fully
observed data. The penalty belongs to the comparator in [](#regret-def);
the sequential algorithm need not find its minimizer.

Here $f(Y\mid x_t;\theta)$ is the posterior density, and $D_{\psi^*}$
is the Bregman divergence of $\psi^*$ with the same argument order as $D_\psi$.
To verify the identity, let
$g_t(\theta)=\psi(\theta)-\langle\theta,\bar t_t\rangle$ be the
parameter-dependent negative complete-data objective for observation $t$.
Parameter-independent carrier terms cancel in likelihood differences.
The update gives $\bar t_t=\tau_t\eta_t-\tau_{t-1}\eta_{t-1}$, and
Legendre duality gives

```{math}
\begin{aligned}
\sum_tg_t(\theta_{t-1})
&=\sum_t\tau_tD_{\psi^*}(\eta_t\Vert\eta_{t-1})
+\tau_0\psi^*(\eta_0)-\tau_T\psi^*(\eta_T),\\
\sum_tg_t(\theta_*)+\tau_0D_\psi(\theta_*\Vert\theta_0)
&=\tau_0\psi^*(\eta_0)-\tau_T\psi^*(\eta_T)
+\tau_TD_{\psi^*}(\eta_T\Vert\eta_*).
\end{aligned}
```

Subtract these expressions. The observed-versus-complete likelihood identity
adds the posterior KL sum, while
$D_{\psi^*}(\eta_t\Vert\eta_{t-1})=D_\psi(\theta_{t-1}\Vert\theta_t)$
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
For this update, write $\mathbb E_{t-1}$ for expectation at
$\vartheta_{t-1}$ and retain the inverse-moment weight
$u_t=\mathbb E_{t-1}[Y^{-1}\mid X=x_t]$.
In the shared GH statistic order,

```{math}
\bar t_t=\bigl(\mathbb E_{t-1}[\log Y\mid X=x_t],u_t,
\mathbb E_{t-1}[Y\mid X=x_t],x_t,x_tu_t,x_tx_t^\top u_t\bigr).
```

The running update remains

```{math}
\eta_t=(1-\rho_t)\eta_{t-1}+\rho_t\bar t_t.
```

The first three slots are $(\log Y,Y^{-1},Y)$, a permutation of the source
online note's $(Y^{-1},Y,\log Y)$. Write $\eta_{t,i}$ for block $i$ of
$\eta_t$. The normal parameters are recovered as

```{math}
\begin{aligned}
\mu_t&=\frac{\eta_{t,4}-\eta_{t,3}\eta_{t,5}}
{1-\eta_{t,2}\eta_{t,3}},\\
\gamma_t&=\frac{\eta_{t,5}-\eta_{t,2}\eta_{t,4}}
{1-\eta_{t,2}\eta_{t,3}},\\
\Sigma_t&=\eta_{t,6}-\eta_{t,5}\mu_t^\top-\mu_t\eta_{t,5}^\top
+\eta_{t,2}\mu_t\mu_t^\top-\eta_{t,3}\gamma_t\gamma_t^\top.
\end{aligned}
```

The mixing-law update maximizes
$L_{\mathrm{GIG}}(p,a,b\mid\eta_{t,1},\eta_{t,2},\eta_{t,3})$.
These are the [batch M-step formulas](./normix-em-algorithm.md#m-step)
applied to running statistics. Their feasibility, nonzero-denominator, and
positive-definiteness requirements still apply; boundary mixing laws need
their own moment checks. A [shrinkage target](./normix-shrinkage.md) can
regularize the statistics, but changes the update and its objective.

## Curved families need a constrained M-step

For classical parameters $\vartheta$ and a curved family
$f(x,y;\vartheta)=h(x,y)\exp\{\langle\theta(\vartheta),t(x,y)\rangle-\psi(\theta(\vartheta))\}$,
the ambient expectation vector need not equal the statistics being averaged.
The appropriate map is

```{math}
\vartheta_t\in\operatorname*{arg\,max}_\vartheta
\{\langle\theta(\vartheta),\eta_t\rangle-\psi(\theta(\vartheta))\}.
```

At an interior stationary point it satisfies

```{math}
D\theta(\vartheta_t)^\top[\eta_t-\nabla\psi(\theta(\vartheta_t))]=0.
```

This is weaker than ambient moment matching. Applying $\nabla\psi^*$ without
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
