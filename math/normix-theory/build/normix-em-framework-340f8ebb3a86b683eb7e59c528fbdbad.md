---
title: EM as sufficient-statistic updates
---

EM separates posterior inference from maximization. A further separation is
useful when observations arrive in batches or when estimation is regularized:
aggregate posterior statistics, transform that aggregate, then recover model
parameters. This mathematical extract of the upstream design develops general
and affine update rules and their composition with Bregman shrinkage.

The prerequisites are [batch EM](./normix-em-algorithm.md),
[exponential-family coordinates](./normix-exponential-family-core.md), and
[mixture architecture](./normix-mixture-architecture.md). The
[full upstream framework](https://xshi19.github.io/normix/design/em_framework.html)
owns the software interfaces, fitting loops, and product options.

## From a batch to a model

For a complete-data exponential family with statistic $t(X,Y)$, observation
batch $\mathcal B_t$, and current classical parameter $u_{t-1}$, compute

```{math}
:label: normix-framework-aggregation
\widehat\eta_t=\frac1{|\mathcal B_t|}\sum_{x\in\mathcal B_t}
\mathbb E_{u_{t-1}}[t(X,Y)\mid X=x].
```

The aggregate contains only the statistic blocks needed by the M-step.
The [GH convention](./normix-em-algorithm.md#e-step) orders them as
$t=(\log Y,Y^{-1},Y,X,X/Y,XX^\top/Y)$.
$Y$ is the mixing variable $W$ in the
[conditioning note](./normix-conditioning-a-mixture.md); $Z$ remains
standard-normal noise. [Factor analysis](./normix-factor-analysis.md)
adds latent-factor cross moments to these six blocks.

After an update rule produces a statistic vector $s$, recover parameters by

```{math}
:label: normix-framework-recovery
\mathcal M(s)\in\operatorname*{arg\,max}_u
\{\langle\theta(u),s\rangle-\psi(\theta(u))\}.
```

Here $\theta(u)$ is the natural-coordinate embedding. In an unconstrained
minimal family with an attained interior solution,
$\theta(\mathcal M(s))=\nabla\phi(s)$. In a curved family,
$\mathcal M$ solves the constrained problem; $s$ is not generally the
expectation vector of the recovered model. We use the name $\eta$ for the
running statistic in the update rules, without asserting ambient moment
matching in that case.

## Two layers of update rules

The general layer is a map with optional state $q_t$:

```{math}
:label: normix-framework-rule
(\eta_t,q_t)=\mathcal U_t(\eta_{t-1},\widehat\eta_t,q_{t-1}).
```

State can record a cumulative observation count or a running schedule.
It is distinct from the sufficient-statistic vector and from the model
parameters. A specialization is an affine rule

```{math}
:label: normix-framework-affine
\eta_t=a_t+B_t\eta_{t-1}+C_t\widehat\eta_t,
```

where $a_t$ is a statistic-space offset and $B_t,C_t$ are linear operators.
Scalar weights multiply every block;
block-diagonal operators can weight different statistics separately.

| Algorithmic scheme | $a_t$ | $B_t$ | $C_t$ |
| --- | --- | --- | --- |
| Ordinary batch EM | $0$ | $0$ | $I$ |
| Exponential forgetting | $0$ | $(1-\rho)I$ | $\rho I$ |
| Decreasing online update | $0$ | $(1-\rho_t)I$ | $\rho_t I$ |
| Sample-count weighting | $0$ | $n/(n+m)I$ | $m/(n+m)I$ |

In the last row, $n$ is the weight already accumulated and $m$ is the new
batch size; the count state becomes $n+m$. The reciprocal schedule
$\rho_t=1/(\tau_0+t)$ recovers single-observation weighting with initial
weight $\tau_0$. More general decreasing schedules and their assumptions
are discussed in [online EM](./normix-online-em.md).

A scalar convex combination of feasible statistics preserves their convex
moment constraints. Arbitrary offsets, operators, or independent block
weights need not preserve positive-definiteness, moment inequalities, or
existence of an interior maximizing map. The general update notation alone
provides no likelihood-ascent or convergence theorem.

## Scalar shrinkage as a Bregman penalty

Let $\eta_0=\nabla\psi(\theta_0)$ be a reference expectation vector and
$\tau\geq0$. The [penalized EM derivation](./normix-shrinkage.md) maximizes

```{math}
\langle\theta,\widehat\eta_t\rangle-\psi(\theta)
-\tau D_\psi(\theta\Vert\theta_0)
=\langle\theta,\widehat\eta_t+\tau\eta_0\rangle
-(1+\tau)\psi(\theta)+\text{constant}.
```

Dividing by $1+\tau$ makes it ordinary parameter recovery at
$S_\tau(\widehat\eta_t)$, where

```{math}
:label: normix-framework-shrinkage
S_\tau(v)=\frac{v+\tau\eta_0}{1+\tau}.
```

With a fresh full-data E-step and exact or improving penalized maximization,
this is EM for the observed likelihood minus the fixed complete-data KL
penalty. In [IG dual coordinates](https://xshi19.github.io/math/ig/information-geometry-duality/), the
penalty is $D_\phi(\eta_0\Vert\eta)$; it is not generally a squared
Euclidean distance in expectation coordinates.

## Composing shrinkage with a running rule

Apply the base rule first, then shrink its statistic output:

```{math}
:label: normix-framework-composition
\begin{aligned}
(v_t,q_t)&=\mathcal U_t(\eta_{t-1},\widehat\eta_t,q_{t-1}),\\
\eta_t&=S_\tau(v_t),\qquad u_t=\mathcal M(\eta_t).
\end{aligned}
```

For an affine base, the composed coefficients are explicit:

```{math}
\eta_t=\frac{a_t+\tau\eta_0}{1+\tau}
+\frac{B_t}{1+\tau}\eta_{t-1}
+\frac{C_t}{1+\tau}\widehat\eta_t.
```

Thus one composition handles ordinary EM, forgetting, and sample-weighted
updates. If the base is nonlinear, $S_\tau$ is still affine in its own
argument, but the complete composed rule need not be affine in its inputs.
The base rule's auxiliary state passes through this construction.

The interpretation depends on the base. With ordinary batch EM, the penalty
proof applies to the current observed-data objective. With running
statistics, the same algebra describes a regularized surrogate, but does not
establish ascent of a fixed observed-data likelihood.

Composition order also matters. Shrinking a running average gives

```{math}
\eta_t=\frac{(1-\rho_t)\eta_{t-1}+\rho_t\widehat\eta_t+\tau\eta_0}
{1+\tau}.
```

Updating toward a shrunk *new* aggregate instead gives

```{math}
\eta_t=(1-\rho_t)\eta_{t-1}
+\rho_t\frac{\widehat\eta_t+\tau\eta_0}{1+\tau}.
```

The reference weights are respectively $\tau/(1+\tau)$ and
$\rho_t\tau/(1+\tau)$. Repeated shrinkage of the whole running statistic
can dominate a vanishing data step: for fixed $\tau>0$, bounded new
statistics, and $\rho_t\to0$, the first recurrence tends to $\eta_0$.
These schemes therefore need separate choices of schedule and objective.

## Blockwise rules and algorithmic limits

For nonnegative block weights $\tau_j$, one can write

```{math}
\eta_{t,j}=\frac{v_{t,j}+\tau_j\eta_{0,j}}{1+\tau_j}.
```

This defines a blockwise statistic transformation. It is not generally
equivalent to the scalar joint-KL penalty, because $\psi$ couples statistic
blocks. A separable potential or another explicit penalty derivation would
be needed for such a claim. Moment feasibility must also be checked.
The [covariance-only example](./normix-shrinkage.md#what-the-covariance-penalty-does)
explains when changing the sixth GH block alone gives a direct dispersion
shrinkage formula.

Batch EM refreshes posterior statistics for all observations at each step.
Incremental or mini-batch schemes refresh only the selected observations,
retain a running aggregate, and apply $\mathcal M$ to that aggregate.
Their fixed points, numerical cost, and convergence depend on the update
schedule and on whether old posterior contributions are revisited.
Factor constraints, approximate solvers, and post-update transformations
must each be assessed against the actual optimization problem.

For numerical inversion and implementation details, see the upstream
[Bessel and solver design](https://xshi19.github.io/normix/design/solvers_and_bessel.html),
[full EM framework](https://xshi19.github.io/normix/design/em_framework.html),
and [package API](https://xshi19.github.io/normix/api/index.html).

## Source and adaptation

Rewritten from `xshi19/normix`, `docs/design/em_framework.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/design/em_framework.md)
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
