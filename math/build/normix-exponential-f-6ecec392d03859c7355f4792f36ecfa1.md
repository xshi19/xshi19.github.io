---
title: Exponential-family core
---

A log-partition function connects three descriptions of an exponential family:
classical distribution parameters, natural coordinates, and expected sufficient
statistics. This note develops that connection for normal and GIG laws, then
explains its role in [GH estimation](./normix-em-algorithm.md).
The [IG entry on exponential families](./information-geometry-exponential-families.md)
provides the foundational definitions and regularity assumptions.

## One potential, two derivative identities

For a fixed carrier $h$ and statistic $t$, write

```{math}
:label: normix-ef-core
f_\theta(x)=h(x)e^{\langle\theta,t(x)\rangle-\psi(\theta)},\qquad
\psi(\theta)=\log\int h(x)e^{\langle\theta,t(x)\rangle}\,dx.
```

Work on the nonempty open interior $\Theta$ of the finite natural-parameter
domain. Under these conditions differentiation of the normalizer gives

```{math}
:label: normix-ef-triad
\eta=\nabla\psi(\theta)=\mathbb E_\theta[t(X)],\qquad
I_\theta=\nabla^2\psi(\theta)=\operatorname{Cov}_\theta(t(X)).
```

The score is $t(X)-\eta$. Thus density normalization, expectation coordinates,
and Fisher information come from the same potential. Minimality makes
$I_\theta$ positive definite; a redundant statistic vector instead gives a
singular covariance. These identities do not claim that every distribution,
or every marginal of an exponential family, is itself such a family.

Classical parameters such as $(p,a,b)$ or $(\mu,\Sigma)$ express features of
the model directly. Their map to natural coordinates is family-specific.
Expectation coordinates express moments; their map back to the model depends
on whether those moments lie in $\nabla\psi(\Theta)$.

## Likelihood and Bregman inversion

For independent observations, reduce the data to
$\widehat\eta=n^{-1}\sum_i t(x_i)$. Maximizing average log likelihood is
then equivalent to minimizing

```{math}
:label: normix-ef-inversion
F(\theta)=\psi(\theta)-\langle\theta,\widehat\eta\rangle,
\qquad \theta\in\Theta.
```

If an interior solution $\widehat\theta$ exists, its gradient equation is
$\nabla\psi(\widehat\theta)=\widehat\eta$. For a minimal family it is the
unique solution. Define the Bregman divergence

```{math}
D_\psi(\theta\Vert\theta')
=\psi(\theta)-\psi(\theta')
-\langle\nabla\psi(\theta'),\theta-\theta'\rangle.
```

With $\theta'=\widehat\theta$,
$F(\theta)-F(\widehat\theta)=D_\psi(\theta\Vert\widehat\theta)$.
This explains the term *Bregman inversion*. It does not guarantee existence:
boundary empirical moments can place the likelihood supremum outside the
finite natural-parameter domain.

Where the dual map exists, the Legendre conjugate obeys

```{math}
\psi^*(\eta)=\langle\theta,\eta\rangle-\psi(\theta),\qquad
\nabla\psi^*(\eta)=\theta,\qquad
\nabla^2\psi^*(\eta)=I_\theta^{-1}.
```

The [duality entry](./information-geometry-duality.md) develops the geometric
interpretation and its domain limits. For a constrained subfamily
$\theta=\theta(\phi)$, stationarity is instead
$J_\theta(\phi)^\top(\nabla\psi-\widehat\eta)=0$; matching all ambient
moments is usually too strong.

## Multivariate normal example

For $X\sim\mathcal N_d(\mu,\Sigma)$ with precision $\Lambda=\Sigma^{-1}$,
use $t(x)=(x,xx^\top)$ and the vector/trace inner product. Then

```{math}
\theta_1=\Lambda\mu,\qquad \theta_2=-\tfrac12\Lambda,\qquad h(x)=1,
```

and completing the square gives

```{math}
\psi(\theta)=\tfrac12\theta_1^\top\Lambda^{-1}\theta_1
-\tfrac12\log|\Lambda|+\tfrac d2\log(2\pi).
```

The expectation coordinates and inverse map are

```{math}
\eta_1=\mu,\quad \eta_2=\Sigma+\mu\mu^\top,\qquad
\mu=\eta_1,\quad \Sigma=\eta_2-\eta_1\eta_1^\top\succ0.
```

This inversion is analytical. Symmetric matrix statistics have only
$d(d+1)/2$ independent entries; off-diagonal duplicates should not be treated
as independent natural coordinates when discussing minimality or Fisher rank.

## GIG example

For the [GIG density](./normix-generalized-inverse-gaussian.md),

```{math}
t(y)=(\log y,y^{-1},y)^\top,\qquad
\theta=(p-1,-b/2,-a/2)^\top,
```

and

```{math}
\psi(\theta)=\log2+\log K_p(\sqrt{ab})+\tfrac p2\log(b/a).
```

Its gradient is the log, inverse, and first moment of $Y$; its Hessian is their
covariance matrix. The inversion is a three-parameter convex problem in
natural coordinates after a single reduction of the data. Evaluation of the
potential and its derivatives still involves Bessel functions. Strict
convexity does not imply a well-conditioned Hessian near a degenerating law.

Reparameterizing constrained coordinates can help maintain feasibility but
can alter convexity. If $\theta=\theta(\phi)$ with Jacobian $J$, then

```{math}
\nabla_\phi^2F
=J^\top(\nabla_\theta^2F)J
+\sum_j(\partial_{\theta_j}F)\nabla_\phi^2\theta_j.
```

The extra term vanishes at a stationary point, but matters during optimization.
This is one reason the [gradient-descent comparison](./normix-why-not-gradient-descent.md)
must distinguish the objective from the coordinates and numerical method.

## From complete data to EM

For a joint exponential family $f_\theta(x,y)$, replace the empirical statistic
by the posterior average

```{math}
\widehat\eta_k=\frac1n\sum_i
\mathbb E_{\theta_k}[t(x_i,Y)\mid X=x_i].
```

The M-step again minimizes [](#normix-ef-inversion), now with
$\widehat\eta=\widehat\eta_k$. For GH, the normal block can be recovered
analytically and the GIG block by moment inversion. The
[mixture architecture](./normix-mixture-architecture.md) describes why the
complete-data and observed-data laws play different roles; the
[EM note](./normix-em-algorithm.md) supplies the actual posterior formulas.

This adaptation retains the mathematical potential and coordinate maps from
the source design note. Class layouts, backend choices, and solver recipes
remain in the [upstream design](https://xshi19.github.io/normix/design/exponential_family.html),
[solver discussion](https://xshi19.github.io/normix/design/solvers_and_bessel.html),
and [package API](https://xshi19.github.io/normix/api/index.html).

## Source and adaptation

Rewritten as a mathematical note from `xshi19/normix`, `docs/design/exponential_family.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/design/exponential_family.md)
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
