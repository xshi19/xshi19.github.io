---
title: Factor analysis for generalized hyperbolic distributions
---

Factor analysis replaces a full dispersion matrix by
$\Sigma=FF^\top+D$, where $F\in\mathbb R^{d\times r}$, $r<d$, and
$D$ is a positive-definite diagonal matrix. This reduces the covariance
parameter count and gives a latent-factor representation. Related mixture
models are studied by Tortora, McNicholas, and Browne in
[2013, *A Mixture of Generalized Hyperbolic Factor Analyzers*](https://arxiv.org/abs/1311.6530).
Here we derive the single-component GH factor model used in the pinned note.

Read [GH distributions](./normix-generalized-hyperbolic.md),
[batch EM](./normix-em-algorithm.md), and the
[joint/marginal distinction](./normix-mixture-architecture.md) first.
Assume $a,b>0$, independent observations, and the representation

```{math}
\begin{aligned}
X&=\mu+\gamma Y+\sqrt Y(FZ+\varepsilon),\\
Y&\sim\operatorname{GIG}(p,a,b),\quad
Z\sim\mathcal N_r(0,I_r),\quad \varepsilon\sim\mathcal N_d(0,D),
\end{aligned}
```

with $Y,Z,\varepsilon$ mutually independent. $Z$ denotes standard-normal
factor noise; the scalar mixing variable $Y$ is $W$ in the
[conditioning note](./normix-conditioning-a-mixture.md). In one dimension,
$\gamma=\beta$ and $\Sigma=\sigma^2$. The classical parameters are
$u=(\mu,\gamma,F,D,p,a,b)$; $\theta(u)$ below denotes their natural-coordinate
embedding.

## Joint Distribution

The conditional distribution of $X$ given $Y$ and $Z$
is $N(\mu + \gamma Y + \sqrt{Y} F Z, \, D Y)$. The joint distribution
of $(X, Y, Z)$ is:

```{math}
:label: fa-joint

\begin{aligned}
&f(x, y, z | \mu, \gamma, F, D, p, a, b) \\
&= \frac{1}{\sqrt{(2\pi)^{d+r} |D|}}
\frac{(a/b)^{p/2}}{2 K_p(\sqrt{ab})}
y^{p - 1 - d/2} \\
&\quad \times \exp\!\left(-\frac{1}{2} z^\top z
- \frac{1}{2}(b \, y^{-1} + a \, y)
- \frac{1}{2}(x - \mu - \gamma y - F \sqrt{y} \, z)^\top
D^{-1}(x - \mu - \gamma y - F \sqrt{y} \, z) \, y^{-1}\right)
\end{aligned}
```

for $y > 0$.

## Curved Exponential Family

The density [](#fa-joint) belongs to a **curved exponential family**:

```{math}
f(x, y, z) = h(x, y, z) \exp\!\left(\theta(u)^\top t(x, y, z)
- \psi(\theta(u))\right),
```

where $\theta(\cdot)$ is a nonlinear mapping from the parameter
$u = (\mu, \gamma, F, D, p, a, b)$ to a higher-dimensional space.

The sufficient statistics $t(x, y, z)$ consist of ten components:

```{math}
s_1 = \log y, \quad s_2 = y^{-1}, \quad s_3 = y, \quad
s_4 = x, \quad s_5 = x y^{-1}, \quad s_6 = x x^\top y^{-1},
```

```{math}
s_7 = x z^\top y^{-1/2}, \quad s_8 = z y^{-1/2}, \quad
s_9 = z y^{1/2}, \quad s_{10} = z z^\top.
```

## Log-Likelihood Function

With $s$ fixed as a complete-data average or a posterior average, the
parameter-dependent part of the per-observation log likelihood is:

```{math}
\begin{aligned}
L_{FA}(u | s) &= -\frac{1}{2} \log|D|
- \frac{1}{2} \mu^\top D^{-1} \mu \, s_2
- \frac{1}{2} \gamma^\top D^{-1} \gamma \, s_3
+ \gamma^\top D^{-1} s_4
+ \mu^\top D^{-1} s_5 \\
&\quad - \frac{1}{2} \operatorname{tr}(D^{-1} s_6)
+ \operatorname{tr}(F^\top D^{-1} s_7)
- \mu^\top D^{-1} F s_8
- \gamma^\top D^{-1} F s_9 \\
&\quad - \frac{1}{2} \operatorname{tr}(F^\top D^{-1} F s_{10})
- \mu^\top D^{-1} \gamma
+ L_{GIG}(p, a, b | s_1, s_2, s_3),
\end{aligned}
```

where $L_{GIG}$ is the [GIG log likelihood](./normix-generalized-inverse-gaussian.md#gig-loglik),
using the order $(\log Y,Y^{-1},Y)$ throughout. The fixed standard-normal
term $-\tfrac12\operatorname{tr}(s_{10})$ and parameter-independent
normalizing constants are omitted from $L_{FA}$.

## M-step for the normal block

Assume $s_{10}$ is invertible and the two location/skew equations below
have a nonsingular coefficient matrix. Eliminating $F$ from the normal
equations gives the following auxiliary quantities:

```{math}
:label: fa-aux

\begin{aligned}
q_1 &= s_8^\top s_{10}^{-1} s_8 - s_2, \\
q_2 &= s_9^\top s_{10}^{-1} s_8 - 1, \\
q_3 &= s_9^\top s_{10}^{-1} s_9 - s_3, \\
q_4 &= s_7 s_{10}^{-1} s_8 - s_5, \\
q_5 &= s_7 s_{10}^{-1} s_9 - s_4.
\end{aligned}
```

Here $s_7$ is $d\times r$, so $q_4,q_5$ are $d$-vectors.
The source transpose on $s_7$ in these two expressions is corrected to
make the products dimensionally consistent. The parameter updates are:

```{math}
:label: fa-mstep

\begin{aligned}
\mu &= \frac{q_2 \, q_5 - q_3 \, q_4}{q_2^2 - q_1 \, q_3}, \\
\gamma &= \frac{q_2 \, q_4 - q_1 \, q_5}{q_2^2 - q_1 \, q_3}, \\
F &= (s_7 - \mu \, s_8^\top - \gamma \, s_9^\top) \, s_{10}^{-1}, \\
D &= \operatorname{diag}\!\big(
s_2 \mu \mu^\top + s_3 \gamma \gamma^\top
- s_4 \gamma^\top - \gamma s_4^\top
- s_5 \mu^\top - \mu s_5^\top + s_6 \\
&\qquad - s_7 F^\top - F s_7^\top
+ F s_8 \mu^\top + \mu (F s_8)^\top
+ F s_9 \gamma^\top + \gamma (F s_9)^\top \\
&\qquad + F s_{10} F^\top
+ \mu \gamma^\top + \gamma \mu^\top\big), \\
(p, a, b) &= \arg\max_{p, a, b} L_{GIG}(p, a, b | s_1, s_2, s_3).
\end{aligned}
```

## Conditional Expectations for the E-Step

Integrating [](#fa-joint) over $z$ gives the ordinary GH joint law of
$(X,Y)$ with dispersion $\Sigma=FF^\top+D$ and conditional covariance
$\operatorname{Cov}(X\mid Y)=Y\Sigma$.
Therefore the conditional distribution of $Y$ given $X$ is:

```{math}
Y | X = x \sim \operatorname{GIG}\!\left(p - \frac{d}{2}, \,
a + \gamma^\top (F F^\top + D)^{-1} \gamma, \,
b + (x - \mu)^\top (F F^\top + D)^{-1} (x - \mu)\right),
```

and the conditional moments $E[Y^\alpha | X, u]$ and
$E[\log Y | X, u]$ are computed using [](./normix-em-algorithm.md#gig-moment-cond).

The conditional distribution of $(X, Z)$ given $Y$ is Gaussian:

```{math}
\begin{aligned}
\begin{pmatrix} (X - \mu - \gamma Y)/\sqrt{Y} \\ Z \end{pmatrix}
\Bigg| Y, u
\sim N\!\left(0, \begin{pmatrix}
F F^\top + D & F \\ F^\top & I
\end{pmatrix}\right).
\end{aligned}
```

Define $A = F^\top (F F^\top + D)^{-1}$. The conditional
expectations of the latent factor $Z$ are:

```{math}
\begin{aligned}
E[Z Y^{-1/2} | X, u] &= A(X - \mu) E[Y^{-1} | X, u] - A \gamma, \\
E[Z Y^{1/2} | X, u] &= A(X - \mu) - A \gamma \, E[Y | X, u], \\
E[Z Z^\top | X, u] &= I - A F
+ A(X - \mu)(X - \mu)^\top A^\top E[Y^{-1} | X, u] \\
&\quad - A(X - \mu)\gamma^\top A^\top
- A \gamma (X - \mu)^\top A^\top
+ A \gamma \gamma^\top A^\top E[Y | X, u].
\end{aligned}
```

## E-Step

Given i.i.d. samples $x_1, \ldots, x_n$ and current parameters
$u_k = (\mu_k, \gamma_k, F_k, D_k, p_k, a_k, b_k)$, the E-step
computes all ten sufficient statistics. The first six are the same as the
standard EM algorithm (see [](./normix-em-algorithm.md#e-step)):

```{math}
\begin{aligned}
s_1^{(k)} &= \frac{1}{n} \sum_{j=1}^n E[\log Y | X = x_j, u_k], \\
s_2^{(k)} &= \frac{1}{n} \sum_{j=1}^n E[Y^{-1} | X = x_j, u_k], \\
s_3^{(k)} &= \frac{1}{n} \sum_{j=1}^n E[Y | X = x_j, u_k], \\
s_4^{(k)} &= \frac{1}{n} \sum_{j=1}^n x_j, \\
s_5^{(k)} &= \frac{1}{n} \sum_{j=1}^n x_j \, E[Y^{-1} | X = x_j, u_k], \\
s_6^{(k)} &= \frac{1}{n} \sum_{j=1}^n
x_j x_j^\top E[Y^{-1} | X = x_j, u_k].
\end{aligned}
```

The remaining four are determined by the first six using
$A_k = F_k^\top (F_k F_k^\top + D_k)^{-1}$:

```{math}
\begin{aligned}
s_7^{(k)} &= (s_6^{(k)} - s_5^{(k)} \mu_k^\top
- s_4^{(k)} \gamma_k^\top) A_k^\top, \\
s_8^{(k)} &= A_k (s_5^{(k)} - \mu_k \, s_2^{(k)} - \gamma_k), \\
s_9^{(k)} &= A_k (s_4^{(k)} - \mu_k - \gamma_k \, s_3^{(k)}), \\
s_{10}^{(k)} &= I - A_k F_k
+ A_k \big(s_6^{(k)} - s_5^{(k)} \mu_k^\top
- \mu_k (s_5^{(k)})^\top + \mu_k \mu_k^\top s_2^{(k)} \\
&\quad - (s_4^{(k)} - \mu_k) \gamma_k^\top
- \gamma_k (s_4^{(k)} - \mu_k)^\top
+ \gamma_k \gamma_k^\top s_3^{(k)}\big) A_k^\top.
\end{aligned}
```

The M-step then applies [](#fa-mstep) and [](#fa-aux) with
$s = (s_1^{(k)}, \ldots, s_{10}^{(k)})$.

## Feasibility and interpretation

The first six blocks agree with the Batch 1 order
$(\log Y,Y^{-1},Y,X,X/Y,XX^\top/Y)$; the source factor note puts the first
three in a different order. The remaining blocks record posterior factor
cross moments. Matrix pairings use traces, or equivalently independent
symmetric coordinates, as in the
[exponential-family core](./normix-exponential-family-core.md).

The displayed normal-block updates solve a weighted regression with diagonal
residual dispersion. They require a nonsingular regressor moment matrix and
strictly positive updated residual variances. Zero diagonal entries put the
optimum on a boundary; blindly inverting that update is invalid. The GIG
block remains a constrained numerical maximization, so the full M-step is
not generally closed form. Under exact posteriors and an improving M-step,
the usual [EM ascent argument](./information-geometry-latent-variables-em.md)
applies to the observed likelihood.

$FF^\top+D$ is positive definite when $D$ is, but factor structure alone
does not give a uniform condition-number bound. Orthogonal rotations
$F\mapsto FO$, $O^\top O=I_r$, leave $\Sigma$ unchanged; individual loading
columns are therefore not identified without further conventions. The GH
mixing-scale ambiguity also remains. Neither ambiguity establishes any new
Fisher-curvature or research-gauge formula.

This is a curved family: fitting its normal block is a constrained
maximization, not inversion of arbitrary ambient moments. Running sufficient
statistics can still be used with a suitable constrained M-step; see
[online EM for curved families](./normix-online-em.md#curved-families-need-a-constrained-m-step).
[Penalized shrinkage](./normix-shrinkage.md) is a separate way to regularize
estimation. The [mathematical EM framework](./normix-em-framework.md)
explains how aggregation and parameter recovery fit together; package
interfaces remain in the
[upstream design](https://xshi19.github.io/normix/design/em_framework.html).

## Source and adaptation

Adapted from `xshi19/normix`, `docs/theory/factor_analysis.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/theory/factor_analysis.md)
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
