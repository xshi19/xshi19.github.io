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
$\vartheta=(\mu,\gamma,F,D,p,a,b)$; $\theta(\vartheta)$ below denotes their
natural-coordinate embedding.

## Joint Distribution

The conditional distribution of $X$ given $Y$ and $Z$
is $\mathcal N_d(\mu + \gamma Y + \sqrt{Y} F Z, \, D Y)$. The joint distribution
of $(X, Y, Z)$ is:

```{math}
:label: fa-joint

\begin{aligned}
&f(x, y, z;\vartheta) \\
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
f(x, y, z;\vartheta) = h(x, y, z) \exp\!\left(\langle\theta(\vartheta),t(x, y, z)\rangle
- \psi(\theta(\vartheta))\right),
```

where $\theta(\cdot)$ is a nonlinear mapping from the parameter
$\vartheta = (\mu, \gamma, F, D, p, a, b)$ to a higher-dimensional space.

The sufficient statistic $t(x,y,z)$ consists of ten blocks, with arguments
suppressed on each $t_i$ below:

```{math}
t_1 = \log y, \quad t_2 = y^{-1}, \quad t_3 = y, \quad
t_4 = x, \quad t_5 = x y^{-1}, \quad t_6 = x x^\top y^{-1},
```

```{math}
t_7 = x z^\top y^{-1/2}, \quad t_8 = z y^{-1/2}, \quad
t_9 = z y^{1/2}, \quad t_{10} = z z^\top.
```

## Log-Likelihood Function

Let $\widehat\eta$ be a complete-data average or a posterior average of
$t(X,Y,Z)$ in the same block order. Holding it fixed, the
parameter-dependent part of the per-observation log likelihood is:

```{math}
\begin{aligned}
L_{\mathrm{FA}}(\vartheta\mid\widehat\eta) &= -\frac{1}{2} \log|D|
- \frac{1}{2} \mu^\top D^{-1} \mu \, \widehat\eta_2
- \frac{1}{2} \gamma^\top D^{-1} \gamma \, \widehat\eta_3
+ \gamma^\top D^{-1} \widehat\eta_4
+ \mu^\top D^{-1} \widehat\eta_5 \\
&\quad - \frac{1}{2} \operatorname{tr}(D^{-1} \widehat\eta_6)
+ \operatorname{tr}(F^\top D^{-1} \widehat\eta_7)
- \mu^\top D^{-1} F \widehat\eta_8
- \gamma^\top D^{-1} F \widehat\eta_9 \\
&\quad - \frac{1}{2} \operatorname{tr}(F^\top D^{-1} F \widehat\eta_{10})
- \mu^\top D^{-1} \gamma
+ L_{\mathrm{GIG}}(p, a, b \mid \widehat\eta_1, \widehat\eta_2, \widehat\eta_3),
\end{aligned}
```

where $L_{\mathrm{GIG}}$ is the [GIG log likelihood](./normix-generalized-inverse-gaussian.md#gig-loglik),
using the order $(\log Y,Y^{-1},Y)$ throughout. The fixed standard-normal
term $-\tfrac12\operatorname{tr}(\widehat\eta_{10})$ and parameter-independent
normalizing constants are omitted from $L_{\mathrm{FA}}$.

## M-step for the normal block

Assume $\widehat\eta_{10}$ is invertible and the two location/skew equations below
have a nonsingular coefficient matrix. Eliminating $F$ from the normal
equations gives the following auxiliary quantities:

```{math}
:label: fa-aux

\begin{aligned}
q_1 &= \widehat\eta_8^\top \widehat\eta_{10}^{-1} \widehat\eta_8 - \widehat\eta_2, \\
q_2 &= \widehat\eta_9^\top \widehat\eta_{10}^{-1} \widehat\eta_8 - 1, \\
q_3 &= \widehat\eta_9^\top \widehat\eta_{10}^{-1} \widehat\eta_9 - \widehat\eta_3, \\
q_4 &= \widehat\eta_7 \widehat\eta_{10}^{-1} \widehat\eta_8 - \widehat\eta_5, \\
q_5 &= \widehat\eta_7 \widehat\eta_{10}^{-1} \widehat\eta_9 - \widehat\eta_4.
\end{aligned}
```

Here $\widehat\eta_7$ is $d\times r$, so $q_4,q_5$ are $d$-vectors.
The source transpose on $\widehat\eta_7$ in these two expressions is corrected to
make the products dimensionally consistent. The parameter updates are:

```{math}
:label: fa-mstep

\begin{aligned}
\mu &= \frac{q_2 \, q_5 - q_3 \, q_4}{q_2^2 - q_1 \, q_3}, \\
\gamma &= \frac{q_2 \, q_4 - q_1 \, q_5}{q_2^2 - q_1 \, q_3}, \\
F &= (\widehat\eta_7 - \mu \, \widehat\eta_8^\top - \gamma \, \widehat\eta_9^\top) \, \widehat\eta_{10}^{-1}, \\
D &= \operatorname{diag}\!\big(
\widehat\eta_2 \mu \mu^\top + \widehat\eta_3 \gamma \gamma^\top
- \widehat\eta_4 \gamma^\top - \gamma \widehat\eta_4^\top
- \widehat\eta_5 \mu^\top - \mu \widehat\eta_5^\top + \widehat\eta_6 \\
&\qquad - \widehat\eta_7 F^\top - F \widehat\eta_7^\top
+ F \widehat\eta_8 \mu^\top + \mu (F \widehat\eta_8)^\top
+ F \widehat\eta_9 \gamma^\top + \gamma (F \widehat\eta_9)^\top \\
&\qquad + F \widehat\eta_{10} F^\top
+ \mu \gamma^\top + \gamma \mu^\top\big), \\
(p, a, b) &= \arg\max_{p, a, b} L_{\mathrm{GIG}}(p, a, b \mid \widehat\eta_1, \widehat\eta_2, \widehat\eta_3).
\end{aligned}
```

## Conditional Expectations for the E-Step

Integrating [](#fa-joint) over $z$ gives the ordinary GH joint law of
$(X,Y)$ with dispersion $\Sigma=FF^\top+D$ and conditional covariance
$\operatorname{Cov}(X\mid Y)=Y\Sigma$.
Therefore the conditional distribution of $Y$ given $X$ is:

```{math}
Y \mid X = x \sim \operatorname{GIG}\!\left(p - \frac{d}{2}, \,
a + \gamma^\top\Sigma^{-1}\gamma, \,
b + (x - \mu)^\top\Sigma^{-1}(x - \mu)\right),
```

and the conditional moments $\mathbb E_\vartheta[Y^\alpha\mid X]$ and
$\mathbb E_\vartheta[\log Y\mid X]$ are computed using [](./normix-em-algorithm.md#gig-moment-cond).

The conditional distribution of $(X, Z)$ given $Y$ is Gaussian:

```{math}
\begin{aligned}
\begin{pmatrix} (X - \mu - \gamma Y)/\sqrt{Y} \\ Z \end{pmatrix}
\Biggm| Y, \vartheta
\sim\mathcal N_{d+r}\!\left(0, \begin{pmatrix}
\Sigma & F \\ F^\top & I_r
\end{pmatrix}\right).
\end{aligned}
```

Define the local regression matrix $A=F^\top\Sigma^{-1}$. The conditional
expectations of the latent factor $Z$ are:

```{math}
\begin{aligned}
\mathbb E_\vartheta[Z Y^{-1/2}\mid X] &= A(X - \mu) \mathbb E_\vartheta[Y^{-1}\mid X] - A \gamma, \\
\mathbb E_\vartheta[Z Y^{1/2}\mid X] &= A(X - \mu) - A \gamma \, \mathbb E_\vartheta[Y\mid X], \\
\mathbb E_\vartheta[Z Z^\top\mid X] &= I_r - A F
+ A(X - \mu)(X - \mu)^\top A^\top \mathbb E_\vartheta[Y^{-1}\mid X] \\
&\quad - A(X - \mu)\gamma^\top A^\top
- A \gamma (X - \mu)^\top A^\top
+ A \gamma \gamma^\top A^\top \mathbb E_\vartheta[Y\mid X].
\end{aligned}
```

## E-Step

Given i.i.d. samples $x_1, \ldots, x_n$ and current parameters
$\vartheta_k = (\mu_k, \gamma_k, F_k, D_k, p_k, a_k, b_k)$, the E-step
computes the posterior average $\widehat\eta_k$ of all ten statistic blocks.
Write $\mathbb E_k$ for expectation under $\vartheta_k$ and
$\widehat\eta_{k,i}$ for block $i$. The first six are the same as the
standard EM algorithm (see [](./normix-em-algorithm.md#e-step)):

```{math}
\begin{aligned}
\widehat\eta_{k,1} &= \frac{1}{n} \sum_{j=1}^n \mathbb E_k[\log Y\mid X=x_j], \\
\widehat\eta_{k,2} &= \frac{1}{n} \sum_{j=1}^n \mathbb E_k[Y^{-1}\mid X=x_j], \\
\widehat\eta_{k,3} &= \frac{1}{n} \sum_{j=1}^n \mathbb E_k[Y\mid X=x_j], \\
\widehat\eta_{k,4} &= \frac{1}{n} \sum_{j=1}^n x_j, \\
\widehat\eta_{k,5} &= \frac{1}{n} \sum_{j=1}^n x_j \, \mathbb E_k[Y^{-1}\mid X=x_j], \\
\widehat\eta_{k,6} &= \frac{1}{n} \sum_{j=1}^n
x_j x_j^\top \mathbb E_k[Y^{-1}\mid X=x_j].
\end{aligned}
```

The remaining four are determined by the first six using
$A_k = F_k^\top (F_k F_k^\top + D_k)^{-1}$:

```{math}
\begin{aligned}
\widehat\eta_{k,7} &= (\widehat\eta_{k,6} - \widehat\eta_{k,5} \mu_k^\top
- \widehat\eta_{k,4} \gamma_k^\top) A_k^\top, \\
\widehat\eta_{k,8} &= A_k (\widehat\eta_{k,5} - \mu_k \, \widehat\eta_{k,2} - \gamma_k), \\
\widehat\eta_{k,9} &= A_k (\widehat\eta_{k,4} - \mu_k - \gamma_k \, \widehat\eta_{k,3}), \\
\widehat\eta_{k,10} &= I_r - A_k F_k
+ A_k \big(\widehat\eta_{k,6} - \widehat\eta_{k,5} \mu_k^\top
- \mu_k (\widehat\eta_{k,5})^\top + \mu_k \mu_k^\top \widehat\eta_{k,2} \\
&\quad - (\widehat\eta_{k,4} - \mu_k) \gamma_k^\top
- \gamma_k (\widehat\eta_{k,4} - \mu_k)^\top
+ \gamma_k \gamma_k^\top \widehat\eta_{k,3}\big) A_k^\top.
\end{aligned}
```

The M-step then applies [](#fa-mstep) and [](#fa-aux) with
$\widehat\eta=\widehat\eta_k=(\widehat\eta_{k,1},\ldots,\widehat\eta_{k,10})$,
suppressing the iteration index in those M-step formulas. These averages
need not equal all ambient model expectations $\nabla\psi(\theta(\vartheta))$
at a constrained optimum.

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
the usual [EM ascent argument](https://xshi19.github.io/math/ig/information-geometry-latent-variables-em/)
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
