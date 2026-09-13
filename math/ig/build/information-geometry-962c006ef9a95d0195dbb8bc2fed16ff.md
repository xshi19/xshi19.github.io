---
title: Fisher geometry and the meaning of L2
---

Euclidean distance between parameter vectors measures changes in the chosen
coordinates. Fisher geometry instead measures first-order changes in the log
density, averaged under the model. Its coordinate matrix changes when the
parameters change, while the length assigned to the same statistical motion
stays fixed.

## A regular model and its scores

Let $p_\theta$ be densities with respect to a fixed measure $\nu$, with
$\theta$ in an open subset of $\mathbb R^d$. Assume a common positive support,
twice continuously differentiable densities, and local integrable bounds that
justify differentiating the normalizing integral twice. Assume the scores are
square integrable. For the KL expansion below, also require that the expected
log density has a second-order Taylor expansion with an integrable remainder.

Write $\ell_\theta(x)=\log p_\theta(x)$ and define the column score vector

```{math}
s_\theta(x)=\nabla_\theta\ell_\theta(x).
```

Along a curve with velocity $v$, the log density changes at rate
$v^\mathsf{T}s_\theta(x)$. Differentiating $\int p_\theta\,d\nu=1$ gives

```{math}
\mathbb E_\theta[s_\theta]
=\int\nabla_\theta p_\theta\,d\nu=0.
```

The **Fisher information matrix per observation** is

```{math}
:label: ig-fisher-information
I_\theta=\mathbb E_\theta[s_\theta s_\theta^\mathsf{T}],\qquad
g_\theta(v,w)=v^\mathsf{T}I_\theta w.
```

Thus $g_\theta(v,v)$ is the mean square directional change in log density.
The matrix is always positive semidefinite. It gives a Riemannian metric on a
regular region where it is positive definite and smooth. A null direction
means a zero first-order change in the density almost everywhere; global
identifiability alone does not exclude such singular parameterizations.

## Local KL divergence gives the same quadratic form

Differentiating the normalizer a second time, using
$\partial_i\partial_jp=p(\partial_i\partial_j\ell+
\partial_i\ell\,\partial_j\ell)$, gives

```{math}
\mathbb E_\theta[\nabla_\theta^2\ell_\theta]=-I_\theta.
```

Now hold the first distribution fixed in

```{math}
D_{\mathrm{KL}}(p_\theta\|p_{\theta+\delta})
=\mathbb E_\theta[\ell_\theta(X)-\ell_{\theta+\delta}(X)].
```

Taylor expansion of the second log density has a linear term with expectation
zero and a quadratic term involving $-I_\theta$. Therefore

```{math}
:label: ig-local-kl
D_{\mathrm{KL}}(p_\theta\|p_{\theta+\delta})
=\frac12\delta^\mathsf{T}I_\theta\delta
 +o(\|\delta\|_2^2).
```

This is a local expansion as $\delta\to0$ at a fixed interior parameter.
Finite KL is generally asymmetric and is not squared Riemannian distance.
For $n$ independent observations, scores add and their cross-covariances vanish,
so the information matrix is $nI_\theta$.

## Reparameterization changes the matrix, not the length

Use a smooth invertible chart $\theta=\theta(\phi)$ and let
$J=\partial\theta/\partial\phi$. By the chain rule,

```{math}
s_\phi=J^\mathsf{T}s_\theta,\qquad
I_\phi=J^\mathsf{T}I_\theta J.
```

Since $d\theta=J\,d\phi$, the two quadratic expressions agree:

```{math}
:label: ig-fisher-invariance
d\theta^\mathsf{T}I_\theta d\theta
=d\phi^\mathsf{T}I_\phi d\phi.
```

This is exactly the [metric transformation law](./information-geometry-euclidean-to-manifold.md#ig-metric-coordinate-change).
A Euclidean metric can also be transformed correctly. The problem arises if
one declares the identity matrix to be the metric anew in every nonlinear
chart: that changes the geometry.

## Bernoulli probability and log odds

For $X\sim\operatorname{Bernoulli}(q)$ with $0<q<1$, differentiation gives

```{math}
s_q(x)=\frac{x}{q}-\frac{1-x}{1-q}
=\frac{x-q}{q(1-q)}.
```

Because $\operatorname{Var}(X)=q(1-q)$,

```{math}
I_q=\frac1{q(1-q)},\qquad ds^2=\frac{dq^2}{q(1-q)}.
```

Equal small changes in probability have different local statistical sizes:
the coefficient is $4$ at $q=1/2$ and $10000/99$ at $q=1/100$. This statement
uses the infinitesimal metric, not a finite-step equality for KL.

Now take the natural parameter $\theta=\log(q/(1-q))$. Since
$dq/d\theta=q(1-q)$, the same metric is

```{math}
:label: ig-bernoulli-fisher-charts
I_\theta=q(1-q),\qquad
ds^2=q(1-q)d\theta^2=\frac{dq^2}{q(1-q)}.
```

This agrees with $I_\theta=\psi''(\theta)$ from the
[exponential-family calculation](./information-geometry-exponential-families.md).
There are also models where Fisher information is constant in a useful chart.
For $N(\mu,\sigma^2)$ with known $\sigma$, the score for $\mu$ is
$(x-\mu)/\sigma^2$, and $ds^2=d\mu^2/\sigma^2$. A scaled Euclidean metric
is appropriate in this case.

## Three different uses of L2

The Euclidean norm on a finite parameter vector is more precisely an
$\ell^2$ norm. It should be distinguished from two function-space constructions.

| Space | Squared size | What determines it |
| --- | --- | --- |
| Parameter coordinates | $\sum_i v_i^2$ | A chosen Euclidean inner product in that chart |
| Scores in $L^2(P_\theta)$ | $\mathbb E_\theta[(v^\mathsf{T}s_\theta)^2]$ | The model law at the point; this is Fisher information |
| Density differences in $L^2(\nu)$ | $\int(p-q)^2\,d\nu$ | A reference measure and densities relative to it; finiteness is an extra requirement |

Raw density $L^2$ is invariant under parameter relabeling because the densities
do not change. It generally depends on the measurement coordinates. For
Lebesgue densities and $Y=aX$ with $a>0$, the transformed densities satisfy
$p_Y(y)=p_X(y/a)/a$, so

```{math}
\int(p_Y-q_Y)^2\,dy=\frac1a\int(p_X-q_X)^2\,dx.
```

A fixed invertible transformation of the observation contributes a
parameter-independent Jacobian to the log density, so it leaves scores and
Fisher information unchanged. This explains a distinction between Fisher
geometry and raw density $L^2$ that parameter invariance alone cannot explain.

## Square-root densities recover Fisher geometry

There is a useful $L^2(\nu)$ representation. Map a density to
$u_\theta=2\sqrt{p_\theta}$, which has norm $2$. Assume this map is
differentiable in $L^2(\nu)$, with derivative along $v$ given by

```{math}
\partial_vu_\theta=\sqrt{p_\theta}\,v^\mathsf{T}s_\theta.
```

Then its squared $L^2(\nu)$ norm is precisely $v^\mathsf{T}I_\theta v$.
This is an isometric realization of the tangent metric by square-root
densities. It is not the raw density-difference construction above.

With the convention

```{math}
H^2(p,q)=1-\int\sqrt{pq}\,d\nu
=\frac12\|\sqrt p-\sqrt q\|_{L^2(\nu)}^2,
```

the same differentiability gives

```{math}
:label: ig-local-hellinger
H^2(p_\theta,p_{\theta+\delta})
=\frac18\delta^\mathsf{T}I_\theta\delta+o(\|\delta\|_2^2).
```

The factors $1/2$ for KL and $1/8$ for this squared Hellinger convention
describe the same local Fisher metric. Finite Hellinger distance measures a
chord between square-root densities; Fisher–Rao distance minimizes path length
within the specified statistical model.

For background on the metric and these representations, see §§3.9–3.12 of
Frank Nielsen's [An elementary introduction to information geometry](https://arxiv.org/abs/1808.08271).
The [conditional-expectation note](./information-geometry-conditional-expectation.md)
explains projection in the other Hilbert space used here, $L^2(P_\theta)$.

Continue with [dual coordinates and KL projections](./information-geometry-duality.md),
or return to the [Information Geometry hub](./ig/index.md).
