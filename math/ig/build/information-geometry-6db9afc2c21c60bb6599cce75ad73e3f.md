---
title: Dual coordinates and KL projections
---

A regular minimal exponential family has two useful coordinate systems:
natural parameters and expected sufficient statistics. They describe the same
distributions but assign different meanings to a straight path. Their relation
to Fisher information gives an exact KL version of Pythagoras.

## Two coordinate systems from one potential

Use the full exponential family

```{math}
p_\theta(x)=h(x)\exp\{\theta^\mathsf{T}t(x)-\psi(\theta)\}
```

with respect to a fixed measure $\nu$, with carrier $h$, sufficient statistic
$t(x)$, and open convex natural domain $\Theta$. Assume minimality and the
regularity from the [exponential-family note](./information-geometry-exponential-families.md).
Write $\eta(\theta)=\nabla\psi(\theta)$ for expectation coordinates and
$I(\theta)=\nabla^2\psi(\theta)>0$ for the Fisher information matrix.
Work on the image of this gradient map; every segment used below is assumed
to remain in the relevant coordinate domain.

The convex conjugate of $\psi$ is

```{math}
\psi^*(\eta)=\sup_{\vartheta\in\Theta}
\{\vartheta^\mathsf{T}\eta-\psi(\vartheta)\}.
```

At $\eta=\nabla\psi(\theta)$, strict convexity puts the unique maximizer at
$\vartheta=\theta$. Differentiating
$\psi^*(\eta)=\theta(\eta)^\mathsf{T}\eta-\psi(\theta(\eta))$
cancels the terms involving $d\theta$, giving

```{math}
:label: ig-legendre-duality
\nabla\psi^*(\eta)=\theta,\qquad
\nabla^2\psi^*(\eta)=I(\theta)^{-1}.
```

The Fisher metric $g$ has matrix $I(\theta)$ in natural coordinates and
$I(\theta)^{-1}$ in expectation coordinates, since
$d\eta=I(\theta)\,d\theta$. For Bernoulli distributions with success probability
$q$, $\eta=q$ and
$\psi^*(q)=q\log q+(1-q)\log(1-q)$ on $(0,1)$, the negative binary entropy.
For a general carrier $h$, the conjugate need not equal negative entropy
relative to $\nu$ without a carrier-dependent term.

## KL is a Bregman divergence with a particular orientation

For a differentiable strictly convex function $F$, define

```{math}
B_F(a,b)=F(a)-F(b)-\nabla F(b)^\mathsf{T}(a-b).
```

It is the gap between $F(a)$ and the tangent plane to $F$ at $b$. Convexity
makes it nonnegative; it is generally asymmetric. Taking the expected log
density ratio inside our family gives

```{math}
:label: ig-kl-bregman
\begin{aligned}
D_{\mathrm{KL}}(p_\theta\|p_\phi)
&=\psi(\phi)-\psi(\theta)-\eta(\theta)^\mathsf{T}(\phi-\theta)\\
&=B_\psi(\phi,\theta)
=B_{\psi^*}(\eta(\theta),\eta(\phi)).
\end{aligned}
```

The order of the natural parameters is reversed in $B_\psi$.
Checking this order prevents sign and projection errors later.

## Two notions of straightness

Parameterize the following segments by $\lambda\in[0,1]$.
An **exponential geodesic**, or e-geodesic, is affine in natural coordinates:
$\theta=(1-\lambda)\theta_0+\lambda\theta_1$. Its density is
proportional to $p_{\theta_0}^{1-\lambda}p_{\theta_1}^\lambda$, with the
normalizer restoring total mass one.

A **mixture geodesic**, or m-geodesic of the induced connection on this family,
is affine in expectation coordinates:
$\eta=(1-\lambda)\eta(\theta_0)+\lambda\eta(\theta_1)$.
It linearly interpolates the expected sufficient statistics. Inside a general
exponential family it need not be the
literal mixture $(1-\lambda)p_{\theta_0}+\lambda p_{\theta_1}$. For example,
mixing two independent product distributions can introduce dependence and
leave a family of independent variables, while interpolating its expectation
parameters stays within that family.

These straightness rules define two flat torsion-free connections,
$\nabla^{(e)}$ and $\nabla^{(m)}$. They are dual with respect to $g$: for
smooth vector fields $U,V,W$,

```{math}
U[g(V,W)]
=g(\nabla^{(e)}_U V,W)+g(V,\nabla^{(m)}_U W).
```

The natural and expectation coordinate bases are metric-dual:

```{math}
g\!\left(\frac{\partial}{\partial\theta_i},
         \frac{\partial}{\partial\eta_j}\right)=\delta_{ij}.
```

Indeed, changing the second basis multiplies by $I(\theta)^{-1}$, and
$I(\theta)I(\theta)^{-1}$ is the identity. This is the algebraic source of
the orthogonality calculation that follows.

## A three-point identity and Pythagoras

For three natural parameters $\theta,\phi,\chi$, substitution into
[](#ig-kl-bregman) and cancellation of the potential terms gives

```{math}
:label: ig-kl-three-point
\begin{aligned}
D_{\mathrm{KL}}(p_\theta\|p_\chi)
&=D_{\mathrm{KL}}(p_\theta\|p_\phi)
 +D_{\mathrm{KL}}(p_\phi\|p_\chi)\\
&\quad +(\eta(\phi)-\eta(\theta))^\mathsf{T}(\chi-\phi).
\end{aligned}
```

At $\phi$, the m-geodesic toward $\theta$ has expectation-coordinate velocity
$\eta(\theta)-\eta(\phi)$. The e-geodesic toward $\chi$ has natural-coordinate
velocity $\chi-\phi$. Their Fisher inner product is
$(\eta(\theta)-\eta(\phi))^\mathsf{T}(\chi-\phi)$ by the dual-basis identity.
If it vanishes, the cross term above vanishes and

```{math}
:label: ig-kl-pythagoras
D_{\mathrm{KL}}(p_\theta\|p_\chi)
=D_{\mathrm{KL}}(p_\theta\|p_\phi)
 +D_{\mathrm{KL}}(p_\phi\|p_\chi).
```

This is an exact divergence identity for these orthogonal dual geodesics.
It is different from a squared-distance identity for arbitrary Riemannian
triangles, and from the
[$L^2$ projection identity](./information-geometry-conditional-expectation.md#ig-conditional-pythagoras).

## When minimization gives the orthogonality condition

Let $S=(a+V)\cap\Theta$ be an affine constraint in natural coordinates,
where $V$ is a linear subspace. Fix $p_\theta$ and suppose a minimizer
$\phi\in S$ of $D_{\mathrm{KL}}(p_\theta\|p_\phi)$ exists. Since $\Theta$
is open, $\phi$ is interior relative to $a+V$. The directional derivative
along any $v\in V$ is

```{math}
v^\mathsf{T}(\eta(\phi)-\eta(\theta))=0.
```

For every $\chi\in S$, we have $\chi-\phi\in V$, so the Pythagorean equality
holds. Strict convexity in the second natural argument makes this minimizer
unique. Existence and interior attainment are hypotheses, not consequences
of the word projection. More general convex constraints yield a Pythagorean
inequality through a one-sided first-order condition; curved constraints do
not give this global affine argument.

For an explicit two-dimensional example, take two independent Bernoulli
variables with success probabilities $(q_1,q_2)$. Their natural parameters
are the component log odds and
$\psi(\theta)=\log(1+e^{\theta_1})+\log(1+e^{\theta_2})$.
Constrain the second success probability to a fixed $c\in(0,1)$, which fixes
the second natural parameter. The projection of $(a,b)$ is $(a,c)$.
For any other constrained point $(d,c)$, KL additivity for product laws gives

```{math}
\begin{aligned}
D((a,b)\|(d,c))
&=d_{\mathrm B}(a\|d)+d_{\mathrm B}(b\|c)\\
&=D((a,b)\|(a,c))+D((a,c)\|(d,c)),\\
d_{\mathrm B}(u\|v)
&=u\log\frac uv+(1-u)\log\frac{1-u}{1-v}.
\end{aligned}
```

Here $D$ denotes KL between the product laws. The m-direction from $(a,c)$
to $(a,b)$ changes only the second expectation coordinate; the e-direction
toward $(d,c)$ changes only the first natural coordinate. Their Fisher inner
product is zero.

## Alpha-connections and the meaning of curvature

For a general regular statistical model, write
$s_\theta(x)=\nabla_\theta\log p_\theta(x)$ for its score. Assuming finite
score third moments, define the symmetric cubic tensor by its components

```{math}
C_{ijk}=\mathbb E_\theta[
  s_{\theta,i}(X)s_{\theta,j}(X)s_{\theta,k}(X)].
```

Assume enough smoothness for the metric and this tensor to define smooth
connections. Let $C^\sharp$ be the vector-valued tensor obtained by raising
its last index with the inverse Fisher metric, so
$g(C^\sharp(U,V),W)=C(U,V,W)$. We use the convention

```{math}
:label: ig-alpha-connections
\nabla^{(\alpha)}_U V
=\nabla^{\mathrm{LC}}_U V-\frac\alpha2 C^\sharp(U,V).
```

The connections for $\alpha$ and $-\alpha$ are dual. In our full exponential
family, $C_{ijk}=\partial_i\partial_j\partial_k\psi$, the Levi-Civita
coefficients with the final index lowered are $C_{ijk}/2$ in natural
coordinates, and therefore

```{math}
\Gamma^{(\alpha)}_{ijk}
=\frac{1-\alpha}{2}\partial_i\partial_j\partial_k\psi.
```

At $\alpha=1$, these coefficients vanish in natural coordinates: this is
the exponential connection. At $\alpha=-1$, the mixture connection has
vanishing coefficients in expectation coordinates. The Levi-Civita connection
is $\alpha=0$.

Curvature measures the obstruction to path-independent parallel transport
locally. It belongs to a specified connection. **Dual flatness** says that the
exponential and mixture connections are flat. It does not assert that the
Levi-Civita connection is flat. Conversely, a general regular statistical
model can possess a Fisher metric and these alpha-connections without a
dually flat structure. A Hessian metric in natural affine coordinates need
not remain an ordinary coordinate Hessian after a nonlinear chart change.

For a systematic treatment of the dual connections and their projection
geometry, see §§3.3–3.9 of Frank Nielsen's
[An elementary introduction to information geometry](https://arxiv.org/abs/1808.08271).
Detailed curvature examples and EM as alternating KL projection belong to the
later sequence. The present [EM note](./information-geometry-latent-variables-em.md)
establishes the ordinary likelihood calculation needed first.

Return to the [Information Geometry hub](./ig/index.md) for the
complete entry path.
