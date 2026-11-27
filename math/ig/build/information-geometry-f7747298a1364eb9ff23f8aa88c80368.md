---
title: From Euclidean space to a manifold
---

A normal law has several coordinate descriptions: mean and standard deviation,
mean and variance, natural parameters, or expected sufficient statistics. A
change of coordinates changes the numbers used to describe a perturbation of
the law. It should preserve the perturbation itself and its statistical size.

This note develops the geometry needed to make that distinction precise. We
construct tangent vectors, use Fisher information to measure them, and define
connections that differentiate and transport them. The family $N(\mu,\sigma^2)$
provides explicit calculations throughout. It also exhibits a distinction that
matters later: its Fisher metric has negative Riemannian curvature, while its
exponential and mixture connections are flat.

## Notation and conventions

We follow the [shared notation](https://xshi19.github.io/math/notation/), with
$\xi$ used **locally in this note** for a general chart. For normals,
$\xi=(\mu,\sigma)^{\mathsf T}$; $\theta$ is reserved for natural parameters,
and $\eta$ for expectation parameters. This local use of $\xi$ is unrelated to
the extreme-value index in the Incerto track.

- Coordinates $\xi^i$ and tangent components $v^i$ carry upper indices;
  covector components $\omega_i$ and metric components $g_{ij}$ carry lower
  indices. An index repeated once above and once below is summed from $1$ to
  the chart dimension. Explicit sums over sample outcomes are written out.
  Superscripts naming components, such as $v^\sigma$, are not powers.
- $\partial_i=\partial/\partial\xi^i$ is the coordinate tangent basis, and
  $d\xi^i$ is its dual covector basis. In particular, $d\theta^i$ denotes the
  dual basis in natural coordinates. The inverse metric has components
  $g^{ij}$, with $g^{ik}g_{kj}=\delta^i{}_j$.
- $df_q:T_qM\to T_{f(q)}N$ is the differential of a smooth map. $Df(\xi)$
  denotes its Jacobian matrix in specified charts, also when $f$ is a change
  of coordinates. Matrices act on column components; the transpose is
  $\mathsf T$.
- $G(\xi)=[g_{ij}(\xi)]$ is a general metric's matrix in the $\xi$ chart. Thus
  $g_\xi(v,w)=g_{ij}v^iw^j=v_\xi^{\mathsf T}G(\xi)w_\xi$.
  For the Fisher metric we write $I(\xi)$ for this matrix, and likewise
  $I(\phi)$, $I(\theta)$, or $I(\eta)$ in the other charts.
  $ds^2=g_{ij}\,d\xi^i d\xi^j$ denotes the corresponding quadratic form.
- $\Gamma_{ij}^k$ means $\nabla_{\partial_i}\partial_j=\Gamma_{ij}^k\partial_k$;
  $\Gamma_{ij,k}=g_{k\ell}\Gamma_{ij}^{\ell}$ lowers its output index. A
  superscript $(\alpha)$ identifies a connection, not a tensor index.
- $\ell_\xi(x)=\log p_\xi(x)$, $s_{\xi,i}(x)=\partial_i\ell_\xi(x)$, and
  $s_\xi(x)=(s_{\xi,i}(x))$ is the column of score components. For fixed $x$,
  these are covector components: the coordinate gradient symbol in
  $s_\xi=\nabla_\xi\ell_\xi$ does not mean a Riemannian gradient.
  $\mathbb E_\xi$ means expectation under $P_\xi$; in natural coordinates we
  write $p_\theta$, $s_\theta$, and $\mathbb E_\theta$.

We use matrices for explicit Jacobians and the Legendre identities
$\eta(\theta)=\nabla\psi(\theta)$ and $\nabla^2\psi^*(\eta)=(\nabla^2\psi(\theta))^{-1}$.
Here the gradients and Hessians are ordinary derivatives in the named affine
coordinates. Their entries will be related to tensor components explicitly.
Curve parameters are denoted by $\lambda$, leaving $t(x)$ for sufficient
statistics. A dot denotes differentiation with respect to $\lambda$.
$U,V,W$ denote vector fields, while $X$ denotes an observation.

## A running example: normal location and scale

Let $M$ be the set of laws $N(\mu,\sigma^2)$ with $\mu\in\mathbb R$ and
$\sigma>0$. Relative to Lebesgue measure on $\mathbb R$,

```{math}
:label: ig-normal-density
p_\xi(x)=\frac{1}{\sqrt{2\pi}\,\sigma}
 \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),
\qquad \xi=(\mu,\sigma)^{\mathsf T}.
```

The sample space is one-dimensional, but the family of laws is
two-dimensional. The mean and variance determine each law uniquely. We
exclude $\sigma=0$, whose point mass is outside this density model.

### Four charts on the same family

The following are global charts on $M$. The symbol $\tau$ is local notation
for variance.

| Chart | Coordinates | Coordinate domain |
| --- | --- | --- |
| Location and scale | $\xi=(\mu,\sigma)^{\mathsf T}$ | $\sigma>0$ |
| Location and variance | $\phi=(\mu,\tau)^{\mathsf T}$, $\tau=\sigma^2$ | $\tau>0$ |
| Natural | $\theta=(\mu/\sigma^2,-1/(2\sigma^2))^{\mathsf T}$ | $\theta^2<0$ |
| Expectation | $\eta=(\mu,\mu^2+\sigma^2)^{\mathsf T}$ | $\eta^2>(\eta^1)^2$ |

:::{prf:proposition} Compatible charts and the normal exponential form
:label: prop-euclidean-to-manifold-normal-charts

Give the normal family $M=\{N(\mu,\sigma^2):\mu\in\mathbb R,\sigma>0\}$
the smooth structure of $\mathbb R\times(0,\infty)$. The four maps in the
table are global smooth charts with the stated open domains and invertible
transition Jacobians. With $t(x)=(x,x^2)^{\mathsf T}$, its Lebesgue density
in natural coordinates is

$$
p_\theta(x)=\exp\{\theta^{\mathsf T}t(x)-\psi(\theta)\},\qquad
\psi(\theta)=-\frac{(\theta^1)^2}{4\theta^2}
 +\frac12\log\frac{\pi}{-\theta^2},\quad\theta^2<0,
$$

and $\nabla\psi(\theta)=\mathbb E_\theta t(X)=(\mu,\mu^2+\sigma^2)^{\mathsf T}$.
:::

:::{prf:proof}
With $t(x)=(x,x^2)^{\mathsf T}$, completing the square gives

```{math}
:label: ig-normal-natural-chart
\begin{aligned}
p_\theta(x)&=\exp\{\theta^{\mathsf T}t(x)-\psi(\theta)\},\\
\psi(\theta)&=-\frac{(\theta^1)^2}{4\theta^2}
 +\frac12\log\frac{\pi}{-\theta^2},\\
\eta(\theta)&=\mathbb E_\theta[t(X)]=\nabla\psi(\theta)
 =(\mu,\mu^2+\sigma^2)^{\mathsf T}.
\end{aligned}
```

The inverse coordinate maps include

```{math}
\mu=-\frac{\theta^1}{2\theta^2},\qquad
\sigma=\sqrt{-\frac1{2\theta^2}};
\qquad
\mu=\eta^1,\qquad
\sigma=\sqrt{\eta^2-(\eta^1)^2}.
```

Direct differentiation gives the Jacobians from the $\xi$ chart to the
other three charts:

```{math}
:label: ig-normal-chart-jacobians
\begin{aligned}
\frac{\partial\phi}{\partial\xi}&=\begin{pmatrix}1&0\\0&2\sigma\end{pmatrix},\\[3pt]
\frac{\partial\theta}{\partial\xi}&=\begin{pmatrix}
 \sigma^{-2}&-2\mu\sigma^{-3}\\0&\sigma^{-3}
\end{pmatrix},\\[3pt]
\frac{\partial\eta}{\partial\xi}&=\begin{pmatrix}1&0\\2\mu&2\sigma\end{pmatrix}.
\end{aligned}
```

Their determinants are nonzero for $\sigma>0$. For an invertible coordinate
change $f$, the inverse Jacobian is
$Df^{-1}(f(\xi))=Df(\xi)^{-1}$; specifying its evaluation point prevents
confusing the two directions of a coordinate change.
:::

Thus the second expectation coordinate is the **second raw moment**, not the
variance. The [exponential-family note](./information-geometry-exponential-families.md#two-examples)
derives these coordinates in the broader setting of sufficient statistics.

### What makes this a manifold

A smooth $d$-manifold is a Hausdorff, second-countable topological space
covered by coordinate neighborhoods, each homeomorphic to an open subset of
$\mathbb R^d$, with smooth transition maps between these charts.
A chart assigns coordinates to points; its inverse interprets coordinate
values as points of the manifold. The normal family obtains such a structure
from its bijection with $\mathbb R\times(0,\infty)$, and the other charts
above are smoothly compatible with it.

A general parameterized statistical model needs more care. Parameters may
repeat the same law, and even an injective parameterization can have a
vanishing first derivative at a point. We work on regular regions where the
model is locally identifiable and the Fisher matrix is smooth and positive
definite. The normal calculation below verifies that condition. A statistical
manifold need not be presented as a curved surface inside a Euclidean space.

## Tangent vectors, functions, and differentials

### Curves define tangent vectors

Fix $q\in M$ and a chart $\xi$ containing it. Two smooth curves $c_1,c_2$
through $q$ at $\lambda=0$ are equivalent when

```{math}
\left.\frac{d}{d\lambda}\xi^i(c_1(\lambda))\right|_{\lambda=0}
=
\left.\frac{d}{d\lambda}\xi^i(c_2(\lambda))\right|_{\lambda=0}
\quad\text{for every }i.
```

:::{prf:proposition} Curve velocities define a tangent space
:label: prop-euclidean-to-manifold-curve-tangents

Let $M$ be a smooth $d$-manifold and $q\in M$. Equality of the coordinate
velocities at zero of two smooth curves through $q$ is independent of the
chosen chart. Their equivalence classes form a vector space $T_qM$, and
in every chart $\xi$ the velocity map $[c]\mapsto(\xi\circ c)'(0)$ is
a linear isomorphism $T_qM\to\mathbb R^d$.
:::

:::{prf:proof}
For a chart transition $\phi=f(\xi)$, the chain rule sends each velocity
column $v$ to $Df(\xi(q))v$. Since this Jacobian is invertible, equality
of velocities is equivalent in the two charts. A tangent
vector is an equivalence class $v=[c]$ of such curves. Its components are
$v^i=(\xi^i\circ c)'(0)$. Every column of $d$ real numbers is realized by a
coordinate line $\xi(c(\lambda))=\xi(q)+\lambda v$ for sufficiently small
$\lambda$. These components give the set of classes a $d$-dimensional
vector-space structure, called $T_qM$; coordinate changes act linearly on that space.
:::

For normals,
$v=v^\mu\partial_\mu+v^\sigma\partial_\sigma$ is represented by
$(\mu(\lambda),\sigma(\lambda))=(\mu+\lambda v^\mu,\sigma+\lambda v^\sigma)$
near $\lambda=0$.
The coordinates of a law and the components of a tangent vector have different
roles: one specifies a distribution; the other specifies rates of change at
that distribution.

### The equivalent definition by derivations

A derivation at $q$ is an $\mathbb R$-linear map
$A:C^\infty(M)\to\mathbb R$ satisfying the pointwise Leibniz rule

```{math}
A(FH)=F(q)A(H)+H(q)A(F).
```

:::{prf:proposition} Curve tangents and point derivations agree
:label: prop-euclidean-to-manifold-derivations

For a smooth $d$-manifold $M$ and $q\in M$, every $\mathbb R$-linear map
$A:C^\infty(M)\to\mathbb R$ satisfying
$A(FH)=F(q)A(H)+H(q)A(F)$ depends only on the germ of $F$ at $q$.
In a chart $\xi$ it has the unique form
$A(F)=v^i\partial_iF(q)$, where $v^i=A(\xi^i)$ is defined using local germs.
The correspondence $[c]\mapsto[F\mapsto(F\circ c)'(0)]$ identifies
$T_qM$ with these derivations, with basis $\partial_i|_q$.
:::

:::{prf:proof}
It depends only on functions near $q$. Indeed, if $F$ vanishes near $q$,
choose a smooth cutoff equal to one at $q$ and supported where $F=0$;
applying the product rule to their zero product gives $A(F)=0$. Thus we can
also work with germs of smooth functions at $q$, and apply $A$ to local
coordinate functions.

A curve class defines such a derivation by

```{math}
:label: ig-tangent-derivation
v[F]=\left.\frac{d}{d\lambda}F(c(\lambda))\right|_{\lambda=0}
=v^i\partial_iF(q),\qquad
\partial_i\big|_q[F]
=\left.\frac{\partial(F\circ\xi^{-1})}{\partial\xi^i}\right|_{\xi(q)}.
```

We now show that every derivation arises this way. On a small coordinate ball,
the integral form of first-order Taylor expansion writes a smooth function as

```{math}
F(y)-F(q)=(\xi^i(y)-\xi^i(q))H_i(y),\qquad
H_i(q)=\partial_iF(q).
```

A derivation annihilates constants, since $A(1)=A(1\cdot1)=2A(1)$.
Applying its product rule to this expansion gives
$A(F)=A(\xi^i)\partial_iF(q)$. Hence $A=v^i\partial_i|_q$ with
$v^i=A(\xi^i)$, represented by the coordinate line above. The two
constructions are inverse, and $\{\partial_i|_q\}$ is a basis of $T_qM$.
:::

For example, the smooth function $F(P_\xi)=\mathbb E_\xi[X^2]
=\mu^2+\sigma^2$ satisfies

```{math}
\partial_\mu F=2\mu,\qquad \partial_\sigma F=2\sigma,
\qquad v[F]=2\mu v^\mu+2\sigma v^\sigma.
```

Here $\partial_\mu$ holds $\sigma$ fixed, and $\partial_\sigma$ holds $\mu$
fixed. Holding a different second coordinate fixed can produce a different
basis vector even if the first coordinate has the same name.

### Coordinate changes and pushforwards

Let $\phi=f(\xi)$ be a transition map and set
$A^a{}_i=\partial\phi^a/\partial\xi^i$ and
$B^i{}_a=\partial\xi^i/\partial\phi^a$, so $B=A^{-1}$. Write
$\partial'_a=\partial/\partial\phi^a$ for the new coordinate basis.

:::{prf:proposition} Transformation of tangent bases and components
:label: prop-euclidean-to-manifold-tangent-charts

For a smooth chart transition $\phi=f(\xi)$, write
$A^a{}_i=\partial\phi^a/\partial\xi^i$, $B=A^{-1}$, and
$\partial'_a=\partial/\partial\phi^a$. At each point of the overlap,
$\partial_i=A^a{}_i\partial'_a$ and a tangent vector has components
$v'^a=A^a{}_iv^i$, or $v_\phi=Df(\xi)v_\xi$.
:::

:::{prf:proof}
Apply $\partial_i$ to a smooth function expressed in the $\phi$ chart;
the chain rule gives $\partial_iF=A^a{}_i\partial'_aF$. Then expand
$v=v^i\partial_i$ in the primed basis to obtain

```{math}
:label: ig-tangent-coordinate-change
\partial_i=A^a{}_i\partial'_a,\qquad
v'^a=A^a{}_i v^i,\qquad
v_\phi=Df(\xi)v_\xi.
```
:::

The basis and components change together, leaving the derivation $v$ fixed.
For the variance chart, [](#ig-normal-chart-jacobians) gives

```{math}
\partial_\mu\big|_\sigma=\partial_\mu\big|_\tau,\qquad
\partial_\sigma=2\sigma\partial_\tau,\qquad
v^\tau=2\sigma v^\sigma.
```

At $\sigma=2$, a unit scale velocity has variance velocity $4$. In expectation
coordinates the same vector has components
$(v^\mu,2\mu v^\mu+2\sigma v^\sigma)^{\mathsf T}$.

More generally, a smooth map $f:M\to N$ has a differential, or pushforward,
defined intrinsically by

```{math}
:label: ig-differential-definition
df_q([c])=[f\circ c],\qquad
(df_qv)[H]=v[H\circ f],\quad H\in C^\infty(N).
```

:::{prf:proposition} The differential has the coordinate Jacobian
:label: prop-euclidean-to-manifold-pushforward

For a smooth map $f:M\to N$ and $q\in M$, the map
$df_q([c])=[f\circ c]$ is well-defined and linear. In charts $\xi$ at $q$
and $\phi$ at $f(q)$ its matrix is $D(\phi\circ f\circ\xi^{-1})(\xi(q))$;
equivalently, $(df_qv)[H]=v[H\circ f]$ for every smooth $H$ on $N$.
:::

:::{prf:proof}
The chain rule gives
$(\phi\circ f\circ c)'(0)=D(\phi\circ f\circ\xi^{-1})(\xi(q))(\xi\circ c)'(0)$.
This depends only on the curve class and linearly on its velocity. Applying
the same rule to $H\circ f\circ c$ proves the action on functions.
Thus in source coordinates $\xi$ and target coordinates $\phi$, its matrix is the
Jacobian of $\phi\circ f\circ\xi^{-1}$, denoted $Df(\xi)$. Its entries are
$\partial f^a/\partial\xi^i$, and
$(df_qv)^a=(\partial f^a/\partial\xi^i)v^i$.
:::

Unlike a coordinate change, a
smooth map need not be invertible or preserve dimension. The variance map
$f:M\to(0,\infty)$, $f(P_\xi)=\sigma^2$, has
$df_qv=2\sigma v^\sigma\partial_\tau$: it discards the location direction.

### Covectors and vector fields

A cotangent vector is a linear functional $\omega:T_qM\to\mathbb R$.
The differentials $d\xi^i$ form the dual basis because
$d\xi^i(\partial_j)=\delta^i{}_j$. Thus

```{math}
\omega=\omega_i\,d\xi^i,\qquad
\omega(v)=\omega_i v^i,\qquad
dF=(\partial_iF)\,d\xi^i.
```

:::{prf:proposition} The dual coordinate transformation
:label: prop-euclidean-to-manifold-covectors

For a chart transition $\phi=f(\xi)$ with Jacobian $A$ and inverse $B$,
the dual bases obey $d\phi^a=A^a{}_i d\xi^i$. If
$\omega=\omega_i d\xi^i=\omega'_a d\phi^a$, then
$\omega'_a=B^i{}_a\omega_i$ and $\omega'_av'^a=\omega_iv^i$.
:::

:::{prf:proof}
Differentiate the coordinate functions $\phi^a(\xi)$ and use the chain
rule. Equating coefficients in $\omega=\omega'_aA^a{}_i d\xi^i$ and
multiplying by $B$ gives the component rule. Thus
$d\phi^a=A^a{}_i d\xi^i$ and $\omega'_a=B^i{}_a\omega_i$. This inverse
transformation is what makes $\omega(v)$ independent of coordinates.
:::

A smooth vector field $V$ assigns $V_q\in T_qM$ at every point, with smooth
components $V^i$ in every chart. It differentiates functions by
$V[F]=V^i\partial_iF$.

:::{prf:proposition} Integral curves of relative scale change
:label: prop-euclidean-to-manifold-scale-field

On the normal family, the field $V=\sigma\partial_\sigma$ has integral
curves $\mu(\lambda)=\mu(0)$, $\sigma(\lambda)=\sigma(0)e^\lambda$.
In the variance chart $\tau=\sigma^2$, the same field is $2\tau\partial_\tau$.
:::

:::{prf:proof}
The integral-curve equations are $\dot\mu=0$, $\dot\sigma=\sigma$;
integrating them gives the stated curves. The identity
$\partial_\sigma=2\sigma\partial_\tau$ gives the variance-chart field.
:::

## Metrics and Fisher information

### A metric is more than a parameter norm

A Riemannian metric $g$ assigns a smooth positive definite symmetric bilinear
form $g_q$ to every tangent space. Its components are
$g_{ij}=g(\partial_i,\partial_j)$.

:::{prf:proposition} Metric components under a change of chart
:label: prop-euclidean-to-manifold-metric-charts

Let $g$ be a Riemannian metric and let $\phi=f(\xi)$ be a smooth chart
transition with inverse Jacobian $B^i{}_a=\partial\xi^i/\partial\phi^a$.
Then $g'_{ab}=B^i{}_aB^j{}_bg_{ij}$, or
$G(\phi)=Df(\xi)^{-\mathsf T}G(\xi)Df(\xi)^{-1}$.
:::

:::{prf:proof}
Substituting
$v^i=B^i{}_a v'^a$ and $w^j=B^j{}_b w'^b$ gives

```{math}
:label: ig-metric-coordinate-change
\begin{aligned}
g'_{ab}(\phi)&=B^i{}_a B^j{}_b g_{ij}(\xi),\\
G(\phi)&=B^{\mathsf T}G(\xi)B
 =Df(\xi)^{-\mathsf T}G(\xi)Df(\xi)^{-1},
\qquad \phi=f(\xi).
\end{aligned}
```
:::

Every Riemannian metric has this covariance. It does not by itself select
Fisher information. Even a Euclidean metric acquires nonconstant coefficients
in nonlinear coordinates.

:::{prf:proposition} Euclidean metrics in polar and variance coordinates
:label: prop-euclidean-to-manifold-coordinate-metrics

On a plane sector with $\rho>0$, the coordinates
$x=\rho\cos\varphi$, $y=\rho\sin\varphi$ give
$dx^2+dy^2=d\rho^2+\rho^2d\varphi^2$.
On the normal parameter domain, $\tau=\sigma^2$ gives
$d\mu^2+d\tau^2=d\mu^2+4\sigma^2d\sigma^2$; this is different from
the metric $d\mu^2+d\sigma^2$.
:::

:::{prf:proof}
Differentiating the polar map gives
$dx=\cos\varphi\,d\rho-\rho\sin\varphi\,d\varphi$ and
$dy=\sin\varphi\,d\rho+\rho\cos\varphi\,d\varphi$.
Squaring and adding cancels the cross terms. On an angular sector, write
$x=\rho\cos\varphi$, $y=\rho\sin\varphi$, with radius $\rho>0$. Then

```{math}
:label: ig-polar-metric
ds^2=dx^2+dy^2=d\rho^2+\rho^2d\varphi^2.
```

For normals, choosing $d\mu^2+d\sigma^2$ in the scale chart and choosing
$d\mu^2+d\tau^2$ afresh in the variance chart gives incompatible metrics:

```{math}
d\mu^2+d\tau^2=d\mu^2+4\sigma^2d\sigma^2.
```

The second identity follows from $d\tau=2\sigma\,d\sigma$.
:::

Thus nonconstant metric coefficients alone do not establish curvature.
At $\sigma=2$, the same unit scale velocity has squared lengths $1$ and $16$.
A Euclidean metric chosen in one chart is legitimate, but must then be
transformed by [](#ig-metric-coordinate-change). A statistical principle is
needed to choose a metric without privileging that chart.

For a piecewise smooth curve $c:[a,b]\to M$, the metric determines length and
intrinsic distance by

```{math}
L_g(c)=\int_a^b\sqrt{g_{ij}(\xi(c(\lambda)))\dot\xi^i\dot\xi^j}\,d\lambda,
\qquad
d_g(q_0,q_1)=\inf_{c:q_0\to q_1}L_g(c).
```

Evaluate the integrand in a chart containing the curve point, splitting the
parameter interval between charts when needed. The infimum is over curves in
the specified manifold. Restricting to a subfamily can increase this distance.

### Scores measure changes in the law

For a general model, assume positive densities on a common support relative
to a fixed measure $\nu$. Assume smooth parameter dependence, square-integrable
scores, and local integrable bounds allowing differentiation of the normalizer
and the expectations used below. We will specify the additional Taylor and
third-moment assumptions where they are needed.

:::{prf:proposition} Centered scores and the Fisher quadratic form
:label: prop-euclidean-to-manifold-score-metric

Let $p_\xi>0$ be smooth densities on a fixed support relative to a fixed
measure $\nu$. Assume square-integrable scores
$s_{\xi,i}=\partial_i\log p_\xi$ and local integrable bounds permitting
differentiation of $\int p_\xi\,d\nu=1$. Then
$\partial_i p_\xi=p_\xi s_{\xi,i}$ and $\mathbb E_\xi s_{\xi,i}=0$.
The quadratic form
$g_\xi(v,w)=\mathbb E_\xi[(s_{\xi,i}v^i)(s_{\xi,j}w^j)]$
is positive semidefinite, with $g_\xi(v,v)=0$ exactly when
$v^i\partial_i p_\xi=0$ almost everywhere. Under a smooth invertible chart
change with inverse Jacobian $B$, its matrix transforms as
$I(\phi)=B^{\mathsf T}I(\xi)B$. A parameter-independent smooth invertible
change of observation coordinates also preserves this form when the usual
density change-of-variables formula applies.
:::

:::{prf:proof}
Differentiating the density and its normalization gives

```{math}
\begin{gathered}
\partial_i p_\xi(x)=p_\xi(x)s_{\xi,i}(x),\qquad
\mathbb E_\xi[s_{\xi,i}(X)]=0,\\
\left.\frac{d}{d\lambda}\ell_{\xi(c(\lambda))}(x)\right|_{\lambda=0}=s_{\xi,i}(x)v^i.
\end{gathered}
```

The **Fisher metric per observation** is

```{math}
:label: ig-fisher-score-metric
\begin{aligned}
g_\xi(v,w)&=\mathbb E_\xi[(s_{\xi,i}(X)v^i)(s_{\xi,j}(X)w^j)],\\
g_{ij}(\xi)&=\mathbb E_\xi[s_{\xi,i}(X)s_{\xi,j}(X)],\\
I(\xi)&=\mathbb E_\xi[s_\xi(X)s_\xi(X)^{\mathsf T}].
\end{aligned}
```

Its value on $(v,v)$ is the expectation of a square. It vanishes exactly
when $s_{\xi,i}v^i=0$ almost surely, equivalently
$v^i\partial_i p_\xi=0$ almost everywhere by positivity of $p_\xi$. Under reparameterization the score obeys
$s_{\phi,a}(x)=B^i{}_a s_{\xi,i}(x)$, so its covariance obeys exactly the
metric transformation law. Invertible, parameter-independent changes of the
observed variable also preserve Fisher information: their density Jacobian
contributes a term to the log density whose parameter derivative is zero.
:::

We restrict to regions where this form is smooth and positive definite,
so that it is a Riemannian metric.

### Computing the normal metric

:::{prf:proposition} Fisher metric of a normal location-scale law
:label: prop-euclidean-to-manifold-normal-metric

For one observation from $N(\mu,\sigma^2)$, $\mu\in\mathbb R$, $\sigma>0$,
the Fisher metric in $\xi=(\mu,\sigma)$ is
$I(\xi)=\operatorname{diag}(\sigma^{-2},2\sigma^{-2})$, so
$ds^2=(d\mu^2+2d\sigma^2)/\sigma^2$ and it is positive definite.
In $\phi=(\mu,\tau)$, $\tau=\sigma^2$, its matrix is
$I(\phi)=\operatorname{diag}(\tau^{-1},(2\tau^2)^{-1})$.
:::

:::{prf:proof}
Set $Z=(X-\mu)/\sigma\sim N(0,1)$. Differentiating [](#ig-normal-density),

```{math}
s_{\xi,\mu}(X)=\frac{X-\mu}{\sigma^2}=\frac Z\sigma,
\qquad
s_{\xi,\sigma}(X)=-\frac1\sigma+\frac{(X-\mu)^2}{\sigma^3}
 =\frac{Z^2-1}{\sigma}.
```

The second subscripts here name the covector components in the $\xi$ chart,
not differentiation of an already defined score. Odd standard-normal moments
vanish, and integration by parts gives $\mathbb E[Z^{2k}]=(2k-1)\mathbb E[Z^{2k-2}]$.
In particular, the second and fourth moments are $1$ and $3$. Consequently,

```{math}
:label: ig-normal-fisher-metric
\begin{aligned}
I(\xi)&=\begin{pmatrix}\sigma^{-2}&0\\0&2\sigma^{-2}\end{pmatrix},\\
g_\xi(v,w)&=\frac{v^\mu w^\mu+2v^\sigma w^\sigma}{\sigma^2},\\
ds^2&=\frac{d\mu^2+2d\sigma^2}{\sigma^2}.
\end{aligned}
```

This is positive definite for every $\sigma>0$. On compact parameter
neighborhoods with scale bounded away from zero, Gaussian tail bounds justify
the differentiations and moments used here and later.

Using $B=\operatorname{diag}(1,1/(2\sigma))$ in the variance chart gives

```{math}
:label: ig-normal-fisher-charts
I(\phi)=\begin{pmatrix}\tau^{-1}&0\\0&(2\tau^2)^{-1}\end{pmatrix},
\qquad
ds^2=\frac{d\mu^2+2d\sigma^2}{\sigma^2}
 =\frac{d\mu^2}{\tau}+\frac{d\tau^2}{2\tau^2}.
```

Substitution of $v^\tau=2\sigma v^\sigma$ recovers the same squared length.
:::

A fixed location
velocity has smaller Fisher length at a larger scale, because its change in
log density is smaller in mean square.

### Natural and expectation coordinates

:::{prf:proposition} Hessian metrics in the normal dual charts
:label: prop-euclidean-to-manifold-normal-dual-metrics

For the normal family in the natural and expectation charts above,
$I(\theta)=\nabla^2\psi(\theta)=\operatorname{Cov}_\theta(t(X))$, namely

$$
I(\theta)=\begin{pmatrix}\sigma^2&2\mu\sigma^2\\
2\mu\sigma^2&4\mu^2\sigma^2+2\sigma^4\end{pmatrix}.
$$

On $\eta^2>(\eta^1)^2$, the Legendre transform is
$\psi^*(\eta)=-\tfrac12\log(2\pi e[\eta^2-(\eta^1)^2])$.
Writing $\mu=\eta^1$ and $\tau=\eta^2-(\eta^1)^2$, the metric there is

$$
I(\eta)=\nabla^2\psi^*(\eta)=I(\theta(\eta))^{-1}
=\begin{pmatrix}\tau^{-1}+2\mu^2\tau^{-2}&-\mu\tau^{-2}\\
-\mu\tau^{-2}&(2\tau^2)^{-1}\end{pmatrix}.
$$
:::

:::{prf:proof}
In natural coordinates $s_\theta(X)=t(X)-\eta(\theta)$, so differentiating the
normalizer in [](#ig-normal-natural-chart) gives the matrix identities

```{math}
:label: ig-normal-dual-metrics
\begin{aligned}
I(\theta)
 &=\nabla^2\psi(\theta)=\operatorname{Cov}_\theta(t(X))\\
 &=\begin{pmatrix}
 \sigma^2&2\mu\sigma^2\\
 2\mu\sigma^2&4\mu^2\sigma^2+2\sigma^4
 \end{pmatrix},\\
\psi^*(\eta)&=\theta^{\mathsf T}\eta-\psi(\theta)
 =-\frac12\log\!\left(2\pi e[\eta^2-(\eta^1)^2]\right),\\
I(\eta)&=\nabla^2\psi^*(\eta)=I(\theta(\eta))^{-1}\\
 &=\begin{pmatrix}
 \tau^{-1}+2\mu^2\tau^{-2}&-\mu\tau^{-2}\\
 -\mu\tau^{-2}&(2\tau^2)^{-1}
 \end{pmatrix},\quad
 \mu=\eta^1,\quad \tau=\eta^2-(\eta^1)^2.
\end{aligned}
```

The expression for $\psi^*$ is the Legendre transform, attained at the natural
parameter $\theta=\theta(\eta)$ corresponding to $\eta$. Indeed,
$d\eta=I(\theta)\,d\theta$, whence $d\theta=I(\theta)^{-1}d\eta$;
the metric transformation law gives $I(\eta)$.
:::

Here $[I(\theta)]_{ij}=\partial^2\psi/(\partial\theta^i\partial\theta^j)$
and $[I(\eta)]_{ab}=\partial^2\psi^*/(\partial\eta^a\partial\eta^b)$.
The inverse of one coordinate matrix supplies the components in the other
chart; it does not change the covariant type of the metric. The Jacobians in
[](#ig-normal-chart-jacobians) pull both matrices back to $I(\xi)$.
An ordinary Hessian need not transform as a tensor under arbitrary nonlinear
coordinates; these Hessian formulas use the particular affine coordinates
$\theta$ and $\eta$.

## Why Fisher information is distinguished

Coordinate covariance is required of every metric. Fisher information has
additional properties concerning statistical distinguishability and the loss
of observations. We prove the local KL and monotonicity statements below;
the uniqueness theorem is cited with its finite-space hypotheses.

### The Hessian of KL at the diagonal

:::{prf:proposition} The local KL Hessian is Fisher information
:label: prop-euclidean-to-manifold-kl-hessian

Let $p_\xi>0$ be densities on a common support relative to a fixed measure,
smooth in an open parameter domain. Suppose scores are square integrable
and local integrable bounds permit twice differentiating normalization and
differentiating $\mathbb E_\xi\ell_\zeta$ with the first law fixed, where
$\ell_\zeta=\log p_\zeta$. Assume this expectation is finite near
$\zeta=\xi$ and has a second-order Taylor expansion there. Then

$$
\mathbb E_\xi[\partial_i\partial_j\ell_\xi]=-g_{ij}(\xi),\qquad
D(p_\xi\|p_{\xi+\delta\xi})
=\tfrac12g_{ij}(\xi)\delta\xi^i\delta\xi^j+o(\|\delta\xi\|^2),
$$

where $D(p\|q)=\mathbb E_p\log(p/q)$ and
$g_{ij}=\mathbb E_\xi[s_{\xi,i}s_{\xi,j}]$. If the Taylor remainder
of $\zeta\mapsto\mathbb E_\xi\ell_\zeta$ is $O(\|\zeta-\xi\|^3)$,
the displayed remainder is $O(\|\delta\xi\|^3)$.
The Hessian in the second argument at equality is therefore the Fisher metric.
:::

:::{prf:proof}
Use the direction
$D(p_\xi\|p_\zeta)=\mathbb E_\xi[\ell_\xi(X)-\ell_\zeta(X)]$, holding the first
law fixed. Differentiating normalization twice yields

```{math}
\begin{aligned}
0&=\mathbb E_\xi[\partial_i\partial_j\ell_\xi(X)+s_{\xi,i}(X)s_{\xi,j}(X)],\\
\mathbb E_\xi[\partial_i\partial_j\ell_\xi(X)]&=-g_{ij}.
\end{aligned}
```

Under the stated bound on the third-order Taylor remainder, expansion gives

```{math}
:label: ig-kl-diagonal-hessian
D(p_\xi\|p_{\xi+\delta\xi})
=\frac12 g_{ij}(\xi)\,\delta\xi^i\delta\xi^j
 +O(\lVert\delta\xi\rVert^3).
```

The zero expected score removes the linear term. The Hessian in the second
argument at equality is therefore $g_{ij}$; because the first derivative
vanishes there, this Hessian transforms tensorially. With only a second-order
Taylor expansion, the remainder is $o(\lVert\delta\xi\rVert^2)$, as in the
[Fisher-versus-$L^2$ note](./information-geometry-fisher-vs-l2.md#local-kl-divergence-gives-the-same-quadratic-form).
:::

Finite KL remains asymmetric and is not half the squared Fisher–Rao distance.
An integrable log density at $\xi$, together with locally bounded integrable
third derivatives of the log density under $P_\xi$, suffices for the cubic
remainder under the preceding regularity assumptions.

:::{prf:proposition} KL divergence between two normal laws
:label: prop-euclidean-to-manifold-normal-kl

For $\mu,\widetilde\mu\in\mathbb R$ and $\sigma,\widetilde\sigma>0$,

$$
D\!\left(N(\mu,\sigma^2)\middle\|N(\widetilde\mu,\widetilde\sigma^2)\right)
=\log\frac{\widetilde\sigma}{\sigma}-\frac12
 +\frac{\sigma^2+(\mu-\widetilde\mu)^2}{2\widetilde\sigma^2}.
$$

For a fixed velocity $v=(v^\mu,v^\sigma)$, replacing the second law's
parameters by $(\mu+\varepsilon v^\mu,\sigma+\varepsilon v^\sigma)$
gives $\varepsilon^2((v^\mu)^2+2(v^\sigma)^2)/(2\sigma^2)+O(\varepsilon^3)$
as $\varepsilon\to0$.
:::

:::{prf:proof}
Subtract the two log densities and take expectation under
$N(\mu,\sigma^2)$. Use $\mathbb E(X-\mu)^2=\sigma^2$ and
$\mathbb E(X-\widetilde\mu)^2=\sigma^2+(\mu-\widetilde\mu)^2$ to obtain

```{math}
D\!\left(N(\mu,\sigma^2)\middle\|N(\widetilde\mu,\widetilde\sigma^2)\right)
=\log\frac{\widetilde\sigma}{\sigma}-\frac12
 +\frac{\sigma^2+(\mu-\widetilde\mu)^2}{2\widetilde\sigma^2}.
```

Putting $(\widetilde\mu,\widetilde\sigma)
=(\mu+\varepsilon v^\mu,\sigma+\varepsilon v^\sigma)$ for a fixed velocity
$v$ at $\sigma>0$ gives
$\varepsilon^2((v^\mu)^2+2(v^\sigma)^2)/(2\sigma^2)+O(\varepsilon^3)$, in agreement with
[](#ig-normal-fisher-metric).
:::

### Coarse-graining, monotonicity, and sufficiency

:::{prf:lemma} Scores after a parameter-independent kernel
:label: lem-euclidean-to-manifold-score-conditioning

Let $X\sim P_\xi$ have positive smooth densities on a fixed support and
square-integrable score column $s^X_\xi$. Pass $X$ through a fixed Markov
kernel $K(dy\mid x)$ to obtain $Y$. Suppose the output has smooth densities
on a common support and differentiation can pass through the kernel integral.
Then its score satisfies
$s^Y_\xi(Y)=\mathbb E_\xi[s^X_\xi(X)\mid Y]$ almost surely.
:::

:::{prf:proof}
Let an observation $X$ pass through a parameter-independent Markov kernel
$K(dy\mid x)$, giving $Y$. Assume the transformed model admits densities and differentiation under
the kernel integral, and that the original scores are square integrable.
Denote the two score covectors by their component columns $s^X_\xi$ and
$s^Y_\xi$, with components $s^X_{\xi,i}$ and $s^Y_{\xi,i}$. Differentiating
the marginal law gives, for bounded measurable $b$,
$\partial_i\mathbb E_\xi b(Y)=\mathbb E_\xi[b(Y)s^X_{\xi,i}(X)]$.
Computing the same derivative from the output density gives
$\mathbb E_\xi[b(Y)s^Y_{\xi,i}(Y)]$. The defining property of conditional
expectation therefore gives

```{math}
:label: ig-score-under-statistic
s^Y_{\xi,i}(Y)=\mathbb E_\xi[s^X_{\xi,i}(X)\mid Y].
```
:::

A deterministic statistic $Y=S(X)$ is a special case of the kernel construction.

:::{prf:corollary} Information loss and equality under a kernel
:label: cor-euclidean-to-manifold-information-monotonicity

Under the hypotheses of [](#lem-euclidean-to-manifold-score-conditioning),
let $I_X=\mathbb E_\xi[s^X_\xi(s^X_\xi)^{\mathsf T}]$ and define $I_Y$
similarly. At every parameter and for every tangent column $v$,

$$
I_X-I_Y=\mathbb E_\xi[\operatorname{Cov}_\xi(s^X_\xi\mid Y)]\succeq0,
\qquad
v^{\mathsf T}(I_X-I_Y)v
=\mathbb E_\xi\operatorname{Var}_\xi(v^{\mathsf T}s^X_\xi\mid Y).
$$

Equality for $v$ holds exactly when $v^{\mathsf T}s^X_\xi(X)$ is determined
by $Y$ almost surely. Matrix equality holds exactly when every score
component is determined by $Y$ almost surely. The output matrix may be singular.
:::

:::{prf:proof}
By [](#lem-euclidean-to-manifold-score-conditioning), the output score is
the conditional expectation of the input score. Orthogonality of conditional
expectation, applied componentwise and to $v^{\mathsf T}s^X_\xi$, gives

```{math}
:label: ig-fisher-monotonicity
\begin{aligned}
g^X_\xi(v,v)-g^Y_\xi(v,v)
 &=\mathbb E_\xi\!\left[
   \operatorname{Var}_\xi(v^i s^X_{\xi,i}(X)\mid Y)\right]\geq0,\\
I_X(\xi)-I_Y(\xi)
 &=\mathbb E_\xi[\operatorname{Cov}_\xi(s^X_\xi(X)\mid Y)]\succeq0.
\end{aligned}
```
The expected conditional variance is zero exactly when the directional score
equals its conditional expectation almost surely; applying this to each
coordinate gives the matrix equality criterion.
:::

The superscripts $X,Y$ identify experiments; they are not tensor indices.
This is a statement about first-order information at that point.

A statistic is **sufficient for the family** if the conditional distribution
of $X$ given the statistic can be chosen independently of the parameter.
Equivalently in the regular dominated setting, the density factors into a
parameter-dependent function of the statistic and a parameter-independent
factor.

:::{prf:proposition} Sufficiency preserves Fisher information
:label: prop-euclidean-to-manifold-sufficient-information

In a regular dominated model satisfying
[](#lem-euclidean-to-manifold-score-conditioning) for $Y=S(X)$, suppose
$p_\xi(x)=a_\xi(S(x))b(x)$ is a sufficient factorization, with
$a_\xi>0$ differentiable in $\xi$ and $b$ parameter-independent.
Then $I_S(\xi)=I_X(\xi)$ throughout the parameter domain.
:::

:::{prf:proof}
Differentiating the log factorization gives
$s^X_{\xi,i}(x)=\partial_i\log a_\xi(S(x))$, a function of the statistic.
Apply the equality criterion in [](#cor-euclidean-to-manifold-information-monotonicity).
:::

:::{prf:proposition} Information preservation characterizes finite-model sufficiency
:label: prop-euclidean-to-manifold-finite-sufficiency

Let $X$ take values in a finite set, with strictly positive smooth probabilities
$p_\xi(x)$ for $\xi$ in a connected open subset of $\mathbb R^d$.
For a fixed statistic $S$, $I_S(\xi)=I_X(\xi)$ at every parameter if and
only if the conditional law of $X$ given $S$ is parameter-independent.
:::

:::{prf:proof}
For a precise converse, take a finite sample space with strictly positive
smooth probabilities on a connected open parameter domain, and a fixed
statistic $S$. Equality of Fisher information at **every** parameter is then
equivalent to sufficiency. To prove the converse, equality in
[](#ig-fisher-monotonicity) gives, for every outcome,

```{math}
\partial_i\log P_\xi(X=x\mid S(X)=S(x))
=s^X_{\xi,i}(x)-s^S_{\xi,i}(S(x))=0.
```

Connectedness makes each conditional probability constant in the parameter,
which proves sufficiency. Conversely, factor the finite positive joint
probabilities into the marginal probability of $S$ and the parameter-independent
conditional probability, then apply [](#prop-euclidean-to-manifold-sufficient-information).
:::

Equality at one parameter, or in one direction,
is insufficient for this conclusion. Extensions beyond this finite positive
setting need corresponding regularity assumptions; see
[Ay, Jost, Lê, and Schwachhöfer on sufficient statistics](https://arxiv.org/abs/1207.6736).

:::{prf:proposition} A sufficient normal sample statistic and the loss from signs
:label: prop-euclidean-to-manifold-normal-coarsening

For $n\ge1$ independent $N(\mu,\sigma^2)$ observations, $\sigma>0$,
$S=(\sum_{a=1}^nX_a,\sum_{a=1}^nX_a^2)$ is sufficient and has information
$nI(\mu,\sigma)$, where $I=\operatorname{diag}(\sigma^{-2},2\sigma^{-2})$.
For one observation, $Y=\mathbf1_{\{X>0\}}$ instead has, at $\mu=0$,
$I_Y(0,\sigma)=\operatorname{diag}(2/(\pi\sigma^2),0)$.
:::

:::{prf:proof}
The normal family illustrates both preservation and loss. For independent
observations $X_1,\ldots,X_n$, the statistic
$S=(\sum_a X_a,\sum_a X_a^2)$ is sufficient by the factorization in
[](#ig-normal-natural-chart). Its information equals that of the full sample,
namely $nI(\xi)$: independent zero-mean scores add and their cross-covariances
vanish. By contrast, retaining only $Y=\mathbf 1_{\{X>0\}}$ gives a Bernoulli
probability $q=\Phi(\mu/\sigma)$, where $\Phi$ is the standard-normal cdf.
The Bernoulli score is $(Y-q)\nabla q/[q(1-q)]$, whose second moment
is $(\nabla q)(\nabla q)^{\mathsf T}/[q(1-q)]$.
At $\mu=0$, $q=1/2$, $\partial_\mu q=1/(\sqrt{2\pi}\sigma)$ and
$\partial_\sigma q=0$, so

```{math}
I_Y(0,\sigma)=\begin{pmatrix}2/(\pi\sigma^2)&0\\0&0\end{pmatrix}.
```
:::

The sign loses all local scale information and retains only a fraction
$2/\pi$ of the location information there.

### Čencov's finite-space uniqueness theorem

The theorem concerns a compatible choice of metrics on **all** finite
probability simplices, not an arbitrary metric on one fixed normal family.
Let

```{math}
\Delta_{n-1}^{\circ}
=\left\{p\in\mathbb R^n:p(a)>0,\ \sum_{a=1}^n p(a)=1\right\},
\qquad
T_p\Delta_{n-1}^{\circ}
=\left\{u\in\mathbb R^n:\sum_{a=1}^n u(a)=0\right\}.
```

A Markov map is a stochastic linear map
$(Kp)(b)=\sum_a K(b\mid a)p(a)$. A **congruent Markov embedding** has a
Markov left inverse $L$ with $LK=\mathrm{id}$ and maps the interior into the
interior. It splits each input outcome into a disjoint block of output
outcomes with fixed conditional probabilities; the block label recovers the
input. Such a splitting preserves the statistical experiment.

Čencov's theorem states that if smooth Riemannian metrics $h^{(n)}$ on
$\Delta_{n-1}^{\circ}$, for every $n\geq2$, make every congruent Markov
embedding an isometry, then there is a single constant $c>0$, independent of
$n$ and $p$, such that

```{math}
:label: ig-chentsov-finite
h^{(n)}_p(u,w)=c\sum_{a=1}^n\frac{u(a)w(a)}{p(a)}.
```

Conversely, these metrics have that invariance. We cite this classification,
not prove it: see Čencov's
[*Statistical Decision Rules and Optimal Inference* (1982)](https://www.ams.org/books/mmono/053/)
and the explicit finite-simplex formulation in Theorem 2 of
[Montúfar, Rauh, and Ay (2014)](https://www.mdpi.com/1099-4300/16/6/3207).

:::{prf:corollary} Pullback and reversible Markov embeddings
:label: cor-euclidean-to-manifold-finite-pullback

For a regular positive smooth finite model $p_\xi(a)$, the pullback of the simplex
form $h_p(u,w)=c\sum_a u(a)w(a)/p(a)$, $c>0$, is $c$ times the Fisher
form. If a family of metrics is monotone under Markov maps and $K$ has
a Markov left inverse $L$ with $LK=\mathrm{id}$, with both maps acting
on the relevant simplex interiors, then $K$ is an isometry for that family.
:::

:::{prf:proof}
Pulling [](#ig-chentsov-finite) back along a regular finite statistical model,
with $u(a)=\partial_i p_\xi(a)v^i=p_\xi(a)s_{\xi,i}(a)v^i$, gives $cg_\xi(v,w)$.
For the second assertion, monotonicity in both directions gives
$h_p(v,v)\ge h_{Kp}(Kv,Kv)\ge h_{LKp}(LKv,LKv)=h_p(v,v)$.
Thus the squared lengths agree; polarization gives equality of the bilinear
forms. This explains why monotonicity implies the embedding invariance in
the cited Čencov classification.
:::

We choose $c=1$ for information per observation. General Markov maps are
contractions by [](#ig-fisher-monotonicity), whereas sufficient transformations
preserve information. Requiring invariance under *every* Markov map would be
wrong: a kernel can discard the observation completely.

For general sample spaces, @ay2017information
develop invariance and uniqueness results for suitable integrable models and
continuous local tensor fields. A different extension by
[Bauer–Bruveris–Michor (2016)](https://arxiv.org/abs/1411.5577)
characterizes the Fisher metric among smooth weak Riemannian metrics invariant
under observation-space diffeomorphisms on smooth positive probability
densities over a closed manifold of dimension greater than one. Neither
statement is an unrestricted uniqueness theorem for every continuous or
singular statistical model; in particular, that latter theorem does not
directly cover densities on $\mathbb R$.

### Density $L^2$ and square-root densities

:::{prf:proposition} Raw density distances depend on observation scale
:label: prop-euclidean-to-manifold-raw-density

Let $p_X,q_X$ be Lebesgue densities on $\mathbb R$ with $p_X-q_X\in L^2$.
For $Y=aX$, $a>0$, let $p_Y,q_Y$ be the corresponding transformed densities.
Then $\int(p_Y-q_Y)^2\,dy=a^{-1}\int(p_X-q_X)^2\,dx$.
For a parameterized model differentiable into $L^2(\nu)$, the form
$g^{\mathrm{raw}}_{ij}=\int(\partial_i p_\xi)(\partial_j p_\xi)\,d\nu$
nevertheless obeys the covariant metric transformation law under smooth
invertible parameter changes.
:::

:::{prf:proof}
Use $p_Y(y)=a^{-1}p_X(y/a)$, and likewise for $q_Y$, then substitute
$y=ax$ in the integral. For parameter changes,
$\partial_{\phi^a}p=B^i{}_a\partial_{\xi^i}p$; inserting this in the
inner product gives $g'_{ab}=B^i{}_aB^j{}_bg_{ij}$.
:::

The raw density metric, when finite,
$g^{\mathrm{raw}}_{ij}=\int(\partial_i p_\xi)(\partial_j p_\xi)\,d\nu$, also
transforms correctly under parameter changes. It depends on the reference
measure and the measurement coordinate: for Lebesgue densities and $Y=aX$,
$a>0$, $\int(p_Y-q_Y)^2\,dy=a^{-1}\int(p_X-q_X)^2\,dx$.
It therefore differs from Fisher geometry even though both constructions are
covariant under reparameterization.

:::{prf:proposition} Square-root densities induce Fisher geometry
:label: prop-euclidean-to-manifold-square-root

Let $p_\xi>0$ be smooth densities relative to a fixed measure $\nu$, with
square-integrable scores $s_{\xi,i}=\partial_i\log p_\xi$.
Assume $\xi\mapsto2\sqrt{p_\xi}$ is differentiable into $L^2(\nu)$.
Its derivative is $\partial_i(2\sqrt{p_\xi})=\sqrt{p_\xi}s_{\xi,i}$.
Its pullback of the $L^2$ inner product is
$g_{ij}=\mathbb E_\xi[s_{\xi,i}s_{\xi,j}]$. For
$H^2(p,q)=\tfrac12\int(\sqrt p-\sqrt q)^2\,d\nu$,

$$
H^2(p_\xi,p_{\xi+\delta\xi})
=\tfrac18g_{ij}(\xi)\delta\xi^i\delta\xi^j+o(\|\delta\xi\|^2).
$$
:::

:::{prf:proof}
Assuming differentiability into $L^2(\nu)$, the map $p\mapsto2\sqrt p$
has derivative $\partial_i(2\sqrt{p_\xi})=\sqrt{p_\xi}\,s_{\xi,i}$.
Taking inner products of the derivatives gives
$\int p_\xi s_{\xi,i}s_{\xi,j}\,d\nu=g_{ij}$.
Differentiability gives an $L^2$ remainder $o(\|\delta\xi\|)$; squaring
the norm of the linear term plus this remainder gives the quadratic
expansion. With
$H^2(p,q)=\tfrac12\int(\sqrt p-\sqrt q)^2\,d\nu$, this gives
$H^2(p_\xi,p_{\xi+\delta\xi})=\tfrac18g_{ij}\delta\xi^i\delta\xi^j
+o(\lVert\delta\xi\rVert^2)$.
:::

The [Fisher-versus-$L^2$ note](./information-geometry-fisher-vs-l2.md#three-different-uses-of-l2)
explains the distinct roles of parameter norms, scores in $L^2(P_\xi)$, and
densities in $L^2(\nu)$. A square-root embedding preserves tangent lengths;
the distance between two normals still minimizes paths *within the normal
family*, rather than over the whole sphere of square-root densities.

## Connections differentiate vector fields

The ordinary derivative of vector components depends on the chosen chart.
For example, a field with constant scale component has variance component
$2\sigma$ times as large, which changes along a scale curve. A connection
supplies the correction that makes differentiation independent of coordinates.
It is extra geometric structure; a metric singles out one connection only
when we impose the conditions below.

### Axioms and Christoffel symbols

An **affine connection** on $M$ is an $\mathbb R$-bilinear operation
$(U,V)\mapsto\nabla_UV$ on smooth vector fields satisfying, for every smooth
function $a$,

```{math}
:label: ig-affine-connection-axioms
\nabla_{aU}V=a\nabla_UV,\qquad
\nabla_U(aV)=U[a]V+a\nabla_UV.
```

Thus the lower argument $U$ is linear over functions; in the differentiated
argument $V$, the Leibniz term is essential. At a point, $\nabla_UV$ depends
only on the value of $U$ and on the first-order behavior of $V$ nearby.

Define the Christoffel symbols in a coordinate basis by

```{math}
\nabla_{\partial_i}\partial_j=\Gamma_{ij}^k\partial_k.
```

:::{prf:lemma} Coordinate form of a covariant derivative
:label: lem-euclidean-to-manifold-connection-coordinates

For an affine connection with $\nabla_{\partial_i}\partial_j=
\Gamma_{ij}^k\partial_k$ and smooth fields $U=U^i\partial_i$,
$V=V^j\partial_j$,
$\nabla_UV=U^i(\partial_iV^k+\Gamma_{ij}^kV^j)\partial_k$.
Conversely, any smooth coefficients $\Gamma_{ij}^k$ define an affine
connection on the chart by this formula.
:::

:::{prf:proof}
Expanding $U=U^i\partial_i$ and $V=V^j\partial_j$ with the axioms gives

```{math}
:label: ig-covariant-derivative-coordinates
\nabla_UV=U^i\bigl(\partial_iV^k+\Gamma_{ij}^kV^j\bigr)\partial_k.
```
Conversely, the displayed expression is bilinear over $\mathbb R$,
linear over smooth functions in $U$, and obeys the Leibniz rule in $V$,
so it defines a connection.
:::

The second term accounts for the change of the coordinate basis under the
connection. Prescribing smooth $\Gamma_{ij}^k$ defines a connection on a
chart; prescriptions on overlapping charts must satisfy the next law.

### Why Christoffel symbols are not a tensor

:::{prf:proposition} Transformation law for Christoffel symbols
:label: prop-euclidean-to-manifold-connection-transformation

For a smooth chart transition $\phi=f(\xi)$, let
$A^a{}_i=\partial\phi^a/\partial\xi^i$ and
$B^i{}_a=\partial\xi^i/\partial\phi^a$. The coefficients of any affine
connection satisfy

$$
\Gamma'{}^c_{ab}=A^c{}_kB^i{}_aB^j{}_b\Gamma_{ij}^k
 +A^c{}_k\frac{\partial^2\xi^k}{\partial\phi^a\partial\phi^b}.
$$
:::

:::{prf:proof}
Use $\phi=f(\xi)$, $A^a{}_i=\partial\phi^a/\partial\xi^i$, and
$B^i{}_a=\partial\xi^i/\partial\phi^a$ as before. Apply the connection
axioms to $\nabla_{B^i{}_a\partial_i}(B^j{}_b\partial_j)$ to obtain

```{math}
:label: ig-christoffel-transformation
\Gamma'{}^c_{ab}
=A^c{}_k B^i{}_a B^j{}_b\Gamma_{ij}^k
 +A^c{}_k\frac{\partial^2\xi^k}{\partial\phi^a\partial\phi^b}.
```
:::

A $(1,2)$-tensor has only the first term. The inhomogeneous second derivative
is why coefficients that vanish in one chart need not vanish in another.
For example, take the connection with $\Gamma_{ij}^k=0$ in the normal
location-scale chart. In the variance chart, $\sigma=\sqrt\tau$, so it has

```{math}
\Gamma'{}^{\tau}_{\tau\tau}
=2\sigma\frac{d^2\sqrt\tau}{d\tau^2}=-\frac1{2\tau}.
```

This is a coordinate-flat connection chosen on the parameter domain; it is
not yet the Fisher Levi-Civita connection. The example isolates the effect of
the second term in [](#ig-christoffel-transformation).

:::{prf:corollary} The difference of connections is a tensor
:label: cor-euclidean-to-manifold-connection-difference

For two affine connections $\nabla,\widetilde\nabla$ on the same smooth
manifold, $(U,V)\mapsto\nabla_UV-\widetilde\nabla_UV$ is a smooth
$(1,2)$-tensor: it is linear over smooth functions in both arguments.
:::

:::{prf:proof}
Subtracting the transformation laws in
[](#prop-euclidean-to-manifold-connection-transformation) cancels their
inhomogeneous terms, giving the $(1,2)$-tensor transformation law.
Equivalently, the Leibniz terms cancel in
$\nabla_UV-\widetilde\nabla_UV$, leaving an expression linear over functions
in both $U$ and $V$.
:::

We will use this fact to compare statistical connections.

### Along a curve: parallel transport and geodesics

For a curve $c(\lambda)$, let $V(\lambda)=V^k(\lambda)\partial_k|_{c(\lambda)}$
be a vector field along the curve. It need not extend to a single vector field
on all of $M$. Its covariant derivative is

```{math}
:label: ig-curve-covariant-derivative
\frac{DV}{d\lambda}:=\nabla_{\dot c}V
=\left(\frac{dV^k}{d\lambda}
 +\Gamma_{ij}^k(\xi(c(\lambda)))\dot\xi^i V^j\right)\partial_k.
```

:::{prf:proposition} Parallel transport along a fixed curve
:label: prop-euclidean-to-manifold-parallel-transport

Let $\nabla$ be a smooth affine connection and let $c:[a,b]\to M$ be a
smooth curve. The coordinate expression
$(DV/d\lambda)^k=\dot V^k+\Gamma_{ij}^k(c)\dot\xi^iV^j$
defines a coordinate-independent derivative of fields along $c$.
For each initial vector at $c(a)$, the equation $DV/d\lambda=0$ has a
unique solution on $[a,b]$. Its endpoint map is a linear isomorphism
$T_{c(a)}M\to T_{c(b)}M$, inverted by transport along the reversed curve.
:::

:::{prf:proof}
Differentiate $V'^a=A^a{}_i(c)V^i$ along the curve and insert
[](#prop-euclidean-to-manifold-connection-transformation). The derivatives
of the transition Jacobians cancel, leaving $(DV/d\lambda)'=A(DV/d\lambda)$,
so the derivative is coordinate-independent. A field is
**parallel** when $DV/d\lambda=0$. Smooth coefficients give existence and uniqueness
for linear ODEs on each compact chart segment; patch finitely many such
segments along the curve. Linearity follows from the homogeneous equation,
and uniqueness identifies reversed transport as its inverse.
:::

Transport generally depends on the path, not just its endpoints.

An affinely parameterized **geodesic** has parallel velocity:

```{math}
:label: ig-geodesic-equation
\nabla_{\dot c}\dot c=0,
\qquad
\ddot\xi^k+\Gamma_{ij}^k\dot\xi^i\dot\xi^j=0.
```

An arbitrary nonlinear reparameterization need not preserve this equation;
affine changes of the parameter do. A connection defines geodesics without
requiring a metric or a length-minimization interpretation.

### Torsion, compatibility, and the Levi-Civita connection

The Lie bracket is the vector field
$[U,V][F]=U[V[F]]-V[U[F]]$. In coordinates,
$[U,V]^k=U^i\partial_iV^k-V^i\partial_iU^k$.
The **torsion** of a connection is

```{math}
T(U,V)=\nabla_UV-\nabla_VU-[U,V],\qquad
T(\partial_i,\partial_j)
=(\Gamma_{ij}^k-\Gamma_{ji}^k)\partial_k.
```

:::{prf:proposition} Torsion is a tensor
:label: prop-euclidean-to-manifold-torsion

For an affine connection, $T(U,V)=\nabla_UV-\nabla_VU-[U,V]$ is a
$(1,2)$-tensor. It vanishes exactly when $\Gamma_{ij}^k=\Gamma_{ji}^k$
in a coordinate basis.
:::

:::{prf:proof}
In coordinates, the derivative terms in the two covariant derivatives
cancel those in the Lie bracket, leaving
$T(U,V)^k=(\Gamma_{ij}^k-\Gamma_{ji}^k)U^iV^j$.
This is linear over functions in both fields and gives the vanishing criterion.
:::

A connection is **metric-compatible**, written $\nabla g=0$, when

```{math}
:label: ig-metric-compatibility
U[g(V,W)]=g(\nabla_UV,W)+g(V,\nabla_UW).
```

:::{prf:proposition} Metric compatibility and preservation under transport
:label: prop-euclidean-to-manifold-metric-transport

An affine connection is compatible with a Riemannian metric $g$ if and only if
$\partial_i g_{jk}=\Gamma_{ij}^{\ell}g_{\ell k}
+\Gamma_{ik}^{\ell}g_{j\ell}$ in every chart. For such a connection,
parallel fields $V,W$ along the same smooth curve have constant $g(V,W)$,
and every affinely parameterized geodesic has constant speed.
:::

:::{prf:proof}
Apply the compatibility identity to the coordinate fields and expand
general fields by the product rule. This proves the equivalence with
$\partial_i g_{jk}=\Gamma_{ij}^{\ell}g_{\ell k}
+\Gamma_{ik}^{\ell}g_{j\ell}$. Along a curve this product rule shows that
parallel transport preserves inner products. In particular, a geodesic for
a metric-compatible connection has constant speed.
:::

:::{prf:theorem} Existence and uniqueness of the Levi-Civita connection
:label: thm-euclidean-to-manifold-levi-civita

Every smooth Riemannian metric $g$ on a smooth manifold has a unique
torsion-free, metric-compatible affine connection $\nabla^{\mathrm{LC}}$.
For all smooth fields $U,V,W$, it is determined by the Koszul formula:

```{math}
:label: ig-koszul-formula
\begin{aligned}
2g(\nabla^{\mathrm{LC}}_UV,W)
={}&U[g(V,W)]+V[g(W,U)]-W[g(U,V)]\\
 &+g([U,V],W)-g([V,W],U)+g([W,U],V).
\end{aligned}
```

Its coordinate coefficients are
$\Gamma^{(0)k}_{ij}=\tfrac12g^{k\ell}
(\partial_i g_{j\ell}+\partial_j g_{i\ell}-\partial_\ell g_{ij})$.
:::

:::{prf:proof}
To see why this determines the connection, write the compatibility identity
for the three cyclic choices of $U,V,W$, add the first two and subtract the
third, and replace $\nabla_UV-\nabla_VU$ by $[U,V]$. This gives
[](#ig-koszul-formula). Nondegeneracy of $g$ then uniquely determines
$\nabla^{\mathrm{LC}}_UV$. For existence, the right side is linear over
functions in $W$, so it defines a vector field by nondegeneracy of $g$.
The product rule gives the connection axioms; subtracting the expression with
$U,V$ exchanged gives torsion-freeness, and adding the expressions with $V,W$
exchanged gives metric compatibility. In a coordinate basis the brackets
vanish, yielding the computable formula

```{math}
:label: ig-levi-civita-christoffels
\Gamma^{(0)k}_{ij}
=\frac12g^{k\ell}
 \left(\partial_i g_{j\ell}+\partial_j g_{i\ell}
             -\partial_\ell g_{ij}\right).
```

:::

We write $(0)$ for Levi-Civita because it will be the $\alpha=0$ statistical
connection. We cite, without proving here, the standard Riemannian result
that its geodesics minimize
length on sufficiently short segments. It does not say every geodesic segment
is globally minimizing on every manifold. For background on this result and
the connection framework, see §2 of
[Nielsen's 2020 introduction](https://arxiv.org/abs/1808.08271).

## Levi-Civita geometry of the normal family

### Christoffel symbols and parallel transport

:::{prf:proposition} Normal Levi-Civita coefficients and fixed-mean transport
:label: prop-euclidean-to-manifold-normal-lc

For the normal Fisher metric $ds^2=(d\mu^2+2d\sigma^2)/\sigma^2$ on
$\mu\in\mathbb R$, $\sigma>0$, the only nonzero Levi-Civita coefficients are

$$
\Gamma^{(0)\mu}_{\mu\sigma}=\Gamma^{(0)\mu}_{\sigma\mu}=-1/\sigma,
\qquad\Gamma^{(0)\sigma}_{\mu\mu}=1/(2\sigma),
\qquad\Gamma^{(0)\sigma}_{\sigma\sigma}=-1/\sigma.
$$

Along a smooth fixed-mean curve from scale $\sigma_0$ to $\sigma_1$,
parallel transport multiplies both coordinate components by $\sigma_1/\sigma_0$
and preserves Fisher inner products.
:::

:::{prf:proof}
Substituting [](#ig-normal-fisher-metric) into
[](#ig-levi-civita-christoffels) gives exactly these nonzero coefficients:

```{math}
:label: ig-normal-lc-christoffels
\Gamma^{(0)\mu}_{\mu\sigma}
=\Gamma^{(0)\mu}_{\sigma\mu}=-\frac1\sigma,
\qquad
\Gamma^{(0)\sigma}_{\mu\mu}=\frac1{2\sigma},
\qquad
\Gamma^{(0)\sigma}_{\sigma\sigma}=-\frac1\sigma.
```

For example,
$\Gamma^{(0)\sigma}_{\mu\mu}
=-\tfrac12 g^{\sigma\sigma}\partial_\sigma g_{\mu\mu}
=-\tfrac12(\sigma^2/2)(-2/\sigma^3)=1/(2\sigma)$.
Along any fixed-mean curve $c(\lambda)=(\mu_0,\sigma(\lambda))$, the parallel
equations reduce to

```{math}
\dot V^\mu-\frac{\dot\sigma}{\sigma}V^\mu=0,
\qquad
\dot V^\sigma-\frac{\dot\sigma}{\sigma}V^\sigma=0.
```

Thus a vector transported from scale $\sigma_0$ to scale $\sigma_1$ has
components multiplied by $\sigma_1/\sigma_0$. Its Fisher squared length
$( (V^\mu)^2+2(V^\sigma)^2 )/\sigma^2$ stays fixed.
:::

This coefficient already shows why a horizontal line with constant scale
and changing mean is not a Levi-Civita geodesic in the full family.

Constant coordinate components would not give parallel transport along
a changing-scale curve.

### Hyperbolic curvature and geodesics

:::{prf:proposition} Curvature and geodesics of the normal Fisher metric
:label: prop-euclidean-to-manifold-normal-hyperbolic

On the full normal family with Fisher metric, put $u=\mu/\sqrt2$.
Then $ds^2=2(du^2+d\sigma^2)/\sigma^2$, and the Gaussian curvature is
$-1/2$, using $R(U,V)W=\nabla_U\nabla_VW-\nabla_V\nabla_UW-\nabla_{[U,V]}W$.
The affinely parameterized Levi-Civita geodesics satisfy

$$
\ddot\mu-2\dot\mu\dot\sigma/\sigma=0,\qquad
\ddot\sigma+\dot\mu^2/(2\sigma)-\dot\sigma^2/\sigma=0.
$$

They are precisely the vertical solutions
$\mu=c$, $\sigma(\lambda)=\sigma(0)e^{b\lambda}$ with
$\sigma(0)>0$, $b\in\mathbb R$, and the nonvertical solutions

$$
\mu(\lambda)=c+\sqrt2 R\tanh(b\lambda+a),\qquad
\sigma(\lambda)=R\operatorname{sech}(b\lambda+a),
\quad c,a\in\mathbb R,\ R>0,\ b\ne0.
$$

Both forms have constant squared speed $2b^2$.
:::

:::{prf:proof}
Make the local coordinate substitution $\mu=\sqrt2\,u$. Then

```{math}
:label: ig-normal-hyperbolic-metric
ds^2=2\frac{du^2+d\sigma^2}{\sigma^2},\qquad \sigma>0.
```

This is twice the Poincaré upper-half-plane metric. Its curvature is $-1/2$,
not $-1$: multiplying a metric by $2$ multiplies lengths by $\sqrt2$ and
divides sectional curvature by $2$.

We can verify the curvature directly from the connection. With the convention

```{math}
R(U,V)W=\nabla_U\nabla_VW-\nabla_V\nabla_UW-\nabla_{[U,V]}W,
```

[](#ig-normal-lc-christoffels) gives
$R(\partial_\mu,\partial_\sigma)\partial_\sigma
=-\sigma^{-2}\partial_\mu$. Hence the Gaussian curvature is

```{math}
K=\frac{g(R(\partial_\mu,\partial_\sigma)\partial_\sigma,\partial_\mu)}
 {g_{\mu\mu}g_{\sigma\sigma}-g_{\mu\sigma}^2}
 =\frac{-\sigma^{-4}}{2\sigma^{-4}}=-\frac12.
```

The geodesic equations are

```{math}
:label: ig-normal-lc-geodesic-odes
\ddot\mu-2\frac{\dot\mu\dot\sigma}{\sigma}=0,
\qquad
\ddot\sigma+\frac{\dot\mu^2}{2\sigma}
                 -\frac{\dot\sigma^2}{\sigma}=0.
```

The first equation integrates to $\dot\mu/\sigma^2=\text{constant}$.
When this constant is zero, the solutions are vertical lines,
$\mu(\lambda)=c$, $\sigma(\lambda)=\sigma(0)e^{b\lambda}$, with
$\sigma(0)>0$ and $b\in\mathbb R$. Otherwise, in the rescaled coordinate $u$,
the solutions trace semicircles orthogonal to the boundary
$\sigma=0$. An affine parameterization in the original chart is

```{math}
:label: ig-normal-lc-geodesic-solutions
\begin{aligned}
\mu(\lambda)&=c+\sqrt2\,R\tanh(b\lambda+a),\\
\sigma(\lambda)&=R\operatorname{sech}(b\lambda+a),\qquad R>0,\quad b\ne0,\\
\frac{(\mu-c)^2}{2}+\sigma^2&=R^2.
\end{aligned}
```

Substitution proves that these curves solve both equations and have constant
squared speed $2b^2$. They exhaust the nonvertical initial conditions: at
any point with $\dot\mu\ne0$, choose
$c=\mu+2\sigma\dot\sigma/\dot\mu$, then choose $R,a,b$ to match the point
and velocity. Local uniqueness for the geodesic ODE identifies the solution.
The images are half-ellipses in $(\mu,\sigma)$ and semicircles in
$(u,\sigma)$; their parameterizations are not linear in the angle.
:::

:::{prf:corollary} Geodesic segments joining normal parameters
:label: cor-euclidean-to-manifold-normal-geodesic-endpoints

For endpoints $(\mu_0,\sigma_0)$, $(\mu_1,\sigma_1)$ with positive scales
and $\mu_0\ne\mu_1$, put

$$
c=\frac{\mu_1^2-\mu_0^2+2(\sigma_1^2-\sigma_0^2)}{2(\mu_1-\mu_0)},
\quad R^2=\frac{(\mu_0-c)^2}{2}+\sigma_0^2,
$$

and set $a=\operatorname{artanh}((\mu_0-c)/(\sqrt2R))$,
$b=\operatorname{artanh}((\mu_1-c)/(\sqrt2R))-a$.
The nonvertical geodesic in [](#prop-euclidean-to-manifold-normal-hyperbolic)
then joins these endpoints over $0\le\lambda\le1$.
For $\mu_0=\mu_1$, the corresponding geodesic is
$\mu=\mu_0$, $\sigma(\lambda)=\sigma_0^{1-\lambda}\sigma_1^\lambda$.
:::

:::{prf:proof}
Use the half-ellipse equation in
[](#prop-euclidean-to-manifold-normal-hyperbolic) at the two endpoints.
Subtracting the equations determines $c$; either equation then determines
$R>0$. Thus the center and radius are

```{math}
c=\frac{\mu_1^2-\mu_0^2+2(\sigma_1^2-\sigma_0^2)}{2(\mu_1-\mu_0)},
\qquad
R^2=\frac{(\mu_0-c)^2}{2}+\sigma_0^2.
```

Set
$a=\operatorname{artanh}((\mu_0-c)/(\sqrt2R))$ and
$b=\operatorname{artanh}((\mu_1-c)/(\sqrt2R))-a$ in
[](#ig-normal-lc-geodesic-solutions) to join them over $0\leq \lambda\leq1$.
For equal means the corresponding segment is
$\sigma(\lambda)=\sigma_0^{1-\lambda}\sigma_1^\lambda$.
:::

In this complete hyperbolic plane, these segments are the unique minimizing
paths up to reparameterization. This global minimizing property is cited,
not proved here; see the Costa–Santos–Strapasson source in the next section.

### Fisher–Rao distance between two normals

:::{prf:corollary} Fisher–Rao distance within the normal family
:label: cor-euclidean-to-manifold-normal-distance

For normal laws with $\mu_0,\mu_1\in\mathbb R$ and $\sigma_0,\sigma_1>0$,
the intrinsic Fisher distance within the full normal family is

$$
d_{\mathrm{FR}}(N(\mu_0,\sigma_0^2),N(\mu_1,\sigma_1^2))
=\sqrt2\,\operatorname{arcosh}\!\left(
1+\frac{(\mu_1-\mu_0)^2+2(\sigma_1-\sigma_0)^2}{4\sigma_0\sigma_1}
\right).
$$

For equal means it is $\sqrt2|\log(\sigma_1/\sigma_0)|$.
For equal scales $\sigma$ and distinct means it is strictly smaller than
$|\mu_1-\mu_0|/\sigma$, the distance restricted to the fixed-scale subfamily.
:::

:::{prf:proof}
By [](#prop-euclidean-to-manifold-normal-hyperbolic), lengths are
$\sqrt2$ times those in the upper-half-plane metric with $u=\mu/\sqrt2$.
Use the cited hyperbolic distance formula, which satisfies
$\cosh d_H=1+((u_1-u_0)^2+(\sigma_1-\sigma_0)^2)/(2\sigma_0\sigma_1)$.
Rescaling by [](#ig-normal-hyperbolic-metric) gives

```{math}
:label: ig-normal-fisher-rao-distance
\begin{aligned}
&d_{\mathrm{FR}}\!\left(N(\mu_0,\sigma_0^2),N(\mu_1,\sigma_1^2)\right)\\
&\quad=\sqrt2\,\operatorname{arcosh}\!\left(
 1+\frac{(\mu_1-\mu_0)^2+2(\sigma_1-\sigma_0)^2}
          {4\sigma_0\sigma_1}\right).
\end{aligned}
```

The hyperbolic minimizing-path and distance results used here are treated in
[Costa–Santos–Strapasson, *Fisher information distance: a geometrical reading* (2015), §2](https://arxiv.org/abs/1210.2354).
Integrating the constant speed in [](#ig-normal-lc-geodesic-solutions) gives
the same expression. In particular,

```{math}
\begin{aligned}
\mu_0=\mu_1:\quad
 d_{\mathrm{FR}}&=\sqrt2\left|\log\frac{\sigma_1}{\sigma_0}\right|,\\
\sigma_0=\sigma_1=\sigma:\quad
 d_{\mathrm{FR}}&=\sqrt2\,\operatorname{arcosh}\!\left(
 1+\frac{(\mu_1-\mu_0)^2}{4\sigma^2}\right).
\end{aligned}
```

In the fixed-scale one-dimensional subfamily, the distance is instead
$|\mu_1-\mu_0|/\sigma$. For distinct means put $z=|\mu_1-\mu_0|/(\sqrt2\sigma)>0$.
Then $\operatorname{arcosh}(1+z^2/2)=2\operatorname{arsinh}(z/2)<z$
because the derivative of $\operatorname{arsinh}(t)$ is less than one for
$t>0$. Multiplying by $\sqrt2$ proves the strict comparison.
:::

The full-family minimizing path can increase scale
while changing location, then decrease it again.

## Statistical connections and duality

A regular statistical model has more structure than its Fisher metric: third
moments of scores distinguish a family of connections. We give their
definitions and normal-family calculations here. The
[duality note](./information-geometry-duality.md) develops the resulting
Bregman divergence and KL Pythagoras identities.

### The cubic tensor and the alpha family

Assume sufficient smoothness and integrability to differentiate the Fisher
metric, and finite absolute third score moments. Define the
**Amari–Chentsov cubic tensor** by

```{math}
:label: ig-amari-chentsov-cubic
C_{ijk}=\mathbb E_\xi[s_{\xi,i}(X)s_{\xi,j}(X)s_{\xi,k}(X)],\qquad
C(U,V,W)=C_{ijk}U^iV^jW^k.
```

Raising the last index defines
$(C^\sharp(U,V))^k=g^{k\ell}C_{ij\ell}U^iV^j$.

For $\alpha\in\mathbb R$, the statistical **$\alpha$-connection** has
coefficients with the output index lowered

```{math}
:label: ig-alpha-expectation-definition
\begin{aligned}
\Gamma^{(\alpha)}_{ij,k}
 &=\mathbb E_\xi\!\left[
 \left(\partial_i\partial_j\ell_\xi(X)
       +\frac{1-\alpha}{2}s_{\xi,i}(X)s_{\xi,j}(X)\right)s_{\xi,k}(X)\right],\\
\Gamma^{(\alpha)k}_{ij}&=g^{k\ell}\Gamma^{(\alpha)}_{ij,\ell}.
\end{aligned}
```

The comma separates the lowered output index; these coefficients themselves
are not a covariant tensor. We use the convention
$\nabla^{(e)}=\nabla^{(1)}$ and $\nabla^{(m)}=\nabla^{(-1)}$.

:::{prf:proposition} The cubic tensor and the Levi-Civita relation
:label: prop-euclidean-to-manifold-alpha-connection

Let $p_\xi>0$ be a smooth model on a fixed support with smooth positive
definite Fisher metric. Assume the expectations defining $C$ and the
connection coefficients are smooth, finite absolute third score moments,
integrability of $(\partial_i\partial_j\ell_\xi)s_{\xi,k}$, and local
integrable bounds permitting differentiation of the expectations defining
the Fisher metric. Then $C_{ijk}=\mathbb E_\xi[s_{\xi,i}s_{\xi,j}s_{\xi,k}]$
defines a symmetric covariant tensor. For every $\alpha\in\mathbb R$, the
coefficients in [](#ig-alpha-expectation-definition) define a torsion-free
connection and satisfy

$$
\Gamma^{(0)}_{ij,k}
=\mathbb E_\xi[(\partial_i\partial_j\ell_\xi)s_{\xi,k}]+\tfrac12C_{ijk},
\qquad
\nabla^{(\alpha)}_UV=\nabla^{\mathrm{LC}}_UV-\tfrac\alpha2 C^\sharp(U,V),
$$

where $(C^\sharp(U,V))^k=g^{k\ell}C_{ij\ell}U^iV^j$.
:::

:::{prf:proof}
Each score transforms as a covector, so its third product expectation is a
symmetric covariant tensor. To identify the connection, for a function
$F=F(\xi,X)$ smooth in the parameters, the stated differentiation assumptions give
$\partial_i\mathbb E_\xi[F]=\mathbb E_\xi[\partial_iF+Fs_{\xi,i}(X)]$.
Applying it to $g_{jk}$ gives

```{math}
\partial_i g_{jk}
=\mathbb E_\xi[(\partial_i\partial_j\ell_\xi(X))s_{\xi,k}(X)]
 +\mathbb E_\xi[s_{\xi,j}(X)(\partial_i\partial_k\ell_\xi(X))]+C_{ijk}.
```

Insert this into the lowered version of the Levi-Civita formula. The mixed
terms cancel, leaving

```{math}
:label: ig-alpha-lc-relation
\begin{aligned}
\Gamma^{(0)}_{ij,k}
 &=\mathbb E_\xi[(\partial_i\partial_j\ell_\xi(X))s_{\xi,k}(X)]+\frac12C_{ijk},\\
\nabla^{(\alpha)}_UV
 &=\nabla^{\mathrm{LC}}_UV-\frac\alpha2 C^\sharp(U,V).
\end{aligned}
```

This proves that the $\alpha=0$ connection is Levi-Civita and expresses the
difference of connections as a tensor. Subtracting a smooth $(1,2)$-tensor
from a connection preserves the connection axioms. Symmetry in $i,j$
preserves torsion-freeness and gives the claimed connections in all charts.
:::

### Dual connections

Two connections $\nabla$ and $\nabla^*$ are **dual with respect to $g$** when

```{math}
:label: ig-dual-connection-identity
U[g(V,W)]=g(\nabla_UV,W)+g(V,\nabla^*_UW).
```

:::{prf:corollary} Duality of opposite alpha-connections
:label: cor-euclidean-to-manifold-alpha-duality

Under [](#prop-euclidean-to-manifold-alpha-connection),
$\nabla^{(\alpha)}$ and $\nabla^{(-\alpha)}$ are dual with respect to $g$:

$$
U[g(V,W)]=g(\nabla^{(\alpha)}_UV,W)+g(V,\nabla^{(-\alpha)}_UW).
$$

Along any smooth curve, a pair of fields parallel for these respective
connections has constant inner product. Moreover,
$(\nabla^{(\alpha)}_U g)(V,W)=\alpha C(U,V,W)$.
:::

:::{prf:proof}
Substitute [](#ig-alpha-lc-relation) on the right, once with $\alpha$ and
once with $-\alpha$. The cubic terms cancel by symmetry, and Levi-Civita
compatibility supplies the left side. Thus $\nabla^{(\alpha)}$ and
$\nabla^{(-\alpha)}$ are dual, including the e/m pair. If $V$ is parallel
for one connection and $W$ for its dual along the same curve, $g(V,W)$ is
constant. Subtracting the two terms with the same $\alpha$ connection from
$U[g(V,W)]$ instead leaves $\alpha C(U,V,W)$, proving the last identity.
:::

Neither transport separately needs to preserve $g(V,V)$; nonzero $\alpha$
generally breaks metric compatibility.

These definitions and the sign convention agree with
@amari2000methods, *Methods of Information Geometry*
and @amari2016information, *Information Geometry and Its Applications*.
The calculations above establish the displayed relations under the stated
regularity assumptions.

### The normal alpha-connections

:::{prf:proposition} Normal cubic tensor, alpha-geodesics, and transport
:label: prop-euclidean-to-manifold-normal-alpha

On the normal family $N(\mu,\sigma^2)$, $\sigma>0$, the only nonzero cubic
components are $C_{\mu\mu\sigma}=C_{\mu\sigma\mu}=C_{\sigma\mu\mu}=2/\sigma^3$
and $C_{\sigma\sigma\sigma}=8/\sigma^3$. For every $\alpha\in\mathbb R$,
the only nonzero coefficients, apart from values that vanish at special
$\alpha$, are

$$
\Gamma^{(\alpha)\mu}_{\mu\sigma}=\Gamma^{(\alpha)\mu}_{\sigma\mu}
=-\frac{1+\alpha}{\sigma},\quad
\Gamma^{(\alpha)\sigma}_{\mu\mu}=\frac{1-\alpha}{2\sigma},\quad
\Gamma^{(\alpha)\sigma}_{\sigma\sigma}=-\frac{1+2\alpha}{\sigma}.
$$

Its affinely parameterized geodesics satisfy
$\ddot\mu-2(1+\alpha)\dot\mu\dot\sigma/\sigma=0$ and
$\ddot\sigma+(1-\alpha)\dot\mu^2/(2\sigma)
-(1+2\alpha)\dot\sigma^2/\sigma=0$.
Along a fixed-mean curve from scale $\sigma_0$ to $\sigma(\lambda)$,
parallel transport multiplies the $\mu$ component by
$(\sigma(\lambda)/\sigma_0)^{1+\alpha}$ and the $\sigma$ component by
$(\sigma(\lambda)/\sigma_0)^{1+2\alpha}$.
:::

:::{prf:proof}
For the scores $Z/\sigma$ and $(Z^2-1)/\sigma$, use
$\mathbb E[Z^6]=15$. The only nonzero cubic components are

```{math}
C_{\mu\mu\sigma}=C_{\mu\sigma\mu}=C_{\sigma\mu\mu}
=\frac2{\sigma^3},\qquad
C_{\sigma\sigma\sigma}=\frac8{\sigma^3}.
```

For example,
$\mathbb E[Z^2(Z^2-1)]=3-1=2$ and
$\mathbb E[(Z^2-1)^3]=15-9+3-1=8$.
Raising the output index in [](#ig-alpha-lc-relation) gives the following
coefficients; all others vanish:

```{math}
:label: ig-normal-alpha-christoffels
\begin{aligned}
\Gamma^{(\alpha)\mu}_{\mu\sigma}
 =\Gamma^{(\alpha)\mu}_{\sigma\mu}&=-\frac{1+\alpha}{\sigma},\\
\Gamma^{(\alpha)\sigma}_{\mu\mu}&=\frac{1-\alpha}{2\sigma},\\
\Gamma^{(\alpha)\sigma}_{\sigma\sigma}&=-\frac{1+2\alpha}{\sigma}.
\end{aligned}
```

Setting $\alpha=0$ recovers [](#ig-normal-lc-christoffels). The general
geodesic equations become

```{math}
:label: ig-normal-alpha-geodesic-odes
\ddot\mu-2(1+\alpha)\frac{\dot\mu\dot\sigma}{\sigma}=0,
\qquad
\ddot\sigma+\frac{1-\alpha}{2\sigma}\dot\mu^2
             -\frac{1+2\alpha}{\sigma}\dot\sigma^2=0.
```

Along a fixed-mean curve, the parallel ODEs are
$\dot V^\mu=(1+\alpha)(\dot\sigma/\sigma)V^\mu$ and
$\dot V^\sigma=(1+2\alpha)(\dot\sigma/\sigma)V^\sigma$.
Solving these scalar linear equations gives

```{math}
V^\mu(\lambda)=V^\mu(0)\left(\frac{\sigma(\lambda)}{\sigma_0}\right)^{1+\alpha},
\qquad
V^\sigma(\lambda)=V^\sigma(0)\left(\frac{\sigma(\lambda)}{\sigma_0}\right)^{1+2\alpha}.
```
:::

For the dual connection the exponents have the opposite $\alpha$ terms.
The paired inner product stays constant, although a vector's own Fisher norm
generally changes when $\alpha\ne0$.

### Natural and expectation coordinates make different connections flat

:::{prf:proposition} Flat exponential and mixture charts for normals
:label: prop-euclidean-to-manifold-normal-dual-flatness

For the full normal family, the exponential connection $\nabla^{(1)}$ has
zero coefficients in the natural chart $\theta$, and the mixture connection
$\nabla^{(-1)}$ has zero coefficients in the expectation chart
$\eta=\nabla\psi(\theta)$. In natural coordinates,
$C_{ijk}=\partial_i\partial_j\partial_k\psi$.
Both connections have zero curvature.
:::

:::{prf:proof}
In this paragraph, use $\partial_i=\partial/\partial\theta^i$ and take the
connection and cubic-tensor components in the natural chart. Then
$\partial_i\partial_j\ell_\theta(x)=-\partial_i\partial_j\psi$ is independent
of the observation. Since every score has expectation zero,
[](#ig-alpha-expectation-definition) at $\alpha=1$ gives
$\Gamma^{(e)k}_{ij}=0$. Also
$C_{ijk}=\partial_i\partial_j\partial_k\psi$ in these coordinates, by
differentiating the covariance of $t(X)$.

The expectation chart makes the dual mixture coefficients vanish. Here is a
coordinate argument. The natural coordinate vector fields
$E_i=\partial/\partial\theta^i$ are e-parallel. Let
$F_a=\partial/\partial\eta^a$. Since $d\theta=I(\theta)^{-1}d\eta$,

```{math}
g(E_i,F_a)
=\sum_j [I(\theta)]_{ij}[I(\theta)^{-1}]_{ja}
=\begin{cases}1,&i=a,\\0,&i\ne a.\end{cases}
```

The explicit sum here multiplies matrices labeled by two different coordinate
bases; the numeric identity compares their paired labels. Differentiate this
constant pairing with any field $U$ and use e/m duality. Because
$\nabla^{(e)}_UE_i=0$, we obtain
$g(E_i,\nabla^{(m)}_UF_a)=0$ for every $i$, hence
$\nabla^{(m)}_UF_a=0$. In particular, all m-Christoffel symbols vanish in
the $\eta$ chart.
In a chart where all connection coefficients vanish, their derivatives
and the quadratic terms in the curvature also vanish.
:::

This **dual flatness**
coexists with the curvature $-1/2$ of $\nabla^{\mathrm{LC}}$. Curvature is
attached to a specified connection; the vanishing of the e- or m-coefficients
in their affine charts does not make the Fisher metric Euclidean.

### Three geodesics between two normal laws

The two affine charts give different interpolations between the same laws.

:::{prf:corollary} Exponential and mixture geodesics between normal laws
:label: cor-euclidean-to-manifold-normal-em-geodesics

Let $P_0=N(\mu_0,\sigma_0^2)$ and $P_1=N(\mu_1,\sigma_1^2)$ have positive
scales and natural coordinates $\theta_0,\theta_1$. For $0\le\lambda\le1$,
the affinely parameterized e-geodesic is

```{math}
:label: ig-normal-e-geodesic
\begin{aligned}
\theta(\lambda)&=(1-\lambda)\theta_0+\lambda\theta_1,\\
\sigma_e^2(\lambda)&=\left(\frac{1-\lambda}{\sigma_0^2}
                         +\frac \lambda{\sigma_1^2}\right)^{-1},\\
\mu_e(\lambda)&=\sigma_e^2(\lambda)
 \left(\frac{(1-\lambda)\mu_0}{\sigma_0^2}
                  +\frac{\lambda\mu_1}{\sigma_1^2}\right).
\end{aligned}
```

Its density is proportional to $p_{\theta_0}^{1-\lambda}p_{\theta_1}^\lambda$.

A straight expectation-coordinate segment is an affinely parameterized
m-geodesic of the connection on this normal family:

```{math}
:label: ig-normal-m-geodesic
\begin{aligned}
\eta&=(1-\lambda)\eta(\theta_0)+\lambda\eta(\theta_1),\\
\mu_m(\lambda)&=(1-\lambda)\mu_0+\lambda\mu_1,\\
\sigma_m^2(\lambda)&=(1-\lambda)\sigma_0^2+\lambda\sigma_1^2
                 +\lambda(1-\lambda)(\mu_1-\mu_0)^2.
\end{aligned}
```

Both paths stay in the normal family, and the m-geodesic has the same
first two moments as $(1-\lambda)P_0+\lambda P_1$.
:::

:::{prf:proof}
By [](#prop-euclidean-to-manifold-normal-dual-flatness), straight segments
in $\theta$ and $\eta$ solve the respective geodesic equations.
Inverting the natural coordinates gives the e-path formulas and positive
precision. Substitution in the natural log density gives the stated density
product. For the m-path, subtract the square of the interpolated mean from
the interpolated second raw moment to obtain its variance formula.
The variance is positive because the endpoint variances are positive.
The moment identities follow from linearity of expectation under a mixture.
Differentiation also verifies [](#ig-normal-alpha-geodesic-odes) at
$\alpha=1$ and $\alpha=-1$, respectively.
:::

The e-path interpolates precision and precision-weighted mean. The m-path
normal is generally not the literal mixture: a mixture of two distinct
normals need not be normal. Thus m-flatness does not assert closure under
density mixtures.

For comparison, the Levi-Civita segment is the vertical path or half-ellipse
in [](#ig-normal-lc-geodesic-solutions), with endpoints chosen as above. It
minimizes Fisher length; e- and m-geodesics are straight for their respective
connections and have no general Fisher-length-minimization property.

For $P_0=N(0,1)$ and $P_1=N(2,1)$ the difference is explicit:

| Connection | Path in $(\mu,\sigma)$ | Scale where $\mu=1$ |
| --- | --- | --- |
| e, $\alpha=1$ | $\mu=2\lambda$, $\sigma=1$ | $1$ |
| Levi-Civita, $\alpha=0$ | $(\mu-1)^2/2+\sigma^2=3/2$ on the joining arc | $\sqrt{3/2}$ |
| m, $\alpha=-1$ | $\mu=2\lambda$, $\sigma^2=1+4\lambda(1-\lambda)$ | $\sqrt2$ |

The Levi-Civita parameter in this comparison is the affine hyperbolic
parameter, not generally $\mu/2$. Its length is
$\sqrt2\,\operatorname{arcosh}(2)$; the horizontal e-path has length $2$.
If the endpoints have equal means, all three paths trace the same vertical
segment, but their affine parameterizations generally differ: e interpolates
precision, m interpolates variance, and Levi-Civita interpolates log scale.

Continue with [exponential families](./information-geometry-exponential-families.md)
for sufficient statistics and the log-partition function, or
[dual coordinates and KL projections](./information-geometry-duality.md)
for the Bregman and Pythagorean consequences. The
[Information Geometry hub](./ig/index.md) lists the complete entry path.
