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
- $G_\xi(\xi)=[g_{ij}(\xi)]$ is the metric matrix in the $\xi$ chart. Thus
  $g_\xi(v,w)=g_{ij}v^iw^j=v_\xi^{\mathsf T}G_\xi w_\xi$.
  Once the metric is Fisher information, $G_\xi=I(\xi)$.
  $ds^2=g_{ij}\,d\xi^i d\xi^j$ denotes the corresponding quadratic form.
- $\Gamma_{ij}^k$ means $\nabla_{\partial_i}\partial_j=\Gamma_{ij}^k\partial_k$;
  $\Gamma_{ij,k}=g_{k\ell}\Gamma_{ij}^{\ell}$ lowers its output index. A
  superscript $(\alpha)$ identifies a connection, not a tensor index.
- $\ell_\xi(x)=\log p_\xi(x)$, $s_i=\partial_i\ell_\xi$, and
  $s_\xi=(s_i)$ is the column of score components. For fixed $x$, these are
  covector components: the coordinate gradient symbol in
  $s_\xi=\nabla_\xi\ell_\xi$ does not mean a Riemannian gradient.
  $\mathbb E_\xi$ means expectation under $P_\xi$; in natural coordinates we
  write $p_\theta$, $s_\theta$, and $\mathbb E_\theta$.

We use matrices for explicit Jacobians and the Legendre identities
$\eta=\nabla\psi(\theta)$ and $\nabla^2\psi^*(\eta)=(\nabla^2\psi(\theta))^{-1}$.
Here the gradients and Hessians are ordinary derivatives in the named affine
coordinates. Their entries will be related to tensor components explicitly.
Curve parameters are denoted by $r$, leaving $t(x)$ for sufficient statistics.
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

With $t(x)=(x,x^2)^{\mathsf T}$, completing the square gives

```{math}
:label: ig-normal-natural-chart
\begin{aligned}
p_\theta(x)&=\exp\{\theta^{\mathsf T}t(x)-\psi(\theta)\},\\
\psi(\theta)&=-\frac{(\theta^1)^2}{4\theta^2}
 +\frac12\log\frac{\pi}{-\theta^2},\\
\eta&=\mathbb E_\theta[t(X)]=\nabla\psi(\theta)
 =(\mu,\mu^2+\sigma^2)^{\mathsf T}.
\end{aligned}
```

Thus the second expectation coordinate is the **second raw moment**, not the
variance. The [exponential-family note](./information-geometry-exponential-families.md#two-examples)
derives these coordinates in the broader setting of sufficient statistics;
its statistic $T$ is written $t$ here, following the shared canon.

The inverse coordinate maps include

```{math}
\mu=-\frac{\theta^1}{2\theta^2},\qquad
\sigma=\sqrt{-\frac1{2\theta^2}};
\qquad
\mu=\eta^1,\qquad
\sigma=\sqrt{\eta^2-(\eta^1)^2}.
```

Write $f_\phi$, $f_\theta$, and $f_\eta$ for the maps from the $\xi$ chart to
these three charts. Direct differentiation gives

```{math}
:label: ig-normal-chart-jacobians
\begin{aligned}
Df_\phi(\xi)&=\begin{pmatrix}1&0\\0&2\sigma\end{pmatrix},\\[3pt]
Df_\theta(\xi)&=\begin{pmatrix}
 \sigma^{-2}&-2\mu\sigma^{-3}\\0&\sigma^{-3}
\end{pmatrix},\\[3pt]
Df_\eta(\xi)&=\begin{pmatrix}1&0\\2\mu&2\sigma\end{pmatrix}.
\end{aligned}
```

Their determinants are nonzero for $\sigma>0$. The inverse Jacobian is
$Df^{-1}(f(\xi))=Df(\xi)^{-1}$; specifying its evaluation point prevents
confusing the two directions of a coordinate change.

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
through $q$ at $r=0$ are equivalent when

```{math}
\left.\frac{d}{dr}\xi^i(c_1(r))\right|_{r=0}
=
\left.\frac{d}{dr}\xi^i(c_2(r))\right|_{r=0}
\quad\text{for every }i.
```

The chain rule makes this equivalence independent of the chart. A tangent
vector is an equivalence class $v=[c]$ of such curves. Its components are
$v^i=(\xi^i\circ c)'(0)$. Every column of $d$ real numbers is realized by a
coordinate line $\xi(c(r))=\xi(q)+rv$ for sufficiently small $r$. These
components give the set of classes a $d$-dimensional vector-space structure,
called $T_qM$; coordinate changes act linearly on that space.

For normals,
$v=v^\mu\partial_\mu+v^\sigma\partial_\sigma$ is represented by
$(\mu(r),\sigma(r))=(\mu+rv^\mu,\sigma+rv^\sigma)$ near $r=0$.
The coordinates of a law and the components of a tangent vector have different
roles: one specifies a distribution; the other specifies rates of change at
that distribution.

### The equivalent definition by derivations

A derivation at $q$ is an $\mathbb R$-linear map
$A:C^\infty(M)\to\mathbb R$ satisfying the pointwise Leibniz rule

```{math}
A(FH)=F(q)A(H)+H(q)A(F).
```

It depends only on functions near $q$. Indeed, if $F$ vanishes near $q$,
choose a smooth cutoff equal to one at $q$ and supported where $F=0$;
applying the product rule to their zero product gives $A(F)=0$. Thus we can
also work with germs of smooth functions at $q$, and apply $A$ to local
coordinate functions.

A curve class defines such a derivation by

```{math}
:label: ig-tangent-derivation
v[F]=\left.\frac{d}{dr}F(c(r))\right|_{r=0}
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

For example, the smooth function $F(P_{\mu,\sigma})=\mathbb E_\xi[X^2]
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
$B^i{}_a=\partial\xi^i/\partial\phi^a$, so $B=A^{-1}$. The chain rule gives

```{math}
:label: ig-tangent-coordinate-change
\partial_i=A^a{}_i\partial'_a,\qquad
v'^a=A^a{}_i v^i,\qquad
v_\phi=Df(\xi)v_\xi.
```

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

In source coordinates $\xi$ and target coordinates $\phi$, its matrix is the
Jacobian of $\phi\circ f\circ\xi^{-1}$, denoted $Df(\xi)$. Its entries are
$\partial f^a/\partial\xi^i$, and
$(df_qv)^a=(\partial f^a/\partial\xi^i)v^i$. Unlike a coordinate change, a
smooth map need not be invertible or preserve dimension. The variance map
$f:M\to(0,\infty)$, $f(P_{\mu,\sigma})=\sigma^2$, has
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

Under the same transition map,
$d\phi^a=A^a{}_i d\xi^i$ and $\omega'_a=B^i{}_a\omega_i$. This inverse
transformation is what makes $\omega(v)$ independent of coordinates.

A smooth vector field $V$ assigns $V_q\in T_qM$ at every point, with smooth
components $V^i$ in every chart. It differentiates functions by
$V[F]=V^i\partial_iF$. On the normal family, $V=\sigma\partial_\sigma$
describes relative scale change. Its integral curves satisfy
$\dot\mu=0$, $\dot\sigma=\sigma$, so $\sigma(r)=\sigma(0)e^r$.
In the variance chart this same field is $2\tau\partial_\tau$.

## Metrics and Fisher information

### A metric is more than a parameter norm

A Riemannian metric $g$ assigns a smooth positive definite symmetric bilinear
form $g_q$ to every tangent space. Its components are
$g_{ij}=g(\partial_i,\partial_j)$. Substituting
$v^i=B^i{}_a v'^a$ and $w^j=B^j{}_b w'^b$ gives

```{math}
:label: ig-metric-coordinate-change
\begin{aligned}
g'_{ab}(\phi)&=B^i{}_a B^j{}_b g_{ij}(\xi),\\
G_\phi(\phi)&=B^{\mathsf T}G_\xi(\xi)B
 =Df(\xi)^{-\mathsf T}G_\xi(\xi)Df(\xi)^{-1},
\qquad \phi=f(\xi).
\end{aligned}
```

Every Riemannian metric has this covariance. It does not by itself select
Fisher information. Even a Euclidean metric acquires nonconstant coefficients
in nonlinear coordinates. On an angular sector of the Euclidean plane, write
$x=\rho\cos\varphi$, $y=\rho\sin\varphi$, with radius $\rho>0$. Then

```{math}
:label: ig-polar-metric
ds^2=dx^2+dy^2=d\rho^2+\rho^2d\varphi^2.
```

Thus nonconstant metric coefficients alone do not establish curvature.

For normals, choosing $d\mu^2+d\sigma^2$ in the scale chart and choosing
$d\mu^2+d\tau^2$ afresh in the variance chart gives incompatible metrics:

```{math}
d\mu^2+d\tau^2=d\mu^2+4\sigma^2d\sigma^2.
```

At $\sigma=2$, the same unit scale velocity has squared lengths $1$ and $16$.
A Euclidean metric chosen in one chart is legitimate, but must then be
transformed by [](#ig-metric-coordinate-change). A statistical principle is
needed to choose a metric without privileging that chart.

For a piecewise smooth curve $c:[a,b]\to M$, the metric determines length and
intrinsic distance by

```{math}
L_g(c)=\int_a^b\sqrt{g_{ij}(\xi(c(r)))\dot\xi^i\dot\xi^j}\,dr,
\qquad
d_g(q_0,q_1)=\inf_{c:q_0\to q_1}L_g(c).
```

The infimum is over curves in the specified manifold. Restricting to a
subfamily can increase this distance.

### Scores measure changes in the law

For a general model, assume positive densities on a common support relative
to a fixed measure $\nu$. Assume smooth parameter dependence, square-integrable
scores, and local integrable bounds allowing differentiation of the normalizer
and the expectations used below. We will specify the additional Taylor and
third-moment assumptions where they are needed.

Differentiating the density and its normalization gives

```{math}
\partial_i p_\xi=p_\xi s_i,\qquad
\mathbb E_\xi[s_i]=0,\qquad
\left.\frac{d}{dr}\ell_{\xi(c(r))}(x)\right|_{r=0}=s_i(x)v^i.
```

The **Fisher metric per observation** is

```{math}
:label: ig-fisher-score-metric
\begin{aligned}
g_\xi(v,w)&=\mathbb E_\xi[(s_i v^i)(s_j w^j)],\\
g_{ij}(\xi)&=\mathbb E_\xi[s_i s_j],\qquad
I(\xi)=\mathbb E_\xi[s_\xi s_\xi^{\mathsf T}].
\end{aligned}
```

It is positive semidefinite; our regularity assumption requires positive
definiteness. A zero length direction would have zero first-order density
change almost everywhere. Under reparameterization the score obeys
$s'_a=B^i{}_a s_i$, so its covariance obeys exactly the metric transformation
law. Invertible, parameter-independent changes of the observed variable also
preserve Fisher information: their density Jacobian contributes a term to the
log density whose parameter derivative is zero.

### Computing the normal metric

Set $Z=(X-\mu)/\sigma\sim N(0,1)$. Differentiating [](#ig-normal-density),

```{math}
s_\mu=\frac{X-\mu}{\sigma^2}=\frac Z\sigma,
\qquad
s_\sigma=-\frac1\sigma+\frac{(X-\mu)^2}{\sigma^3}
 =\frac{Z^2-1}{\sigma}.
```

The subscripts here denote covector components, not differentiation of an
already defined score. Odd standard-normal moments vanish, and integration
by parts gives $\mathbb E[Z^{2k}]=(2k-1)\mathbb E[Z^{2k-2}]$.
In particular, the second and fourth moments are $1$ and $3$. Consequently,

```{math}
:label: ig-normal-fisher-metric
\begin{aligned}
G_\xi=I(\xi)&=\begin{pmatrix}\sigma^{-2}&0\\0&2\sigma^{-2}\end{pmatrix},\\
g_\xi(v,w)&=\frac{v^\mu w^\mu+2v^\sigma w^\sigma}{\sigma^2},\\
ds^2&=\frac{d\mu^2+2d\sigma^2}{\sigma^2}.
\end{aligned}
```

This is positive definite for every $\sigma>0$. On compact parameter
neighborhoods with scale bounded away from zero, Gaussian tail bounds justify
the differentiations and moments used here and later. A fixed location
velocity has smaller Fisher length at a larger scale, because its change in
log density is smaller in mean square.

Using $B=\operatorname{diag}(1,1/(2\sigma))$ in the variance chart gives

```{math}
:label: ig-normal-fisher-charts
G_\phi=\begin{pmatrix}\tau^{-1}&0\\0&(2\tau^2)^{-1}\end{pmatrix},
\qquad
ds^2=\frac{d\mu^2+2d\sigma^2}{\sigma^2}
 =\frac{d\mu^2}{\tau}+\frac{d\tau^2}{2\tau^2}.
```

Substitution of $v^\tau=2\sigma v^\sigma$ recovers the same squared length.

### Natural and expectation coordinates

In natural coordinates $s_\theta=t-\eta$, so differentiating the normalizer
in [](#ig-normal-natural-chart) gives the matrix identities

```{math}
:label: ig-normal-dual-metrics
\begin{aligned}
G_\theta(\theta)=I(\theta)
 &=\nabla^2\psi(\theta)=\operatorname{Cov}_\theta(t(X))\\
 &=\begin{pmatrix}
 \sigma^2&2\mu\sigma^2\\
 2\mu\sigma^2&4\mu^2\sigma^2+2\sigma^4
 \end{pmatrix},\\
\psi^*(\eta)&=\theta^{\mathsf T}\eta-\psi(\theta)
 =-\frac12\log\!\left(2\pi e[\eta^2-(\eta^1)^2]\right),\\
G_\eta(\eta)&=\nabla^2\psi^*(\eta)=G_\theta(\theta(\eta))^{-1}\\
 &=\begin{pmatrix}
 \tau^{-1}+2\mu^2\tau^{-2}&-\mu\tau^{-2}\\
 -\mu\tau^{-2}&(2\tau^2)^{-1}
 \end{pmatrix},\quad
 \mu=\eta^1,\quad \tau=\eta^2-(\eta^1)^2.
\end{aligned}
```

The expression for $\psi^*$ is the Legendre transform, attained at the natural
parameter corresponding to $\eta$. Indeed, $d\eta=G_\theta d\theta$, whence
$d\theta=G_\theta^{-1}d\eta$; the metric transformation law gives $G_\eta$.
Here $(G_\theta)_{ij}=\partial^2\psi/(\partial\theta^i\partial\theta^j)$
and $(G_\eta)_{ab}=\partial^2\psi^*/(\partial\eta^a\partial\eta^b)$.
The inverse of one coordinate matrix supplies the components in the other
chart; it does not change the covariant type of the metric. The Jacobians in
[](#ig-normal-chart-jacobians) pull both matrices back to $G_\xi$.
An ordinary Hessian need not transform as a tensor under arbitrary nonlinear
coordinates; these Hessian formulas use the particular affine coordinates
$\theta$ and $\eta$.

## Why Fisher information is distinguished

Coordinate covariance is required of every metric. Fisher information has
additional properties concerning statistical distinguishability and the loss
of observations. We prove the local KL and monotonicity statements below;
the uniqueness theorem is cited with its finite-space hypotheses.

### The Hessian of KL at the diagonal

Use the direction
$D(p_\xi\|p_\zeta)=\mathbb E_\xi[\ell_\xi-\ell_\zeta]$, holding the first
law fixed. Differentiating normalization twice yields

```{math}
0=\mathbb E_\xi[\partial_i\partial_j\ell_\xi+s_i s_j],
\qquad
\mathbb E_\xi[\partial_i\partial_j\ell_\xi]=-g_{ij}.
```

Assume in addition that $\zeta\mapsto\mathbb E_\xi[\ell_\zeta]$ has a
third-order Taylor remainder bounded by a constant times
$\lVert\zeta-\xi\rVert^3$ locally. For example, locally bounded integrable
third derivatives of the log density under $P_\xi$ suffice. Taylor expansion
then gives

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
Finite KL remains asymmetric and is not half the squared Fisher–Rao distance.

For normals the exact expression is

```{math}
D\!\left(N(\mu,\sigma^2)\middle\|N(\widetilde\mu,\widetilde\sigma^2)\right)
=\log\frac{\widetilde\sigma}{\sigma}-\frac12
 +\frac{\sigma^2+(\mu-\widetilde\mu)^2}{2\widetilde\sigma^2}.
```

Putting $(\widetilde\mu,\widetilde\sigma)
=(\mu+\varepsilon a,\sigma+\varepsilon b)$ at a fixed $\sigma>0$ gives
$\varepsilon^2(a^2+2b^2)/(2\sigma^2)+O(\varepsilon^3)$, in agreement with
[](#ig-normal-fisher-metric).

### Coarse-graining, monotonicity, and sufficiency

Let an observation $X$ pass through a parameter-independent Markov kernel
$K(dy\mid x)$, giving $Y$. A deterministic statistic $Y=S(X)$ is a special
case. Assume the transformed model admits densities and differentiation under
the kernel integral, and that the original scores are square integrable.
Denote the two score covectors by $s^X_i$ and $s^Y_i$, and their component
columns by $s^X_\xi$ and $s^Y_\xi$. Differentiating the marginal density of
$Y$ gives

```{math}
:label: ig-score-under-statistic
s^Y_i(Y)=\mathbb E_\xi[s^X_i(X)\mid Y].
```

Thus the new directional score is the conditional expectation of the old one.
The conditional-variance identity proves information monotonicity:

```{math}
:label: ig-fisher-monotonicity
\begin{aligned}
g^X_\xi(v,v)-g^Y_\xi(v,v)
 &=\mathbb E_\xi\!\left[
   \operatorname{Var}_\xi(v^i s^X_i(X)\mid Y)\right]\geq0,\\
I_X(\xi)-I_Y(\xi)
 &=\mathbb E_\xi[\operatorname{Cov}_\xi(s^X_\xi\mid Y)]\succeq0.
\end{aligned}
```

The superscripts $X,Y$ identify experiments; they are not tensor indices.
The output information can be singular. Equality for one vector at one
parameter means exactly that its directional score is determined by $Y$
almost surely. Equality of the matrices at that parameter means this for
every score component. This is a statement about first-order information at
that point.

A statistic is **sufficient for the family** if the conditional distribution
of $X$ given the statistic can be chosen independently of the parameter.
Equivalently in the regular dominated setting, the density factors into a
parameter-dependent function of the statistic and a parameter-independent
factor. Its scores are functions of the statistic, so sufficiency implies
equality of Fisher information throughout the family.

For a precise converse, take a finite sample space with strictly positive
smooth probabilities on a connected open parameter domain, and a fixed
statistic $S$. Equality of Fisher information at **every** parameter is then
equivalent to sufficiency. To prove the converse, equality in
[](#ig-fisher-monotonicity) gives, for every outcome,

```{math}
\partial_i\log P_\xi(X=x\mid S(X)=S(x))
=s^X_i(x)-s^S_i(S(x))=0.
```

Connectedness makes each conditional probability constant in the parameter,
which proves sufficiency. Equality at one parameter, or in one direction,
is insufficient for this conclusion. Extensions beyond this finite positive
setting need corresponding regularity assumptions; see
[Ay, Jost, Lê, and Schwachhöfer on sufficient statistics](https://arxiv.org/abs/1207.6736).

The normal family illustrates both preservation and loss. For independent
observations $X_1,\ldots,X_n$, the statistic
$S=(\sum_a X_a,\sum_a X_a^2)$ is sufficient by the factorization in
[](#ig-normal-natural-chart). Its information equals that of the full sample,
namely $nI(\xi)$: independent zero-mean scores add and their cross-covariances
vanish. By contrast, retaining only $Y=\mathbf 1_{\{X>0\}}$ gives a Bernoulli
probability $q=\Phi(\mu/\sigma)$, where $\Phi$ is the standard-normal cdf.
At $\mu=0$, $\partial_\mu q=1/(\sqrt{2\pi}\sigma)$ and
$\partial_\sigma q=0$, so

```{math}
I_Y(0,\sigma)=\begin{pmatrix}2/(\pi\sigma^2)&0\\0&0\end{pmatrix}.
```

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

Pulling [](#ig-chentsov-finite) back along a regular finite statistical model,
with $u(a)=\partial_i p_\xi(a)v^i=p_\xi(a)s_i(a)v^i$, gives $cg_\xi(v,w)$.
We choose $c=1$ for information per observation. General Markov maps are
contractions by [](#ig-fisher-monotonicity), whereas sufficient transformations
preserve information. Requiring invariance under *every* Markov map would be
wrong: a kernel can discard the observation completely. Equivalently, demanding
monotonicity for a family of metrics under all such maps forces the isometries
above, since $K$ and its Markov left inverse give opposite inequalities.

For general sample spaces, [Ay–Jost–Lê–Schwachhöfer (2017)](https://link.springer.com/book/10.1007/978-3-319-56478-4)
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

The raw density metric, when finite,
$g^{\mathrm{raw}}_{ij}=\int(\partial_i p)(\partial_j p)\,d\nu$, also
transforms correctly under parameter changes. It depends on the reference
measure and the measurement coordinate: for Lebesgue densities and $Y=aX$,
$a>0$, $\int(p_Y-q_Y)^2\,dy=a^{-1}\int(p_X-q_X)^2\,dx$.
It therefore differs from Fisher geometry even though both constructions are
covariant under reparameterization.

Assuming differentiability into $L^2(\nu)$, the map $p\mapsto2\sqrt p$
has derivative $\partial_i(2\sqrt p)=\sqrt p\,s_i$. Its pullback of the
$L^2$ inner product is exactly $g_{ij}$. With
$H^2(p,q)=\tfrac12\int(\sqrt p-\sqrt q)^2\,d\nu$, this gives
$H^2(p_\xi,p_{\xi+\delta\xi})=\tfrac18g_{ij}\delta\xi^i\delta\xi^j
+o(\lVert\delta\xi\rVert^2)$.
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

Expanding $U=U^i\partial_i$ and $V=V^j\partial_j$ with the axioms gives

```{math}
:label: ig-covariant-derivative-coordinates
\nabla_UV=U^i\bigl(\partial_iV^k+\Gamma_{ij}^kV^j\bigr)\partial_k.
```

The second term accounts for the change of the coordinate basis under the
connection. Prescribing smooth $\Gamma_{ij}^k$ defines a connection on a
chart; prescriptions on overlapping charts must satisfy the next law.

### Why Christoffel symbols are not a tensor

Use $\phi=f(\xi)$, $A^a{}_i=\partial\phi^a/\partial\xi^i$, and
$B^i{}_a=\partial\xi^i/\partial\phi^a$ as before. Apply the connection
axioms to $\nabla_{B^i{}_a\partial_i}(B^j{}_b\partial_j)$ to obtain

```{math}
:label: ig-christoffel-transformation
\Gamma'{}^c_{ab}
=A^c{}_k B^i{}_a B^j{}_b\Gamma_{ij}^k
 +A^c{}_k\frac{\partial^2\xi^k}{\partial\phi^a\partial\phi^b}.
```

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

The difference of two connections *is* a $(1,2)$-tensor: their inhomogeneous
terms cancel. Equivalently, the Leibniz terms cancel in
$\nabla_UV-\widetilde\nabla_UV$, leaving an expression linear over functions
in both $U$ and $V$. We will use this fact to compare statistical connections.

### Along a curve: parallel transport and geodesics

For a curve $c(r)$, let $V(r)=V^k(r)\partial_k|_{c(r)}$ be a vector field
along the curve. It need not extend to a single vector field on all of $M$.
Its covariant derivative is

```{math}
:label: ig-curve-covariant-derivative
\frac{DV}{dr}:=\nabla_{\dot c}V
=\left(\frac{dV^k}{dr}
 +\Gamma_{ij}^k(\xi(c(r)))\dot\xi^i V^j\right)\partial_k.
```

The transformation law makes this independent of coordinates. A field is
**parallel** when $DV/dr=0$. Given an initial vector, this linear ODE has a
unique solution along a smooth curve on a compact parameter interval. It
defines a linear isomorphism between the endpoint tangent spaces, called
parallel transport. Reversing the curve gives its inverse. Transport generally
depends on the path, not just its endpoints.

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

It is a tensor. A connection is torsion-free exactly when its lower
Christoffel indices are symmetric in a coordinate basis.

A connection is **metric-compatible**, written $\nabla g=0$, when

```{math}
:label: ig-metric-compatibility
U[g(V,W)]=g(\nabla_UV,W)+g(V,\nabla_UW).
```

Equivalently,
$\partial_i g_{jk}=\Gamma_{ij}^{\ell}g_{\ell k}
+\Gamma_{ik}^{\ell}g_{j\ell}$. Along a curve this product rule shows that
parallel transport preserves inner products. In particular, a geodesic for
a metric-compatible connection has constant speed.

The **Levi-Civita connection** $\nabla^{\mathrm{LC}}$ is the unique
connection that is both torsion-free and compatible with $g$. The identity
that determines it is the Koszul formula:

```{math}
:label: ig-koszul-formula
\begin{aligned}
2g(\nabla^{\mathrm{LC}}_UV,W)
={}&U[g(V,W)]+V[g(W,U)]-W[g(U,V)]\\
 &+g([U,V],W)-g([V,W],U)+g([W,U],V).
\end{aligned}
```

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

We write $(0)$ for Levi-Civita because it will be the $\alpha=0$ statistical
connection. We use the standard Riemannian result that its geodesics minimize
length on sufficiently short segments. It does not say every geodesic segment
is globally minimizing on every manifold. For background on this result and
the connection framework, see §2 of
[Nielsen's 2020 introduction](https://arxiv.org/abs/1808.08271).

## Levi-Civita geometry of the normal family

### Christoffel symbols and parallel transport

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
This coefficient already shows why a horizontal line with constant scale
and changing mean is not a Levi-Civita geodesic in the full family.

Along any fixed-mean curve $c(r)=(\mu_0,\sigma(r))$, the parallel equations
reduce to

```{math}
\dot V^\mu-\frac{\dot\sigma}{\sigma}V^\mu=0,
\qquad
\dot V^\sigma-\frac{\dot\sigma}{\sigma}V^\sigma=0.
```

Thus a vector transported from scale $\sigma_0$ to scale $\sigma_1$ has
components multiplied by $\sigma_1/\sigma_0$. Its Fisher squared length
$( (V^\mu)^2+2(V^\sigma)^2 )/\sigma^2$ stays fixed. Constant coordinate
components would not give parallel transport along this curve.

### Hyperbolic curvature and geodesics

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
$\mu(r)=c$, $\sigma(r)=A e^{br}$, with $A>0$. Otherwise, in the rescaled
coordinate $u$, the solutions trace semicircles orthogonal to the boundary
$\sigma=0$. An affine parameterization in the original chart is

```{math}
:label: ig-normal-lc-geodesic-solutions
\begin{aligned}
\mu(r)&=c+\sqrt2\,R\tanh(br+a),\\
\sigma(r)&=R\operatorname{sech}(br+a),\qquad R>0,\quad b\ne0,\\
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

For endpoints $(\mu_0,\sigma_0)$ and $(\mu_1,\sigma_1)$ with different
means, the center and radius are

```{math}
c=\frac{\mu_1^2-\mu_0^2+2(\sigma_1^2-\sigma_0^2)}{2(\mu_1-\mu_0)},
\qquad
R^2=\frac{(\mu_0-c)^2}{2}+\sigma_0^2.
```

Set
$a=\operatorname{artanh}((\mu_0-c)/(\sqrt2R))$ and
$b=\operatorname{artanh}((\mu_1-c)/(\sqrt2R))-a$ in
[](#ig-normal-lc-geodesic-solutions) to join them over $0\leq r\leq1$.
For equal means the corresponding segment is
$\sigma(r)=\sigma_0^{1-r}\sigma_1^r$. In this complete hyperbolic plane,
these segments are the unique minimizing paths up to reparameterization.

### Fisher–Rao distance between two normals

The upper-half-plane distance satisfies
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
$|\mu_1-\mu_0|/\sigma$. The full-family minimizing path can increase scale
while changing location, then decrease it again. For distinct means it is
strictly shorter than the horizontal fixed-scale path.

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
C_{ijk}=\mathbb E_\xi[s_i s_j s_k],\qquad
C(U,V,W)=C_{ijk}U^iV^jW^k.
```

It is a symmetric covariant tensor because each score transforms as a
covector. Raising the last index defines
$(C^\sharp(U,V))^k=g^{k\ell}C_{ij\ell}U^iV^j$.

For $\alpha\in\mathbb R$, the statistical **$\alpha$-connection** has
coefficients with the output index lowered

```{math}
:label: ig-alpha-expectation-definition
\begin{aligned}
\Gamma^{(\alpha)}_{ij,k}
 &=\mathbb E_\xi\!\left[
 \left(\partial_i\partial_j\ell_\xi
       +\frac{1-\alpha}{2}s_i s_j\right)s_k\right],\\
\Gamma^{(\alpha)k}_{ij}&=g^{k\ell}\Gamma^{(\alpha)}_{ij,\ell}.
\end{aligned}
```

The comma separates the lowered output index; these coefficients themselves
are not a covariant tensor. Under a change of chart, the second derivative
of $\ell$ contributes the inhomogeneous term in
[](#ig-christoffel-transformation). The coefficients are symmetric in $i,j$,
so all these connections are torsion-free. We use the convention
$\nabla^{(e)}=\nabla^{(1)}$ and $\nabla^{(m)}=\nabla^{(-1)}$.

We can identify their relation to Levi-Civita directly. Differentiation of an
expectation obeys
$\partial_i\mathbb E_\xi[F]=\mathbb E_\xi[\partial_iF+Fs_i]$.
Applying it to $g_{jk}$ gives

```{math}
\partial_i g_{jk}
=\mathbb E_\xi[(\partial_i\partial_j\ell)s_k]
 +\mathbb E_\xi[s_j(\partial_i\partial_k\ell)]+C_{ijk}.
```

Insert this into the lowered version of the Levi-Civita formula. The mixed
terms cancel, leaving

```{math}
:label: ig-alpha-lc-relation
\begin{aligned}
\Gamma^{(0)}_{ij,k}
 &=\mathbb E_\xi[(\partial_i\partial_j\ell)s_k]+\frac12C_{ijk},\\
\nabla^{(\alpha)}_UV
 &=\nabla^{\mathrm{LC}}_UV-\frac\alpha2 C^\sharp(U,V).
\end{aligned}
```

This proves that the $\alpha=0$ connection is Levi-Civita and expresses the
difference of connections as a tensor.

### Dual connections

Two connections $\nabla$ and $\nabla^*$ are **dual with respect to $g$** when

```{math}
:label: ig-dual-connection-identity
U[g(V,W)]=g(\nabla_UV,W)+g(V,\nabla^*_UW).
```

Substitute [](#ig-alpha-lc-relation) on the right, once with $\alpha$ and
once with $-\alpha$. The cubic terms cancel by symmetry, and Levi-Civita
compatibility supplies the left side. Thus $\nabla^{(\alpha)}$ and
$\nabla^{(-\alpha)}$ are dual, including the e/m pair. If $V$ is parallel
for one connection and $W$ for its dual along the same curve, $g(V,W)$ is
constant. Neither transport separately needs to preserve $g(V,V)$.
Indeed, $(\nabla^{(\alpha)}_U g)(V,W)=\alpha C(U,V,W)$, so nonzero $\alpha$
generally breaks metric compatibility.

These definitions and the sign convention agree with
[Amari and Nagaoka, *Methods of Information Geometry* (2000)](https://doi.org/10.1090/mmono/191)
and [Amari, *Information Geometry and Its Applications* (2016)](https://link.springer.com/book/10.1007/978-4-431-55978-8).
The calculations above establish the displayed relations under the stated
regularity assumptions.

### The normal alpha-connections

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

There is also an explicit transport comparison. Along a fixed-mean curve,
$\alpha$-parallel transport from scale $\sigma_0$ to $\sigma(r)$ gives

```{math}
V^\mu(r)=V^\mu(0)\left(\frac{\sigma(r)}{\sigma_0}\right)^{1+\alpha},
\qquad
V^\sigma(r)=V^\sigma(0)\left(\frac{\sigma(r)}{\sigma_0}\right)^{1+2\alpha}.
```

For the dual connection the exponents have the opposite $\alpha$ terms.
The paired inner product stays constant, although a vector's own Fisher norm
generally changes when $\alpha\ne0$.

### Natural and expectation coordinates make different connections flat

In the natural chart,
$\partial_i\partial_j\ell_\theta=-\partial_i\partial_j\psi$ is independent
of the observation. Since every score has expectation zero,
[](#ig-alpha-expectation-definition) at $\alpha=1$ gives
$\Gamma^{(e)k}_{ij}=0$. Also
$C_{ijk}=\partial_i\partial_j\partial_k\psi$ in these coordinates, by
differentiating the covariance of $t(X)$.

The expectation chart makes the dual mixture coefficients vanish. Here is a
coordinate argument. The natural coordinate vector fields
$E_i=\partial/\partial\theta^i$ are e-parallel. Let
$F_a=\partial/\partial\eta^a$. Since $d\theta=G_\theta^{-1}d\eta$,

```{math}
g(E_i,F_a)
=\sum_j (G_\theta)_{ij}(G_\theta^{-1})_{ja}
=\begin{cases}1,&i=a,\\0,&i\ne a.\end{cases}
```

The explicit sum here multiplies matrices labeled by two different coordinate
bases; the numeric identity compares their paired labels. Differentiate this
constant pairing with any field $U$ and use e/m duality. Because
$\nabla^{(e)}_UE_i=0$, we obtain
$g(E_i,\nabla^{(m)}_UF_a)=0$ for every $i$, hence
$\nabla^{(m)}_UF_a=0$. In particular, all m-Christoffel symbols vanish in
the $\eta$ chart.

Both connections therefore have zero curvature. This **dual flatness**
coexists with the curvature $-1/2$ of $\nabla^{\mathrm{LC}}$. Curvature is
attached to a specified connection; the vanishing of the e- or m-coefficients
in their affine charts does not make the Fisher metric Euclidean.

### Three geodesics between two normal laws

Let $P_0=N(\mu_0,\sigma_0^2)$ and $P_1=N(\mu_1,\sigma_1^2)$ with positive
scales, and let $0\leq r\leq1$. A straight natural-coordinate segment is an
affinely parameterized e-geodesic. In location-scale coordinates it is

```{math}
:label: ig-normal-e-geodesic
\begin{aligned}
\theta(r)&=(1-r)\theta_0+r\theta_1,\\
\sigma_e^2(r)&=\left(\frac{1-r}{\sigma_0^2}
                         +\frac r{\sigma_1^2}\right)^{-1},\\
\mu_e(r)&=\sigma_e^2(r)
 \left(\frac{(1-r)\mu_0}{\sigma_0^2}
                  +\frac{r\mu_1}{\sigma_1^2}\right).
\end{aligned}
```

It interpolates precision and precision-weighted mean. Its density is
proportional to $p_0^{1-r}p_1^r$, as follows by interpolating the natural form.
The precision stays positive, so the whole segment belongs to the family.

A straight expectation-coordinate segment is an affinely parameterized
m-geodesic of the connection on this normal family:

```{math}
:label: ig-normal-m-geodesic
\begin{aligned}
\eta(r)&=(1-r)\eta_0+r\eta_1,\\
\mu_m(r)&=(1-r)\mu_0+r\mu_1,\\
\sigma_m^2(r)&=(1-r)\sigma_0^2+r\sigma_1^2
                 +r(1-r)(\mu_1-\mu_0)^2.
\end{aligned}
```

The last term follows by subtracting the square of the interpolated mean
from the interpolated second raw moment. It keeps variance positive. This
normal has the same first two moments as $(1-r)P_0+rP_1$, but is generally
not that literal mixture: a mixture of two distinct normals need not be
normal. Thus m-flatness of this family does not assert closure under density
mixtures. Differentiating the two displayed paths verifies
[](#ig-normal-alpha-geodesic-odes) at $\alpha=1$ and $\alpha=-1$ respectively.

For comparison, the Levi-Civita segment is the vertical path or half-ellipse
in [](#ig-normal-lc-geodesic-solutions), with endpoints chosen as above. It
minimizes Fisher length; e- and m-geodesics are straight for their respective
connections and have no general Fisher-length-minimization property.

For $P_0=N(0,1)$ and $P_1=N(2,1)$ the difference is explicit:

| Connection | Path in $(\mu,\sigma)$ | Scale where $\mu=1$ |
| --- | --- | --- |
| e, $\alpha=1$ | $\mu=2r$, $\sigma=1$ | $1$ |
| Levi-Civita, $\alpha=0$ | $(\mu-1)^2/2+\sigma^2=3/2$ on the joining arc | $\sqrt{3/2}$ |
| m, $\alpha=-1$ | $\mu=2r$, $\sigma^2=1+4r(1-r)$ | $\sqrt2$ |

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
