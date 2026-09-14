---
title: From Euclidean space to a manifold
---

A normal distribution can be described by its mean and standard deviation,
or by its mean and variance. The coordinate vectors differ, but the
distribution is the same. We use this two-parameter family to explain what
tangent vectors and metrics describe, and how they behave when coordinates
change. Symbols follow the
[shared notation](https://xshi19.github.io/math/notation/).

## A running example: normal location and scale

Let $X\sim N(\mu,\sigma^2)$, where $\mu\in\mathbb R$ is the mean and
$\sigma>0$ is the standard deviation. With respect to Lebesgue measure on
$\mathbb R$, the density is

```{math}
\begin{gathered}
p_\theta(x)=\frac{1}{\sqrt{2\pi}\,\sigma}
\exp\!\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\},
\\
\theta=(\mu,\sigma)^\mathsf{T}\in\Theta=\mathbb R\times(0,\infty).
\end{gathered}
```

Both parameters vary. Changing $\mu$ translates the density; changing
$\sigma$ changes its spread. Thus even though each observation is a single
real number, the family of laws is two-dimensional. Here $\theta$ denotes
location and scale coordinates; the
[exponential-family note](./information-geometry-exponential-families.md#two-examples)
later expresses the same laws in natural coordinates.

We can instead use $\phi=(\mu,\tau)^\mathsf{T}$, where the local symbol
$\tau=\sigma^2>0$ denotes variance. The maps
$(\mu,\sigma)\mapsto(\mu,\sigma^2)$ and
$(\mu,\tau)\mapsto(\mu,\sqrt\tau)$ are smooth inverses on their domains.
They relabel the same normal laws. The case $\sigma=0$ is excluded: a point
mass has no density of the displayed form.

## Coordinates and the geometry they describe

In Euclidean space, a displacement $v\in\mathbb R^d$ has squared length
$v^\mathsf{T}v$. This formula uses Cartesian coordinates and a chosen inner
product. After changing coordinates, the same inner product usually has a
different matrix.

For example, away from the origin and a chosen angular cut, write a point of
the plane using radius $r>0$ and polar angle $\varphi$ as
$x=r\cos\varphi$, $y=r\sin\varphi$. Differentiating gives

```{math}
\begin{aligned}
dx&=\cos\varphi\,dr-r\sin\varphi\,d\varphi,\\
dy&=\sin\varphi\,dr+r\cos\varphi\,d\varphi.
\end{aligned}
```

Squaring and adding cancels the cross terms:

```{math}
:label: ig-polar-metric
ds^2=dx^2+dy^2=dr^2+r^2d\varphi^2.
```

The factor $r^2$ records the distance traveled by an angular displacement. The
underlying plane is still Euclidean. A metric matrix that varies with the
coordinates does not by itself establish curvature.

For the normal family, the coordinate displacement $(d\mu,d\sigma)$ becomes
$(d\mu,d\tau)=(d\mu,2\sigma\,d\sigma)$. Assigning the sum of squared
components in each chart would give different lengths to the same change in
the law. We need a statistical criterion for choosing the metric, and a
coordinate transformation rule for expressing that one metric in both charts.

## The local role of a manifold

A smooth $d$-dimensional manifold is a space that can be described locally by
$d$ real coordinates, with smooth invertible changes of coordinates on overlaps.
A coordinate map on such a neighborhood is a **chart**. One chart need not
cover the whole space: an angle describes a circle locally, but a single real
angle with no identification cannot describe it globally and continuously.

An open parameter domain $\Theta\subset\mathbb R^d$ is already a manifold.
For a statistical model, the map from parameters to densities,

```{math}
\theta\longmapsto p_\theta
```

must also be examined. Distinct parameter values can give the same law, and a
nonzero parameter velocity can sometimes have zero first-order effect on the
law. To use parameters as regular local coordinates for distributions, we need
local identifiability and a nondegenerate differential of this map. Merely
listing $d$ parameters does not establish a $d$-dimensional regular model.
For the normal family, the mean and variance determine the law uniquely.
The score calculation below will also show that every nonzero parameter
velocity has a nonzero first-order effect on the density.

## Tangent vectors are velocities

Let $\theta(\varepsilon)$ be a smooth curve through $\theta(0)=\theta$.
Its velocity $v=\dot\theta(0)$, with the dot denoting differentiation in
$\varepsilon$, represents a tangent vector at that point. Curves with the
same first-order motion represent the same tangent vector; their accelerations
can differ.

For the normal family, write $v=(v^\mu,v^\sigma)^\mathsf{T}$, where
$v^\mu=\dot\mu(0)$ and $v^\sigma=\dot\sigma(0)$. The superscripts name
components, not powers. The curve

```{math}
\theta(\varepsilon)=
\begin{pmatrix}\mu+\varepsilon v^\mu\\\sigma+\varepsilon v^\sigma\end{pmatrix}
```

realizes this velocity for sufficiently small $\varepsilon$ with
$\sigma+\varepsilon v^\sigma>0$. The vector $(1,0)^\mathsf{T}$ changes the
mean while holding the standard deviation fixed; $(0,1)^\mathsf{T}$ changes
the standard deviation while holding the mean fixed. A general $v$ combines
these motions. Its components specify rates of perturbation of the existing
parameters.

Under a smooth coordinate change $\phi=f(\theta)$, let $Df(\theta)$ denote
its Jacobian matrix. The chain rule gives

```{math}
:label: ig-tangent-coordinate-change
v_\phi=Df(\theta)\,v_\theta.
```

For the variance chart this reads

```{math}
v_\phi=
\begin{pmatrix}1&0\\0&2\sigma\end{pmatrix}v_\theta
=\begin{pmatrix}v^\mu\\2\sigma v^\sigma\end{pmatrix}.
```

For example, at $\sigma=2$, increasing the standard deviation at unit rate
increases the variance at rate $4$. These component vectors describe the same
tangent vector. A smooth scalar function $F$ on the manifold detects that
vector by its directional derivative:

```{math}
v[F]=\left.\frac{d}{d\varepsilon}F(\theta(\varepsilon))\right|_{\varepsilon=0}
=\sum_{i=1}^d v^i\partial_iF(\theta).
```

Here $\partial_i$ differentiates with respect to the $i$th parameter
coordinate. Applying this operation to $F(\theta)=\log p_\theta(x)$ for a
fixed $x$ gives the **directional score** $v^\mathsf{T}s_\theta(x)$, where
$s_\theta(x)=\nabla_\theta\log p_\theta(x)$. Equivalently, the density itself
changes at rate

```{math}
\left.\frac{d}{d\varepsilon}p_{\theta(\varepsilon)}(x)
\right|_{\varepsilon=0}
=p_\theta(x)\,v^\mathsf{T}s_\theta(x).
```

This identifies the statistical effect of a parameter velocity. We next use
it to measure the size of that velocity.

## A metric assigns inner products to tangent spaces

A Riemannian metric assigns a smoothly varying positive definite inner product
$g_\theta(v,w)$ to pairs of tangent vectors $v,w$ at the same point $\theta$.
In a chart it has a symmetric matrix $G_\theta$:

```{math}
g_\theta(v,w)=v_\theta^\mathsf{T}G_\theta w_\theta.
```

If $\theta=\theta(\phi)$ and $J=\partial\theta/\partial\phi$, then
$v_\theta=Jv_\phi$. Substitution gives the transformation rule

```{math}
:label: ig-metric-coordinate-change
G_\phi=J^\mathsf{T}G_\theta J.
```

The number $g(v,w)$ is unchanged. Orthogonality means $g(v,w)=0$ at the point
where both vectors live. For a piecewise smooth curve $\theta(t)$,
$a\le t\le b$, the length is

```{math}
L(\theta)=\int_a^b
\sqrt{\dot\theta(t)^\mathsf{T}G_{\theta(t)}\dot\theta(t)}\,dt.
```

Taking the infimum of these lengths over curves joining two points defines
their Riemannian distance within a connected component. A local quadratic form
and a finite distance are therefore related but different objects.

## Fisher lengths in the normal family

We now derive the normal family's Fisher metric by averaging products of
directional scores. Differentiating the log density gives

```{math}
s_\theta(x)=
\begin{pmatrix}
\dfrac{x-\mu}{\sigma^2}\\[4pt]
-\dfrac1\sigma+\dfrac{(x-\mu)^2}{\sigma^3}
\end{pmatrix}.
```

Put $Z=(X-\mu)/\sigma\sim N(0,1)$. Then the two score components are
$Z/\sigma$ and $(Z^2-1)/\sigma$, and the directional score is

```{math}
v^\mathsf{T}s_\theta(X)
=\frac{v^\mu Z+v^\sigma(Z^2-1)}{\sigma}.
```

The mean component is odd in $Z$: it shifts probability from one side of the
mean to the other. The scale component is negative for $|Z|<1$ and positive
for $|Z|>1$: increasing $\sigma$ lowers the density near the mean and raises
it in the tails. These two changes in the density are linearly independent.

Symmetry gives $\mathbb E[Z]=\mathbb E[Z^3]=0$, while
$\mathbb E[Z^2]=1$ and $\mathbb E[Z^4]=3$. The fourth moment follows by
integrating by parts using the derivative $-z p_{(0,1)}(z)$ of the standard
normal density: $\mathbb E[Z^4]=3\mathbb E[Z^2]$. Consequently,

```{math}
\begin{aligned}
\mathbb E[Z(Z^2-1)]&=0,\\
\mathbb E[(Z^2-1)^2]&=3-2+1=2.
\end{aligned}
```

Taking the expected outer product of the score vector therefore yields the
Fisher information matrix per observation. For a second velocity
$w=(w^\mu,w^\sigma)^\mathsf{T}$, it also gives the inner product:

```{math}
:label: ig-normal-fisher-metric
\begin{aligned}
I(\theta)&=\mathbb E_\theta[s_\theta(X)s_\theta(X)^\mathsf{T}]\\
&=\begin{pmatrix}\sigma^{-2}&0\\0&2\sigma^{-2}\end{pmatrix},\\[4pt]
g_\theta(v,w)&=\frac{v^\mu w^\mu+2v^\sigma w^\sigma}{\sigma^2}.
\end{aligned}
```

This matrix is smooth and positive definite for every $\sigma>0$. In
particular, a nonzero $v$ has a directional score with positive mean square,
so its first-order density change cannot vanish almost everywhere. Normal
densities have common positive support and finite moments of every order;
on a bounded parameter neighborhood with $\sigma$ bounded away from zero,
their derivatives have integrable Gaussian bounds. The regularity needed for
this Fisher calculation holds throughout $\Theta$.

The Fisher speed of the perturbation is

```{math}
\sqrt{g_\theta(v,v)}
=\frac{\sqrt{(v^\mu)^2+2(v^\sigma)^2}}{\sigma}.
```

It is the root mean square rate of change in log density under the current
law. At $\sigma=2$, the velocities $(1,0)^\mathsf{T}$ and
$(0,1)^\mathsf{T}$ have Fisher speeds $1/2$ and $1/\sqrt2$, respectively,
and they are Fisher-orthogonal. For a fixed mean velocity, doubling the
current standard deviation halves the speed: a shift of the mean produces a
smaller relative change in a broader density. Fixing $\sigma$ instead gives
the one-parameter subfamily with metric $d\mu^2/\sigma^2$.

In the variance chart, the inverse Jacobian is

```{math}
J=\frac{\partial(\mu,\sigma)}{\partial(\mu,\tau)}
=\begin{pmatrix}1&0\\0&1/(2\sigma)\end{pmatrix}.
```

Using [](#ig-metric-coordinate-change) with $G_\theta=I(\theta)$ gives

```{math}
:label: ig-normal-fisher-charts
\begin{aligned}
G_\phi&=\begin{pmatrix}\tau^{-1}&0\\0&(2\tau^2)^{-1}\end{pmatrix},\\[4pt]
ds^2&=\frac{(d\mu)^2+2(d\sigma)^2}{\sigma^2}\\
&=\frac{(d\mu)^2}{\tau}+\frac{(d\tau)^2}{2\tau^2}.
\end{aligned}
```

Substituting $v_\phi=(v^\mu,2\sigma v^\sigma)^\mathsf{T}$ into
$v_\phi^\mathsf{T}G_\phi v_\phi$ recovers [](#ig-normal-fisher-metric).
Both coordinate systems assign the same speed to the same statistical
perturbation. The [Fisher note](./information-geometry-fisher-vs-l2.md)
derives the general score construction and relates it to local KL divergence.

## Comparing velocities at different points

A metric compares tangent vectors at the same point. A **connection** adds a
rule for differentiating vector fields along curves and transporting vectors
between tangent spaces. A connection geodesic has velocity parallel along
itself, meaning its covariant acceleration vanishes.

The metric determines a distinguished torsion-free, metric-compatible
connection, the **Levi-Civita connection**. Its geodesics minimize length on
sufficiently short segments. Information geometry also uses a pair of dual
connections. Their geodesics can differ from the Levi-Civita geodesics; their
curvature can differ as well. The
[duality note](./information-geometry-duality.md) makes this distinction
explicit for exponential families.

The connection results in this section are stated without proof; see §2 of
Frank Nielsen's
[An elementary introduction to information geometry](https://arxiv.org/abs/1808.08271).
The tangent-vector and normal-metric calculations above use the chain rule,
inner products, and normal moments.

Continue with [exponential families](./information-geometry-exponential-families.md),
or return to the [Information Geometry hub](./ig/index.md).
