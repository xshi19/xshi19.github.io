---
title: From Euclidean space to a manifold
---

A probability model can be described by several parameter vectors. A Bernoulli
law may use its success probability, its odds, or its log odds. A geometric
statement about the model should have the same meaning in each description.
This is the reason to introduce manifolds here.

## Coordinates and the geometry they describe

In Euclidean space, a displacement $v\in\mathbb R^d$ has squared length
$v^\mathsf{T}v$. This formula uses Cartesian coordinates and a chosen inner
product. After changing coordinates, the same inner product usually has a
different matrix.

For example, away from the origin and a chosen angular cut, write a point of
the plane as $x=r\cos\phi$, $y=r\sin\phi$. Differentiating gives

```{math}
\begin{aligned}
dx&=\cos\phi\,dr-r\sin\phi\,d\phi,\\
dy&=\sin\phi\,dr+r\cos\phi\,d\phi.
\end{aligned}
```

Squaring and adding cancels the cross terms:

```{math}
:label: ig-polar-metric
ds^2=dx^2+dy^2=dr^2+r^2d\phi^2.
```

The factor $r^2$ records the distance traveled by an angular displacement. The
underlying plane is still Euclidean. A metric matrix that varies with the
coordinates does not by itself establish curvature.

For a positive parameter $\lambda$, compare coordinates $\lambda$ and
$u=\log\lambda$. If the chosen metric is $ds^2=d\lambda^2$, then its expression
in $u$ is $ds^2=e^{2u}du^2$. Choosing $ds^2=du^2$ instead gives a different
metric: it measures relative changes in $\lambda$. Both choices define a
geometry; a statistical argument is needed to select one for a probability
model.

## The local role of a manifold

A smooth $d$-dimensional manifold is a space that can be described locally by
$d$ real coordinates, with smooth invertible changes of coordinates on overlaps.
A coordinate map on such a neighborhood is a **chart**. One chart need not
cover the whole space: an angle describes a circle locally, but a single real
angle with no identification cannot describe it globally and continuously.

An open parameter domain $\Theta\subset\mathbb R^d$ is already a manifold.
For a statistical model, the additional map

```{math}
\theta\longmapsto p_\theta
```

must be examined. Distinct parameter values can give the same law, and a
nonzero parameter velocity can sometimes have zero first-order effect on the
law. To use parameters as regular local coordinates for distributions, we need
local identifiability and a nondegenerate differential of this map. Merely
listing $d$ parameters does not establish a $d$-dimensional regular model.

## Tangent vectors are velocities

Let $\theta(t)$ be a smooth curve through $\theta(0)=\theta$. Its velocity
$v=\dot\theta(0)$ represents a tangent vector at that point. Curves with the
same first-order motion represent the same tangent vector; their accelerations
can differ.

Under coordinates $\phi=f(\theta)$, the chain rule gives

```{math}
:label: ig-tangent-coordinate-change
v_\phi=Df(\theta)\,v_\theta.
```

Thus a tangent vector is an underlying velocity with coordinate-dependent
components. A function $F$ on the manifold detects it by the directional
derivative

```{math}
v[F]=\left.\frac{d}{dt}F(\theta(t))\right|_{t=0}
=\sum_{i=1}^d v^i\partial_iF(\theta).
```

In a statistical model, applying this operation to $\log p_\theta(x)$ will
produce the directional score. That is the link to
[Fisher geometry](./information-geometry-fisher-vs-l2.md).

## A metric assigns inner products to tangent spaces

A Riemannian metric is a smoothly varying positive definite inner product
$g_\theta(v,w)$. In a chart it has a symmetric matrix $G_\theta$:

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
where both vectors live. The length of a piecewise smooth curve is

```{math}
L(\theta)=\int_a^b
\sqrt{\dot\theta(t)^\mathsf{T}G_{\theta(t)}\dot\theta(t)}\,dt.
```

Taking the infimum of these lengths over curves joining two points defines
their Riemannian distance within a connected component. A local quadratic form
and a finite distance are therefore related but different objects.

## Example: the interior probability simplex

A strictly positive distribution on $k$ outcomes is a vector

```{math}
\Delta_{k-1}^{\circ}
=\left\{p\in\mathbb R^k:p_i>0,\ \sum_{i=1}^k p_i=1\right\}.
```

It has $k-1$ free coordinates. One chart uses $p_1,\ldots,p_{k-1}$, with
$p_k=1-\sum_{i<k}p_i$. Another uses log odds

```{math}
\theta_i=\log\frac{p_i}{p_k},\qquad
p_i=\frac{e^{\theta_i}}{1+\sum_{j<k}e^{\theta_j}},\qquad
p_k=\frac{1}{1+\sum_{j<k}e^{\theta_j}}.
```

The second chart covers the interior with $\mathbb R^{k-1}$. Differentiating
$\sum_i p_i(t)=1$ shows that an ambient tangent velocity satisfies
$\sum_i\dot p_i=0$. The manifold structure alone has not yet assigned a length
to these velocities. Fisher information will supply a metric using the
probabilities themselves. Points with a zero probability lie on the boundary
and are outside these charts.

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

For further definitions, see §2 of Frank Nielsen's
[An elementary introduction to information geometry](https://arxiv.org/abs/1808.08271).
The calculations above require only the chain rule and inner products.

Continue with [exponential families](./information-geometry-exponential-families.md),
or return to the [Information Geometry hub](./information-geometry.md).
