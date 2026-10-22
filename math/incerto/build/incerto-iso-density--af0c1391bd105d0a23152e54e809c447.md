---
title: Iso-Density Tail Geometry
options:
  concept:
    id: iso-density-tail-geometry
    type: method
    depends_on:
      - survival-tail-ratio
      - subexponentiality
    related:
      - body-shoulder-tail
      - max-to-sum-ratio
      - multivariate-tails
    tags:
      - diagnostics
      - fat-tails
      - geometry
---

(statement)=
## Independent coordinates

Iso-density tail geometry studies the level sets of the iid joint density

$$
f_{X_1,X_2}(x_1,x_2)=f(x_1)f(x_2)
$$

near a large-deviation constraint such as $x_1+x_2=s$.  The diagnostic asks
whether the most likely configurations along the constraint sit near the
diagonal, where both coordinates are moderately large, or near the axes, where
one coordinate is large and the other is ordinary.

Here $X_1,X_2$ are independent copies of a continuous random variable with
density $f$, and $x_1,x_2$ are its coordinate values. The sum level $s$ is
fixed for each comparison. See [Shared notation](https://xshi19.github.io/math/notation/).

## Normal versus Cauchy geometry

For two iid standard normal variables, the joint density is proportional to

$$
\exp\left[-\frac{x_1^2+x_2^2}{2}\right].
$$

On the line $x_1+x_2=s$,

$$
x_1^2+x_2^2
=
2\left(x_1-\frac{s}{2}\right)^2+\frac{s^2}{2},
$$

so the joint density is maximized at the equal split
$(s/2,s/2)$.  A large Gaussian sum is geometrically a many-moderate-deviation
event.

For two iid standard Cauchy variables,

$$
f(x)=\frac{1}{\pi(1+x^2)}.
$$

Compare the equal split $(s/2,s/2)$ with an axial split $(0,s)$.  The axial to
equal density ratio is

$$
\frac{f(0)f(s)}{f(s/2)^2}
=
\frac{(1+s^2/4)^2}{1+s^2}
\sim
\frac{s^2}{16}
\to\infty.
$$

Along a large-sum line, the Cauchy geometry increasingly favors one large
coordinate rather than two equal coordinates.  This is the density-contour
version of the one-big-jump intuition in
[Subexponentiality](./incerto-subexponentiality.md) and the
[Survival Tail Ratio](./incerto-survival-tail-ratio.md).

## Static density comparison

(iso-density-tail-geometry-contours)=
For the same sum level $s$, the normal and Cauchy axial-to-equal ratios are

$$
R_{\mathrm N}(s)=e^{-s^2/4},
\qquad
R_{\mathrm C}(s)=\frac{(1+s^2/4)^2}{1+s^2}.
$$

| Sum level $s$ | Normal ratio $R_{\mathrm N}(s)$ | Cauchy ratio $R_{\mathrm C}(s)$ |
| --- | --- | --- |
| $4$ | $0.018316$ | $25/17\approx1.471$ |
| $8$ | $1.1254\times10^{-7}$ | $289/65\approx4.446$ |
| $12$ | $2.3195\times10^{-16}$ | $1369/145\approx9.441$ |

The table evaluates exact density ratios, without estimating probabilities.
Normal equal splits dominate axial splits more strongly as $s$ increases;
Cauchy axial splits eventually dominate equal splits.

The axial points are comparison points, not the exact Cauchy maximizers for
finite positive $s$. To locate those maximizers, minimize the denominator
$D(x)=(1+x^2)(1+(s-x)^2)$ along the line. Its derivative factors as

$$
D'(x)=2(2x-s)(x^2-sx+1).
$$

For $s>2$, the two minima occur at

$$
x=\frac{s\pm\sqrt{s^2-4}}{2},
$$

so the maximizing pairs approach $(1/s,s-1/s)$ and its swap as $s$ grows.
They sit near the axes, while the equal split becomes a local density minimum.
For $0\le s\le2$, the equal split maximizes the constrained density.

This algebra describes density along a constraint of probability zero. It does
not prove a tail probability asymptotic. The Cauchy example is two-sided;
applying the nonnegative-summand subexponential theorem to it requires a
separate right-tail argument.

(tail-geometry-elliptical)=
## The same Cauchy marginals, a different joint law

The preceding calculation proves a comparison under **independence**. A
spherical bivariate Cauchy has instead the normalized density

$$
g(x_1,x_2)=\frac{1}{2\pi}(1+x_1^2+x_2^2)^{-3/2}.
$$

The [multivariate Student construction](incerto-multivariate-tails.md#multivariate-student)
derives its normalization and Cauchy marginals at $\nu=1$. Its density is
strictly decreasing in squared radius. The same completed-square identity
used for the Gaussian therefore proves that, on $x_1+x_2=s$, its density
is maximized at the equal split for every $s$. Heavy marginal tails alone
do not determine the constrained density geometry.

There is a probability comparison as well. In polar coordinates, integrating
$g$ over the full circle gives

$$
P(X_1^2+X_2^2>u^2)=\int_u^\infty r(1+r^2)^{-3/2}\,dr
=\frac{1}{\sqrt{1+u^2}},\qquad u\ge0.
$$

Angle is uniform and independent of radius because the joint polar density
factorizes. A sector of angle width $\delta\in[0,2\pi]$, outside that
circle, has probability $\delta/(2\pi\sqrt{1+u^2})$.
At $u=3$ and $\delta=\pi/3$, this is $1/(6\sqrt{10})\approx0.052705$.
Unlike a point-density ratio, this quantity integrates over a region of
positive area. It is not a large-sum probability.

(tail-geometry-switching)=
## Switching correlation can destroy ellipticality

Consider an equal mixture of two centered bivariate normals with unit
marginal variances and correlations $r$ and $-r$, where $0<r<1$.
Its covariance is $I_2$, and conditioning gives characteristic function

$$
\varphi(t_1,t_2)=e^{-(t_1^2+t_2^2)/2}\cosh(rt_1t_2).
$$

We prove that this mixture is not elliptical. For a nondegenerate elliptical
law with finite covariance, covariance is a positive scalar multiple of
scatter: whitening gives a rotationally invariant law, whose covariance
commutes with every orthogonal matrix and hence is scalar. Thus an
elliptical law with covariance $I_2$ must be spherical. But the displayed
characteristic function at $(v,0)$ is $e^{-v^2/2}$, whereas at
$(v/\sqrt2,v/\sqrt2)$ it is $e^{-v^2/2}\cosh(rv^2/2)$.
For $v\ne0$ these differ at the same radius. This disproves sphericality
and hence ellipticality. Both marginal laws remain standard normal.

The iid Cauchy vector gives another distinction: every linear combination
is Cauchy, with scale $|a|+|b|$ for coefficients $a,b$ not both zero,
since its characteristic function is $e^{-|t_1|-|t_2|}$.
It is a stable vector but not elliptical. To see the latter without moments,
its characteristic function on either axis would force the elliptical
profile to be exponential in the square root of its argument. It is also
unchanged when one coordinate changes sign, forcing off-diagonal scatter
zero. The equal marginal scales would then force circular level sets;
$|t_1|+|t_2|$ instead has diamond level sets. Thus closure under linear
combinations does not require ellipticality.

```{figure} ../assets/figures/incerto/multivariate-geometry.svg
:alt: Three vertically stacked contour panels compare independent Cauchy, spherical Cauchy, and a normal mixture with correlations plus and minus 0.8. Each legend identifies contours at 5, 15, 35 and 65 percent of that panel's density at the origin.

Exact density contours on the same coordinate window. The first two laws
have identical standard Cauchy marginals but different contours. In the
third, both marginals are standard normal and covariance is the identity,
yet the contours are not elliptical. The color and line-style keys identify
levels at 5%, 15%, 35% and 65% of each panel's density at the origin;
these are density levels, not equal enclosed probabilities.
The finite window is a plotting range, not a truncation of any law.
```

Use [`bivariate_student`](incerto-api-reference.md#api-bivariate-student).

```python
import numpy as np
from xmath.incerto.multivariate import bivariate_student

rng = np.random.default_rng(20261020)
sample = bivariate_student(rng, nu=1, rho=0, size=10000)
radius = np.linalg.norm(sample, axis=1)
angle = np.arctan2(sample[:, 1], sample[:, 0])
sector = (radius > 3) & (angle >= 0) & (angle <= np.pi / 3)
print(f"Sector frequency: {sector.mean():.4g}")
print(f"Exact probability: {1 / (6 * np.sqrt(10)):.4g}")
```

**Output**

```text
Sector frequency: 0.0533
Exact probability: 0.0527
```

The 10,000 seeded draws illustrate an integrated probability, not a density contour level.

Figure regeneration: `uv run python demos/incerto/multivariate_tails.py` for the contours,
the sector integral and the earlier constrained-density ratios. The chapter
6 guide links the [sample-correlation and information calculations](incerto-multivariate-tails.md).

## Caveats

- Iso-density geometry requires a density.  Discrete, singular, censored, or
  heavily rounded data need a different diagnostic.
- Density at a point is not probability mass.  The contour picture explains
  local geometry; probability statements require integration over regions.
- The iid product-density formula fails under dependence.  Volatility
  clustering, common factors, and contagion can rotate or bend the contours.
- The examples here are symmetric and centered.  Skewed losses or one-sided
  positive variables should first be put into a problem-specific coordinate
  system.

## References

- @taleb2025scoftv4, third edition (2025), arXiv:2001.10488v4,
  §§6.2–6.3.1, pp. 109–114: ellipticality and independent versus shared-scale
  Cauchy context. The constrained extrema, sector integral, switching example
  and stable-vector counterexample above are proved here.
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).

## Backlinks

- Depends on: [Survival Tail Ratio](./incerto-survival-tail-ratio.md) and
  [Subexponentiality](./incerto-subexponentiality.md).
- Related diagnostic: [Max-to-Sum Ratio](./incerto-max-to-sum-ratio.md).
- Related density geometry: [Body, Shoulders, and Tails](./incerto-body-shoulder-tail.md).
