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
    tags:
      - diagnostics
      - fat-tails
      - geometry
---

## Statement

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
fixed for each comparison. Shared notation is planned.

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

- Taleb, *Statistical Consequences of Fat Tails* [2020](https://arxiv.org/abs/2001.10488).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).

## Backlinks

- Depends on: [Survival Tail Ratio](./incerto-survival-tail-ratio.md) and
  [Subexponentiality](./incerto-subexponentiality.md).
- Related diagnostic: [Max-to-Sum Ratio](./incerto-max-to-sum-ratio.md).
- Related density geometry: [Body, Shoulders, and Tails](./incerto-body-shoulder-tail.md).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/methods/iso-density-tail-geometry.md`, revision `9717c9c`
(2026-09-13 import). Copyright (c) 2023 xshi19. Licensed under MIT.
Links and notation were adapted for this site; executable figures and simulations
were replaced with static calculations. No upstream execution or formal-proof
verification is claimed for this adaptation.

:::{dropdown} MIT permission notice

MIT License

Copyright (c) 2023 xshi19

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
