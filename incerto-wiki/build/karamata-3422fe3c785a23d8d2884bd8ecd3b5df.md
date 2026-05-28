# Karamata's Theorem

## Statement

Let $L$ be slowly varying at infinity. Karamata's theorem gives the first-order
asymptotics of integrals of regularly varying functions.

If $\rho>-1$, then

$$
\int_0^x t^\rho L(t)\,dt
\sim
\frac{x^{\rho+1}L(x)}{\rho+1},
\qquad x\to\infty.
$$

If $\rho<-1$, then

$$
\int_x^\infty t^\rho L(t)\,dt
\sim
\frac{x^{\rho+1}L(x)}{-\rho-1},
\qquad x\to\infty.
$$

For a nonnegative random variable with survival tail
$\bar F(x)=x^{-\alpha}L(x)$, this converts tail exponents into moment
boundaries. For $p<\alpha$,

$$
\mathbb E[X^p\mathbf 1_{\{X>x\}}]
\sim
\frac{\alpha}{\alpha-p}x^p\bar F(x),
\qquad x\to\infty.
$$

For $p>\alpha$, the truncated lower moment grows like

$$
\mathbb E[X^p\mathbf 1_{\{X\le x\}}]
\sim
\frac{\alpha}{p-\alpha}x^p\bar F(x),
\qquad x\to\infty.
$$

The borderline $p=\alpha$ is logarithmic for an exact Pareto tail and needs
separate treatment for a general slowly varying $L$.

## Intuition

Regular variation says that, far enough in the tail, powers dominate slowly
varying corrections. Karamata's theorem says the same thing remains true after
integration: the integral is controlled by the endpoint and the power index.

That is why a single tail exponent can decide whether a mean, variance, or
higher moment exists. The distribution does not need to be exactly Pareto; an
asymptotically Pareto survival tail is enough, away from the logarithmic
boundary.

## Proof

A full proof uses the uniform convergence theorem for slowly varying functions.
The essential idea is visible in the case $\rho>-1$. Write

$$
\int_0^x t^\rho L(t)\,dt
=
x^{\rho+1}L(x)\int_0^1 u^\rho\frac{L(xu)}{L(x)}\,du.
$$

For each fixed $u>0$, slow variation gives $L(xu)/L(x)\to1$. Uniform
convergence on compact subintervals of $(0,\infty)$ and Potter-type bounds
control the small-$u$ part of the integral. The limit becomes

$$
\int_0^1 u^\rho\,du=\frac{1}{\rho+1}.
$$

The upper-tail case follows the same change of variables:

$$
\int_x^\infty t^\rho L(t)\,dt
=
x^{\rho+1}L(x)\int_1^\infty u^\rho\frac{L(xu)}{L(x)}\,du,
$$

and the integral over $[1,\infty)$ is finite exactly when $\rho<-1$.

For the moment implication, use the tail integration identity

$$
\mathbb E[X^p\mathbf 1_{\{X>x\}}]
=
x^p\bar F(x)+p\int_x^\infty t^{p-1}\bar F(t)\,dt.
$$

Substituting $\bar F(t)=t^{-\alpha}L(t)$ gives an integral with
$\rho=p-\alpha-1$. When $p<\alpha$, Karamata's upper-tail form yields

$$
p\int_x^\infty t^{p-\alpha-1}L(t)\,dt
\sim
\frac{p}{\alpha-p}x^{p-\alpha}L(x).
$$

Adding the endpoint term $x^{p-\alpha}L(x)$ gives the coefficient
$\alpha/(\alpha-p)$. The truncated lower-moment formula is the analogous
lower-tail integration-by-parts identity plus Karamata with
$\rho=p-\alpha-1>-1$.

## Python

The snippet compares Karamata's upper-tail approximation for a tail
$\bar F(x)=x^{-\alpha}\log x$ with a numerical integral. The slowly varying
factor is not constant, but the ratio still approaches 1.

```python
import numpy as np
from scipy.integrate import quad

alpha = 1.7
p = 1.0

def survival(t):
    return t ** (-alpha) * np.log(t)

def tail_moment(x):
    endpoint = x**p * survival(x)
    integral, _ = quad(lambda t: p * t ** (p - 1) * survival(t), x, np.inf)
    return endpoint + integral

for x in [10, 100, 1000, 10000]:
    approximation = alpha / (alpha - p) * x**p * survival(x)
    print(x, tail_moment(x) / approximation)
```

## Caveats

- The theorem is asymptotic. It does not say where the tail approximation
  becomes accurate in finite data.
- The endpoint power condition matters. The case $\rho=-1$, corresponding to
  $p=\alpha$ for moment boundaries, is not covered by the displayed forms.
- A slowly varying factor can change finite-sample behavior substantially even
  when it does not change the leading moment boundary.
- The moment statements here assume a nonnegative random variable and a
  regularly varying survival tail. Two-sided variables need separate treatment
  for positive and negative tails or for absolute moments.

## References

- Bingham, Goldie, and Teugels, *Regular Variation*
  [@bingham1987regular].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].

## Backlinks

- Depends on: [Regular Variation](regular-variation.md).
- Used by: [Pareto Moment Existence](pareto-moment-existence.md),
  [LLN Failure Under Infinite Mean](lln-failure.md), and the
  [Chapter 5 reading guide](../../reading-guides/taleb-scoft/ch5.md).
