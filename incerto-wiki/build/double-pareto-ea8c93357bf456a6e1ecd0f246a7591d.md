---
kernelspec:
  name: python3
  display_name: Python 3
---

# Double Pareto Distribution

## Statement

Let $Y$ have a Pareto Type I distribution with lower cutoff $1$ and tail
exponent $\alpha>0$.  Let $B$ be independent of $Y$ with
$\mathbb P(B=1)=\mathbb P(B=-1)=1/2$.  The symmetric double Pareto variable used
in `incerto` is

$$
X = B(Y-1).
$$

Its density on the real line is

$$
f(x;\alpha)=\frac{\alpha}{2}(1+|x|)^{-(\alpha+1)},\qquad x\in\mathbb R.
$$

The CDF is

$$
F(x)=
\begin{cases}
\frac12(1-x)^{-\alpha}, & x<0,\\
1-\frac12(1+x)^{-\alpha}, & x\ge 0.
\end{cases}
$$

For $x\ge0$, each one-sided tail has probability

$$
\mathbb P(X>x)=\frac12(1+x)^{-\alpha},
\qquad
\mathbb P(X<-x)=\frac12(1+x)^{-\alpha}.
$$

Consequently,

$$
\mathbb P(|X|>x)=(1+x)^{-\alpha}.
$$

The mean is $0$ by symmetry.  The variance is finite exactly when
$\alpha>2$, in which case

$$
\operatorname{Var}(X)=\frac{2}{(\alpha-1)(\alpha-2)}.
$$

## Intuition

The ordinary Pareto distribution is one-sided: the rare extreme is always on
the right.  The double Pareto keeps the same power-law magnitude but gives the
shock a sign.  This makes it useful for toy return models where both large gains
and large losses are possible, while preserving a transparent tail exponent.

## Proof

For $x\ge0$,

$$
\mathbb P(X>x)
=\mathbb P(B=1, Y-1>x)
=\frac12\mathbb P(Y>1+x)
=\frac12(1+x)^{-\alpha}.
$$

The negative tail is identical by symmetry.  Differentiating the CDF on either
side gives the density.  The mean is zero whenever the first absolute moment
exists, and symmetry gives the centered value used by SciPy's distribution
interface.  Since $X^2=(Y-1)^2$,

$$
\mathbb E[X^2]
=\mathbb E[Y^2]-2\mathbb E[Y]+1
=\frac{\alpha}{\alpha-2}-2\frac{\alpha}{\alpha-1}+1
=\frac{2}{(\alpha-1)(\alpha-2)}
$$

for $\alpha>2$.  If $\alpha\le2$, the second moment of $Y$ is infinite, so the
variance of $X$ is infinite.

## Python

```{code-cell} python
:label: double-pareto-python-check

import numpy as np
from incerto.distributions import double_pareto

alpha = 3.0
x = np.array([-4.0, 0.0, 4.0])
pdf = double_pareto.pdf(x, alpha)
cdf = double_pareto.cdf(x, alpha)
mean, variance = double_pareto.stats(alpha, moments="mv")

print(f"x grid: {x}")
print(f"PDF values: {np.round(pdf, 6)}")
print(f"CDF values: {np.round(cdf, 6)}")
print(f"Mean and variance: {(float(mean), float(variance))}")
```

The implementation is consolidated in `incerto.distributions`, with the
duplicate class definition removed and the formulas vectorized.

## Caveats

- This page describes the symmetric shifted construction in `incerto`, not every
  distribution called "double Pareto" in the literature.
- For $\alpha\le1$, even the absolute first moment is infinite.  Symmetry can
  make a formal location look harmless while absolute exposure remains
  uncontrolled.
- The distribution is a pedagogical model.  Empirical returns usually need
  skew, truncation, volatility clustering, dependence, or threshold modeling
  before a tail fit is credible.

## References

- Taleb, *Statistical Consequences of Fat Tails*, Chapter 3
  [@taleb2020scoft].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].

## Backlinks

- Depends on: [Pareto Distribution](pareto.md) and
  [Regular Variation](../theorems/regular-variation.md).
- Used by: Phase 1 distribution tests and future two-sided tail examples.

<!-- incerto-provenance:start -->
:::{div} incerto-provenance
**Provenance.** Source: `content/concepts/distributions/double-pareto.md`. Last verified: 2026-05-24. Checked against cited sources, page proof or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
