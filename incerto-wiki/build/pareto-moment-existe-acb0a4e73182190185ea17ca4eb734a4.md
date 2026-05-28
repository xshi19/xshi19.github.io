---
kernelspec:
  name: python3
  display_name: Python 3
---

# Pareto Moment Existence

## Statement

Let $X$ have a Pareto Type I distribution with lower cutoff $x_m>0$ and tail
exponent $\alpha>0$:

$$
\bar F(x)=\mathbb P(X>x)=\left(\frac{x_m}{x}\right)^\alpha,
\qquad x\ge x_m.
$$

For any moment order $p>0$,

$$
\mathbb E[X^p] < \infty
\quad\Longleftrightarrow\quad
p<\alpha.
$$

When $p<\alpha$,

$$
\mathbb E[X^p]=\frac{\alpha x_m^p}{\alpha-p}.
$$

When $p=\alpha$, the truncated moment diverges logarithmically:

$$
\mathbb E[X^\alpha\mathbf 1_{\{X\le x\}}]
=
\alpha x_m^\alpha\log\frac{x}{x_m},
\qquad x\ge x_m.
$$

When $p>\alpha$, the truncated moment grows as a power:

$$
\mathbb E[X^p\mathbf 1_{\{X\le x\}}]
=
\frac{\alpha x_m^\alpha}{p-\alpha}
\left(x^{p-\alpha}-x_m^{p-\alpha}\right).
$$

In particular, the Pareto mean exists only for $\alpha>1$, and the variance
exists only for $\alpha>2$.

## Intuition

The density contributes a factor $x^{-(\alpha+1)}$, while the $p$th moment
multiplies by $x^p$. The tail integral therefore behaves like
$\int x^{p-\alpha-1}\,dx$. A moment exists only when the remaining exponent is
small enough for the integral at infinity to converge.

The failure cases are not numerical accidents. At $p=\alpha$, every logarithmic
scale contributes roughly the same amount. At $p>\alpha$, later and later
scales contribute more, so a larger sample can reveal a much larger empirical
moment instead of stabilizing the old estimate.

## Proof

For $x\ge x_m$, the Pareto density is

$$
f(x)=\alpha x_m^\alpha x^{-(\alpha+1)}.
$$

For $p>0$,

$$
\mathbb E[X^p]
=
\int_{x_m}^{\infty}x^p\alpha x_m^\alpha x^{-(\alpha+1)}\,dx
=
\alpha x_m^\alpha
\int_{x_m}^{\infty}x^{p-\alpha-1}\,dx.
$$

The integral converges exactly when $p-\alpha-1<-1$, equivalently
$p<\alpha$. Evaluating the convergent case gives

$$
\alpha x_m^\alpha
\cdot\frac{x_m^{p-\alpha}}{\alpha-p}
=
\frac{\alpha x_m^p}{\alpha-p}.
$$

For the truncated cases, integrate only from $x_m$ to $x$. If $p=\alpha$, the
integrand is $\alpha x_m^\alpha x^{-1}$, giving logarithmic growth. If
$p>\alpha$, direct integration gives the displayed power-growth formula.

## Python

The sample mean can look calm for a while when $\alpha$ is just above 1, but
the theoretical boundary is exact.

```{code-cell} python
:label: pareto-moment-existence-python-check

import numpy as np

from incerto.distributions import pareto_type1

rng = np.random.default_rng(20260523)
alpha = 1.2
x_m = 1.0

sample = pareto_type1.rvs(alpha, scale=x_m, size=50_000, random_state=rng)
running_mean = np.cumsum(sample) / np.arange(1, sample.size + 1)
theoretical_mean = alpha / (alpha - 1)
sample_second_moment = np.mean(sample**2)

print(f"Theoretical mean for alpha={alpha}: {theoretical_mean:.3f}")
print(f"Last five running means: {np.round(running_mean[-5:], 3)}")
print(
    "Sample second raw moment "
    f"(variance is infinite for alpha <= 2): {sample_second_moment:.3f}"
)
```

## Caveats

- Moment existence is a property of the generating distribution, not proof that
  a finite sample estimate will be accurate.
- When $\alpha$ is close to a boundary, convergence can be so slow that the
  formal moment is a poor operational summary.
- For two-sided heavy-tailed variables, check absolute moments or one-sided
  tails explicitly. Symmetry can make a location parameter look finite while
  absolute exposure is infinite.
- Empirical Pareto fits require threshold checks. A moment calculation using a
  fitted $\widehat\alpha$ inherits the uncertainty and bias of the tail fit.

## References

- Taleb, *Statistical Consequences of Fat Tails*, Chapter 5
  [@taleb2020scoft].
- Bingham, Goldie, and Teugels, *Regular Variation*
  [@bingham1987regular].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].

## Backlinks

- Depends on: [Pareto Distribution](../distributions/pareto.md) and
  [Karamata's Theorem](karamata.md).
- Used by: [LLN Failure Under Infinite Mean](lln-failure.md),
  [Pre-Asymptotic LLN Behavior](../examples/lln-preasymptotic.md), and the
  [Chapter 5 reading guide](../../reading-guides/taleb-scoft/ch5.md).
