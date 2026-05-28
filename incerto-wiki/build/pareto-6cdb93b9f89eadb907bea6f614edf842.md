---
kernelspec:
  name: python3
  display_name: Python 3
---

# Pareto Distribution

## Statement

Let $X$ be a positive random variable with lower cutoff $x_m>0$ and tail
exponent $\alpha>0$.  The Pareto Type I distribution is defined by the survival
function

$$
\bar F(x) = \mathbb P(X>x) =
\begin{cases}
1, & 0 \le x < x_m,\\
\left(\frac{x_m}{x}\right)^\alpha, & x \ge x_m.
\end{cases}
$$

For $x \ge x_m$, the CDF and density are

$$
F(x)=1-\left(\frac{x_m}{x}\right)^\alpha,\qquad
f(x)=\alpha x_m^\alpha x^{-(\alpha+1)}.
$$

For probability level $0<q<1$, the quantile function is

$$
Q(q)=x_m(1-q)^{-1/\alpha}.
$$

The raw moment of order $p>0$ is finite exactly when $p<\alpha$:

$$
\mathbb E[X^p]=\frac{\alpha x_m^p}{\alpha-p},\qquad p<\alpha.
$$

For $p\ge\alpha$, the moment is infinite.  At $p=\alpha$ the divergence is
logarithmic; for $p>\alpha$ it is a power divergence.  In particular, the mean
exists only for $\alpha>1$, and the variance exists only for $\alpha>2$.

## Intuition

The Pareto tail has no characteristic scale.  Multiplying the threshold by a
fixed factor changes the exceedance probability by a fixed power:

$$
\frac{\bar F(tx)}{\bar F(x)}=t^{-\alpha},\qquad t>0.
$$

This is why the exponent $\alpha$ is the central knob.  Smaller $\alpha$ means
that threshold doublings are punished less severely, so rare observations remain
large enough to dominate sums, moments, and empirical estimates.

## Proof

For $x\ge x_m$, differentiating the CDF gives the density:

$$
\frac{d}{dx}\left(1-x_m^\alpha x^{-\alpha}\right)
=\alpha x_m^\alpha x^{-(\alpha+1)}.
$$

For $p>0$,

$$
\mathbb E[X^p]
=\int_{x_m}^{\infty}x^p\alpha x_m^\alpha x^{-(\alpha+1)}\,dx
=\alpha x_m^\alpha\int_{x_m}^{\infty}x^{p-\alpha-1}\,dx.
$$

The final integral converges exactly when $p-\alpha-1<-1$, equivalently
$p<\alpha$.  Evaluating it gives

$$
\alpha x_m^\alpha\frac{x_m^{p-\alpha}}{\alpha-p}
=\frac{\alpha x_m^p}{\alpha-p}.
$$

At the boundary $p=\alpha$, the integral becomes

$$
\alpha x_m^\alpha\int_{x_m}^{\infty}\frac{dx}{x},
$$

so it diverges logarithmically.  For $p>\alpha$, the exponent
$p-\alpha-1>-1$, so the cutoff integral grows as a positive power of the
cutoff.  The exact Pareto calculation is the simplest instance of the general
moment test supplied by [Karamata's theorem](../theorems/karamata.md).

The ratio identity follows immediately from the survival function:

$$
\frac{(x_m/(tx))^\alpha}{(x_m/x)^\alpha}=t^{-\alpha}.
$$

## Python

The package implementation uses the same tail exponent $\alpha$ and the
SciPy-style `scale` keyword for the lower cutoff $x_m$.

```{code-cell} python
:label: pareto-python-check

import numpy as np
import matplotlib.pyplot as plt

from incerto.distributions import pareto_type1

np.set_printoptions(precision=4, suppress=True)

alpha = 1.16
x_m = 1.0

x = np.array([1, 2, 4, 8, 16], dtype=float)
survival = pareto_type1.sf(x, alpha, scale=x_m)
ratio = pareto_type1.sf(2 * x, alpha, scale=x_m) / survival
mean, variance = pareto_type1.stats(alpha, scale=x_m, moments="mv")

x_grid = np.geomspace(x_m, 100, 200)
fig, ax = plt.subplots(figsize=(7, 4))
for exponent in (0.8, 1.5, 3.0):
    ax.loglog(
        x_grid,
        pareto_type1.sf(x_grid, exponent, scale=x_m),
        lw=2,
        label=f"alpha={exponent}",
    )

ax.set_xlabel("threshold x")
ax.set_ylabel("survival P(X > x)")
ax.set_title("Pareto survival tails")
ax.legend()
plt.show()

print(f"Survival values at x={x.tolist()}: {survival}")
print(f"Survival ratio P(X>2x)/P(X>x): {ratio}")
print(f"Mean and variance: {(float(mean), float(variance))}")
```

The package also includes `incerto.distributions.double_pareto`, which uses the
same tail exponent on both sides of the origin.

## Caveats

- Taleb often writes the tail exponent as $a$; the wiki uses $\alpha$.
- Empirical data rarely follows an exact Pareto law from its minimum value.  A
  tail model needs a threshold choice, diagnostic plots, and sensitivity checks.
- A finite theoretical mean can still be practically hard to estimate when
  $\alpha$ is close to 1.  The issue is pre-asymptotic behavior, not merely the
  formal existence of $\mathbb E[X]$.
- The exact failure mode depends on the moment order.  At $p=\alpha$, the
  cutoff moment grows like $\log x$; for $p>\alpha$, it grows like
  $x^{p-\alpha}$.
- Do not estimate high moments of heavy-tailed samples without checking whether
  those moments are implied by the fitted tail exponent.

## References

- Taleb, *Statistical Consequences of Fat Tails*, Chapter 3
  [@taleb2020scoft].
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].

## Backlinks

- Depends on: [Regular Variation](../theorems/regular-variation.md) and
  [Karamata's theorem](../theorems/karamata.md).
- Used by: [Double Pareto Distribution](double-pareto.md),
  [Pareto Moment Existence](../theorems/pareto-moment-existence.md),
  [Pre-Asymptotic LLN Behavior](../examples/lln-preasymptotic.md),
  [Max-to-Sum Ratio](../methods/max-to-sum-ratio.md),
  [Hill Estimator](../methods/hill-estimator.md), and the
  [Chapter 3 reading guide](../../reading-guides/taleb-scoft/ch3.md).
