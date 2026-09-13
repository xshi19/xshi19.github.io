---
title: Frechet Distribution and Frechet-Type Limits
options:
  concept:
    id: frechet
    type: distribution
    depends_on:
      - generalized-extreme-value
      - pareto
      - regular-variation
    tags:
      - extreme-value-theory
      - block-maxima
      - fat-tails
---

## Statement

The Frechet distribution is the positive-shape extreme-value law for normalized
maxima with heavy right tails.  In the standard one-parameter form with shape
$\alpha>0$, its CDF is

$$
\Phi_\alpha(y)
=
\begin{cases}
0, & y\le0,\\
\exp(-y^{-\alpha}), & y>0.
\end{cases}
$$

For $y>0$, the density is

$$
\phi_\alpha(y)
=
\alpha y^{-\alpha-1}\exp(-y^{-\alpha}).
$$

It is the $\xi>0$ member of the
[generalized extreme-value distribution](./incerto-generalized-extreme-value.md).  With
$\xi=1/\alpha$, location $\mu=1$, and scale $\sigma=\xi$,

$$
H_{\xi,1,\xi}(y)
=
\exp\left[-y^{-1/\xi}\right]
=
\Phi_\alpha(y),
\qquad y>0.
$$

The phrase Frechet-type refers to distributions whose normalized maxima
converge to this law.  In the usual right-tail setting, this is the
[regularly varying](./incerto-regular-variation.md) maximum-domain-of-attraction
case: a [Pareto-type](./incerto-pareto.md) survival tail with exponent $\alpha$ has
extreme-value index $\xi=1/\alpha>0$. Here $F$ is the parent CDF,
$\bar F=1-F$ its survival function, $X_i$ its iid draws, and $x_m>0$
the lower cutoff in the exact Pareto example. $\Phi_\alpha$ and
$\phi_\alpha$ denote the Frechet CDF and density.

## Maxima interpretation

Frechet-type behavior is a statement about extremes, not necessarily about the
full parent distribution.  If $X_1,\dots,X_n$ are iid and the right tail is
regularly varying with exponent $\alpha>0$, then an appropriate positive
normalization $a_n$ puts the maximum

$$
M_n=\max_{1\le i\le n} X_i
$$

on the Frechet scale.  The full regular-variation domain-of-attraction theorem
is cited in the references; the exact Pareto calculation below shows the core
mechanism.

For [Pareto Type I](./incerto-pareto.md) with lower cutoff $x_m$ and exponent $\alpha$,
take $a_n=x_m n^{1/\alpha}$.  Then for fixed $y>0$ and all large enough $n$,

$$
\mathbb P(M_n/a_n\le y)
=
\left(1-\frac{1}{ny^\alpha}\right)^n
\to
\exp(-y^{-\alpha})
=
\Phi_\alpha(y).
$$

The same shape parameter is therefore seen in two coordinates:

$$
\alpha \quad \text{for the Pareto-type survival exponent},
\qquad
\xi=\frac1\alpha \quad \text{for the extreme-value index}.
$$

## Tail behavior

Although the Frechet law is a limit law for maxima, it is itself heavy-tailed.
Its survival function satisfies

$$
\bar\Phi_\alpha(y)
=
1-\exp(-y^{-\alpha})
\sim
y^{-\alpha},
\qquad y\to\infty.
$$

Thus the Frechet survival tail is regularly varying with index $-\alpha$.
This is why the $\xi>0$ GEV regime is also the Pareto-type regime: the
block-maximum limit and the parent survival tail share the same reciprocal
coordinate $\xi=1/\alpha$.

## Finite-block calculation

(frechet-pareto-maxima-check)=
For any $\alpha>0$, the exact Pareto probability at $y=1$ is $(1-1/n)^n$.
The limit is $e^{-1}$, independently of $\alpha$ because the normalization
already includes $n^{1/\alpha}$.

| Block size $n$ | Exact $\mathbb P(M_n/a_n\le1)$ | Frechet limit |
| --- | --- | --- |
| $10$ | $0.348678$ | $0.367879$ |
| $100$ | $0.366032$ | $0.367879$ |
| $600$ | $0.367573$ | $0.367879$ |

For fixed $y>0$, set $t=y^{-\alpha}$. Expanding the logarithm gives

$$
n\log(1-t/n)=-t-\frac{t^2}{2n}+O(n^{-2}).
$$

This explains the finite-block correction for an exact Pareto parent at a
fixed $y$. It is not a uniform error bound over all thresholds or a rate for
arbitrary regularly varying parents. No simulated or fitted CDF is used here.

## Caveats

- Frechet-type is a domain-of-attraction label.  It does not say the original
  observations themselves follow a Frechet distribution.
- The $\xi>0$ conclusion is asymptotic.  Finite samples can look Frechet-like
  over one range and deviate elsewhere because of body contamination,
  dependence, truncation, censoring, or second-order tail behavior.
- GEV block-maxima fitting uses a location-scale family.  The unit Frechet law
  above is a convenient standard representative of the positive-shape class.

## References

- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  [2001](https://doi.org/10.1007/978-1-4471-3675-0).

## Backlinks

- Depends on: [Generalized Extreme-Value Distribution](./incerto-generalized-extreme-value.md),
  [Pareto Distribution](./incerto-pareto.md), and
  [Regular Variation](./incerto-regular-variation.md).
- Used by: [Extreme Value Index Estimation](./incerto-extreme-value-index.md)
  and the positive-shape regime of
  [Generalized Extreme-Value Distribution](./incerto-generalized-extreme-value.md).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/distributions/frechet.md`, revision `9717c9c`
(2026-09-13 Batch 2 import). Copyright (c) 2023 xshi19. Licensed under MIT.
Links, notation, and qualifications were adapted for this site; executable
figures and simulations were replaced with static calculations. No upstream
execution or formal-proof verification is claimed for this adaptation.

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
