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

```{figure} ../assets/figures/incerto/frechet-card.svg
:name: frechet-distribution-card
:alt: Fréchet density, CDF and log-log survival for three exponents, with the support boundary at zero and curved survival approaching a power tail.

Unit-scale Fréchet laws with $\alpha=0.8,1.5,3$, using the same exponent
colors as the Pareto card. The boundary is zero. Survival approaches a
power tail at large positive thresholds; it is not an exact power over
the whole support. The last panel's lower visible probability is $10^{-8}$.
```

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

```{figure} ../assets/figures/incerto/frechet-maxima.svg
:name: frechet-maxima-comparison
:alt: Exact CDFs of normalized Pareto maxima for block sizes 1, 4, 16 and 64 approach the Fréchet CDF. A second panel displays the finite-block CDF difference.

Exact population CDFs for iid Pareto parents with $\alpha=1.5$ and
normalization $a_n=x_m n^{1/\alpha}$; no maxima are simulated or fitted.
Dotted lines mark the finite-block lower boundaries $n^{-1/\alpha}$.
The second panel plots the limit CDF minus each finite-block CDF, making
small corrections visible without displacing overlapping curves.
```

Use [`frechet`](incerto-api-reference.md#api-frechet) and the existing
[`pareto_normalized_maximum_cdf`](incerto-api-reference.md#api-pareto-normalized-maximum-cdf).

```python
from xmath.incerto.distributions import frechet
from xmath.incerto.glossary import pareto_normalized_maximum_cdf

for n in (1, 4, 16, 64):
    finite = pareto_normalized_maximum_cdf([1], alpha=1.5, n=n)[0]
    print(f"n={n:d}: cdf(1)={finite:.4g}, limit={frechet(1.5).cdf(1):.4g}")
```

**Output**

```text
n=1: cdf(1)=0, limit=0.3679
n=4: cdf(1)=0.3164, limit=0.3679
n=16: cdf(1)=0.3561, limit=0.3679
n=64: cdf(1)=0.365, limit=0.3679
```

Regenerate both figures with `uv run python demos/incerto/distribution_cards.py`.

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
  @embrechts1997modelling.
- Resnick, *Heavy-Tail Phenomena* @resnick2007heavy.
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  @coles2001introduction.

## Backlinks

- Depends on: [Generalized Extreme-Value Distribution](./incerto-generalized-extreme-value.md),
  [Pareto Distribution](./incerto-pareto.md), and
  [Regular Variation](./incerto-regular-variation.md).
- Used by: [Extreme Value Index Estimation](./incerto-extreme-value-index.md)
  and the positive-shape regime of
  [Generalized Extreme-Value Distribution](./incerto-generalized-extreme-value.md).
