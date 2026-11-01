---
title: Generalized Extreme-Value Distribution
options:
  concept:
    id: generalized-extreme-value
    type: distribution
    depends_on:
      - extreme-value-index
    tags:
      - extreme-value-theory
      - block-maxima
      - tail-risk
---

## Statement

The generalized extreme-value distribution is the standard limit family for
normalized block maxima.  With location $\mu$, scale $\sigma>0$, shape
$\xi$, and standardized variable $z=(x-\mu)/\sigma$, its CDF is

$$
\begin{gathered}
H_{\xi,\mu,\sigma}(x)\\
=\exp\left[-\left(1+\xi z\right)^{-1/\xi}\right],\\
\xi\ne0,
\end{gathered}
$$

where $1+\xi z>0$. For $\xi>0$, extend the CDF by zero at and
below $\mu-\sigma/\xi$; for $\xi<0$, extend it by one at and above
$\mu-\sigma/\xi$. The boundary case is the Gumbel law,

$$
H_{0,\mu,\sigma}(x)
=
\exp\left[-e^{-z}\right].
$$

The shape parameter $\xi$ is the same extreme-value index used in threshold
excesses: $\xi>0$ is [Frechet-type](./incerto-frechet.md) heavy-tail behavior,
$\xi=0$ is Gumbel-type behavior, and $\xi<0$ is
Weibull-type finite endpoint.
Here $X_i$ are observations and $M_n=\max_{1\le i\le n}X_i$. The Pareto
example below uses lower cutoff $x_m>0$ and exponent $\alpha>0$.
Gumbel attraction is broader than exponential parent tails: normal and
lognormal parents are examples, despite their different tail decay.

## Maxima intuition

The distribution is to block maxima what the [Generalized Pareto Distribution](./incerto-generalized-pareto.md)
is to threshold exceedances. For iid observations, if there are constants
$a_n>0$ and $b_n$ such that $(M_n-b_n)/a_n$ converges to a nondegenerate
law, that law is a GEV distribution up to location and scale. Existence
of such a limit is an assumption; it does not hold for every parent law.

For fat-tail work, $\xi>0$ is the main case.  A [Pareto tail](./incerto-pareto.md) with exponent
$\alpha$ has $\xi=1/\alpha$, so the fitted block-maximum shape should point to
the same tail coordinate as a Hill or GPD threshold analysis when the modeling
assumptions are reasonable.

## Shape regimes

(generalized-extreme-value-cdf-plot)=
With $\mu=0$ and $\sigma=1$, the support makes the three shapes explicit.

| Shape $\xi$ | Interior CDF | Support |
| --- | --- | --- |
| $-0.3$ | $\exp[-(1-0.3x)^{10/3}]$ | $x<10/3$; CDF is one above |
| $0$ | $\exp[-e^{-x}]$ | All real $x$ |
| $0.3$ | $\exp[-(1+0.3x)^{-10/3}]$ | $x>-10/3$; CDF is zero below |

Every curve passes through $H(0)=e^{-1}$. Positive shape gives a power-law
right tail; negative shape gives a finite right endpoint.

```{figure} ../assets/figures/incerto/gev-card.svg
:name: gev-distribution-card
:alt: GEV densities and CDFs show a left endpoint for positive shape and a right endpoint for negative shape. A third log-log panel shows survival only at positive thresholds.

Exact GEV laws with $(\mu,\sigma)=(0,1)$ and $\xi=-0.3,0,0.3$.
The dotted boundaries are $-10/3$ for positive shape and $10/3$ for
negative shape. The Gumbel law has no finite endpoint. The log-log panel
shows only positive thresholds, with survival below $10^{-8}$ outside
the viewport and exact zero omitted.
```

[`generalized_extreme_value`](incerto-api-reference.md#api-generalized-extreme-value)
uses the shape sign in this page's CDF. It passes `c=-xi` to
[SciPy's `genextreme`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html).

```python
from xmath.incerto.distributions import generalized_extreme_value

for xi in (-0.3, 0, 0.3):
    law = generalized_extreme_value(xi)
    left, right = law.support()
    print(f"xi={xi:.4g}: support=[{left:.4g}, {right:.4g}], cdf(0)={law.cdf(0):.4g}")
```

**Output**

```text
xi=-0.3: support=[-inf, 3.333], cdf(0)=0.3679
xi=0: support=[-inf, inf], cdf(0)=0.3679
xi=0.3: support=[-3.333, inf], cdf(0)=0.3679
```

Regenerate the card with `uv run python demos/incerto/distribution_cards.py`.

## Pareto block maxima example

If $X_1,\dots,X_n$ are iid [Pareto Type I](./incerto-pareto.md) with lower cutoff $x_m$ and exponent
$\alpha$, then for $x\ge x_m$,

$$
\begin{aligned}
&\mathbb P(M_n\le x)\\
&\quad=\left(1-\left(\frac{x_m}{x}\right)^\alpha\right)^n.
\end{aligned}
$$

With $a_n=x_m n^{1/\alpha}$, for fixed $y>0$ and sufficiently large $n$,

$$
\begin{gathered}
\begin{aligned}
&\mathbb P(M_n/a_n\le y)\\
&\quad=\left(1-\frac{1}{ny^\alpha}\right)^n\\
&\quad\to\exp(-y^{-\alpha}),
\end{aligned}\\
y>0.
\end{gathered}
$$

This is the [Frechet member](./incerto-frechet.md) of the GEV family with
$\xi=1/\alpha$ after a standard location-scale reparameterization.

## Block-size calibration

(generalized-extreme-value-python-check)=
For an exact Pareto parent with $\alpha=1.8=9/5$, the limiting shape is
$\xi=5/9$. At block size $n=250$, the normalizing scale is
$a_n=x_m250^{5/9}$. At the normalized threshold $y=1$,

$$
\begin{gathered}
\begin{aligned}
&\mathbb P(M_{250}/a_{250}\le1)\\
&\quad=(249/250)^{250}\approx0.367142,
\end{aligned}\\
\Phi_{1.8}(1)=e^{-1}\approx0.367879.
\end{gathered}
$$

This compares an exact finite-block CDF with its limit; it is not a GEV fit
or an assessment of sampling uncertainty. The standard Frechet limit has GEV
parameters $(\xi,\mu,\sigma)=(5/9,1,5/9)$ on the normalized scale.

(gev-gaussian-maxima)=
## Gaussian maxima and normalization

SCoFT v4 §9.1.2, pp. 174–176, compares exact normal maxima with their
Gumbel limit [@taleb2025scoftv4]. We use the maxima CDF
$\Lambda(z)=\exp(-e^{-z})$ throughout. For iid standard normals,
independence and differentiation give the exact identities

$$
\begin{aligned}
P(M_n\le x)&=\Phi(x)^n,\\
f_{M_n}(x)&=n\phi(x)\Phi(x)^{n-1}.
\end{aligned}
$$

Here $\Phi$ and $\phi$ are the standard normal CDF and density. Choose

$$
\begin{gathered}
b_n=\Phi^{-1}(1-1/n),\\
a_n=\frac{1-\Phi(b_n)}{\phi(b_n)}.
\end{gathered}
$$

These quantile/Mills-ratio constants are given by Gasull, Jolis and Utzet,
§2, pp. 3–4, Eqs. (7)–(9) [@gasull2015normalmaxima]. We prove convergence
for this choice. Integration by parts bounds the normal survival, for $t>0$,
by $t\phi(t)/(1+t^2)$ below and $\phi(t)/t$ above: the upper bound follows
from $x/t\ge1$, and for the lower bound use
$\bar\Phi(t)=\phi(t)/t-\int_t^\infty\phi(x)/x^2\,dx$
and bound the last integral by $\bar\Phi(t)/t^2$.
Thus $a_nb_n\to1$ and $a_n\to0$. For fixed $z$, the same bounds give

$$
\begin{gathered}
\frac{\bar\Phi(b_n+a_nz)}{\bar\Phi(b_n)}\\
\sim\frac{b_n}{b_n+a_nz}
\exp(-b_na_nz)\\
{}\qquad\times\exp\left(-\frac{a_n^2z^2}{2}\right)\\
\longrightarrow e^{-z}.
\end{gathered}
$$

Since $n\bar\Phi(b_n)=1$, taking logarithms of the exact maximum CDF
proves $P((M_n-b_n)/a_n\le z)\to\Lambda(z)$. This proves a fixed-$z$
limit; it does not bound a relative density error far into the tail.

A commonly used explicit alternative is

$$
\begin{gathered}
s_n=\sqrt{2\log n},\qquad \widetilde a_n=1/s_n,\\
\widetilde b_n=s_n-\frac{\log\log n+\log(4\pi)}{2s_n}.
\end{gathered}
$$

The normal-tail bounds imply
$b_n=s_n+O(\log\log n/s_n)$. Substitution into
$b_n^2=2\log n-2\log b_n-\log(2\pi)+O(b_n^{-2})$
then gives $b_n-\widetilde b_n=O((\log\log n)^2/s_n^3)$.
Consequently $\widetilde a_n/a_n\to1$ and
$(\widetilde b_n-b_n)/a_n\to0$, proving the same limit.
Their finite-$n$ errors can differ. A CDF limit supplies neither a universal
finite-block accuracy threshold nor a relative tail-error guarantee.

Use [`gaussian_maximum_norming`](incerto-api-reference.md#api-gaussian-maximum-norming)
and [`gaussian_maximum_cdf`](incerto-api-reference.md#api-gaussian-maximum-cdf).

```python
from xmath.incerto.extremes import gaussian_maximum_norming, gaussian_maximum_cdf

for method in ("mills", "asymptotic"):
    a, b = gaussian_maximum_norming(100, method=method)
    print(f"{method}: a={a:.4g}, b={b:.4g}, cdf at z=1={gaussian_maximum_cdf(b+a, 100):.4g}")
for x in (-float("inf"), float("inf")):
    print(f"x={x:g}: cdf={gaussian_maximum_cdf(x, 100):.4g}")
```

**Output**

```text
mills: a=0.3752, b=2.326, cdf at z=1=0.7077
asymptotic: a=0.3295, b=2.366, cdf at z=1=0.7034
x=-inf: cdf=0
x=inf: cdf=1
```

```{figure} ../assets/figures/incerto/gaussian-maxima.svg
:alt: Exact normalized normal-maximum CDFs approach Gumbel with increasing block size. Separate CDF and density error panels at size 100 compare Mills-ratio and explicit asymptotic normalizations.

Top: exact normal maxima with the quantile/Mills-ratio constants, at
$n=10,100,10000$, and the Gumbel limit. Lower panels: signed
CDF and density differences from Gumbel at $n=100$, for both choices
of constants. The normalized density includes the Jacobian $a_n$.
These are population calculations on $z\in[-3,5]$, not simulated fits,
global error suprema, or relative errors at arbitrarily rare levels.
```

Regenerate with `uv run python demos/incerto/extreme_values.py`.
For a normal parent with mean $\mu$ and standard deviation $\sigma>0$,
replace $(a_n,b_n)$ by $(\sigma a_n,\mu+\sigma b_n)$.
The [hidden-tail normal integral](incerto-hidden-tail.md#hidden-tail-gaussian)
uses the exact maximum law without either limiting approximation.

## Caveats

- Block maxima discard within-block information.  Threshold methods may use
  extremes more efficiently, but they require a threshold choice.
- Blocks should be chosen with dependence and seasonality in mind.  Overlapping
  or strongly dependent blocks can make fitted uncertainty too optimistic.
- A GEV fit describes maxima, not the entire parent distribution.
- Software may use a different shape sign convention. Translate fitted
  parameters into the displayed CDF before comparing maxima and excesses.

## References

- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  @embrechts1997modelling.
- Resnick, *Heavy-Tail Phenomena* @resnick2007heavy.
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  @coles2001introduction.
- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  (2nd ed., Wiley, 1971).

## Backlinks

- Depends on: [Extreme-Value Index Estimation](./incerto-extreme-value-index.md).
- Used by: [Frechet Distribution and Frechet-Type Limits](./incerto-frechet.md),
  [Pickands-Balkema-de Haan Theorem](./incerto-pickands-balkema-de-haan.md)
  as the block-maxima counterpart to threshold excess limits, and by future
  return-level pages.
