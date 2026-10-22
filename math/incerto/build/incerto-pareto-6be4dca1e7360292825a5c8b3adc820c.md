---
title: Pareto Distribution
options:
  concept:
    id: pareto
    type: distribution
    prerequisites: []
    special_case_of: [regular-variation]
    applications: [pareto-moment-existence, hill-estimator]
    tags:
      - fat-tails
      - power-law
---

## Overview

How quickly does the chance of a large observation fall as we raise the
threshold? The Pareto law models a positive quantity above a lower cutoff.
Its tail exponent $\alpha$ controls that rate: smaller exponents give more
probability to very large observations. Doubling any threshold in its support
multiplies the survival probability by $2^{-\alpha}$.

Basic probability and integration are enough for this page. A survival
probability means $\mathbb P(X>x)$. The formulas below define the CDF $F$,
density $f$, and quantile $Q$; the survival table illustrates the exponent.

(pareto-definition)=
## Definition and formula summary

Let $X$ be a positive random variable with lower cutoff $x_m>0$ and tail
exponent $\alpha>0$.  The Pareto Type I distribution is defined by

$$
\bar F(x)=\mathbb P(X>x)=
\begin{cases}
1, & x < x_m,\\
\left(\frac{x_m}{x}\right)^\alpha, & x\ge x_m.
\end{cases}
$$

For $x\ge x_m$, the CDF, density, and quantile function are

$$
F(x)=1-\left(\frac{x_m}{x}\right)^\alpha,
\qquad
f(x)=\alpha x_m^\alpha x^{-(\alpha+1)},
\qquad
Q(q)=x_m(1-q)^{-1/\alpha},\quad 0<q<1.
$$

The exact Pareto tail is [regularly varying](./incerto-regular-variation.md)
with index $-\alpha$.  Its raw moment of order $p>0$ exists exactly when
$p<\alpha$:

$$
\mathbb E[X^p]=\frac{\alpha x_m^p}{\alpha-p},\qquad p<\alpha.
$$

In particular,

$$
\mathbb E[X]=\frac{\alpha x_m}{\alpha-1},\qquad \alpha>1,
$$

and

$$
\operatorname{Var}(X)
=\frac{\alpha x_m^2}{(\alpha-1)^2(\alpha-2)},\qquad \alpha>2.
$$

Thus the mean exists only for $\alpha>1$, and a finite variance exists only
for $\alpha>2$. Here $p$ is the moment order and $q$ is the quantile level.

## Shape and tail intuition

(pareto-survival-scale)=
Above its lower cutoff, the Pareto law has no characteristic upper scale.  Once
both thresholds lie in the Pareto tail, multiplying the threshold by a fixed factor changes the exceedance
probability by a fixed power:

$$
\frac{\bar F(tx)}{\bar F(x)}=t^{-\alpha},\qquad t>0,\; x\ge x_m,\; tx\ge x_m.
$$

Smaller $\alpha$ gives a larger exceedance probability at every $x>x_m$.
The consequences for sums depend on whether the sample size or the exceedance
threshold grows; moment existence also depends on the moment order.

The density also shows how the distribution piles mass near the lower cutoff
while leaving a long right tail. Its values and scaling make this explicit.

(pareto-density-plot)=
At the cutoff, $f(x_m)=\alpha/x_m$. For $x\ge x_m$,

$$
\frac{f(2x)}{f(x)}=2^{-(\alpha+1)}.
$$

This density exponent differs by one from the survival exponent.

Larger $\alpha$ concentrates more mass near $x_m$ and makes
the density decay faster.  Smaller $\alpha$ keeps more visible mass far from the
cutoff.

```{figure} ../assets/figures/incerto/pareto-card.svg
:name: pareto-distribution-card
:alt: Three stacked panels show Pareto densities and CDFs on linear axes and straight survival lines on log-log axes, with the lower cutoff at one marked.

Exact Pareto laws with $x_m=1$ and $\alpha=0.8,1.5,3$.
Dotted vertical lines mark the lower support boundary. Larger exponents
concentrate more density near $1$. For $x\ge1$, the log-log survival slope
is $-\alpha$, while the density exponent is $-(\alpha+1)$; below the
cutoff, survival is one. The last panel uses positive thresholds and a
lower visible probability of $10^{-8}$.
```

The card uses [`pareto_type1`](incerto-api-reference.md#api-pareto-type1).
Regenerate it with `uv run python demos/incerto/distribution_cards.py`.

## Power-law survival

Survival probabilities are usually more revealing than densities for fat-tail
work.  For an exact Pareto law,

$$
\bar F(x)=x_m^\alpha x^{-\alpha},\qquad x\ge x_m,
$$

so a log-log survival plot is a straight line with slope $-\alpha$:

$$
\log \bar F(x)=\alpha\log x_m-\alpha\log x.
$$

(pareto-survival-plot)=
For $x_m=1$, the survival values are powers of the threshold:

| Tail exponent $\alpha$ | $\bar F(1)$ | $\bar F(2)$ | $\bar F(4)$ |
| --- | --- | --- | --- |
| $0.8$ | $1$ | $2^{-0.8}$ | $2^{-1.6}$ |
| $1.5$ | $1$ | $2^{-1.5}$ | $2^{-3}$ |
| $3$ | $1$ | $2^{-3}$ | $2^{-6}$ |

These survival functions trace straight lines on log-log axes because each is
an exact power. The heaviest tail has the shallowest line.

The same scaling appears numerically. For a fixed multiplier $t>0$, the
ratio $\bar F(tx)/\bar F(x)$ depends only on $t$, provided both
$x\ge x_m$ and $tx\ge x_m$. The identity follows by canceling $(x_m/x)^\alpha$.

(pareto-survival-ratio-check)=
For example, with $\alpha=1.16$, doubling any threshold in the support
multiplies survival by $2^{-1.16}\approx0.4475$. This is a direct evaluation
of the scaling identity, not an empirical estimate.

## Moment thresholds

For a positive moment order $p$, the exact Pareto formula is

$$
\mathbb E[X^p]=\frac{\alpha x_m^p}{\alpha-p}\quad (p<\alpha),
$$

and the moment is infinite for $p\ge\alpha$. The
[Pareto moment theorem](./incerto-pareto-moment-existence.md) gives the direct
integration proof and the truncated-moment formulas and table. Truncation
makes each truncated population moment finite: raising the upper cutoff reveals
whether it approaches a limit, grows logarithmically at $p=\alpha$, or grows
as a power for $p>\alpha$.

For a general [regularly varying tail](./incerto-regular-variation.md),
$\bar F(x)=x^{-\alpha}L(x)$, the boundary $p=\alpha$ depends on $L$.
[Karamata's theorem](./incerto-karamata.md) explains this distinction; the
exact Pareto boundary always diverges.

The moment-existence thresholds give a concrete example:

(pareto-stat-check)=
With $x_m=1$ and $\alpha=1.16$, the mean is
$1.16/(1.16-1)=7.25$ and the second raw moment is infinite; the variance
is therefore infinite.

For simulation, inverse transform sampling gives

$$
X=x_m U^{-1/\alpha},\qquad U\sim\operatorname{Uniform}(0,1).
$$

When $x_m$ is known and the observations are independent and identically
distributed exact Pareto draws, the maximum-likelihood estimator is

$$
\widehat\alpha=\frac{n}{\sum_{i=1}^n\log(X_i/x_m)}.
$$

This formula requires a positive denominator, which holds almost surely under
the continuous model. If every observation equals $x_m$, the likelihood has
no finite maximizer; it increases as $\alpha\to\infty$.

This estimator is not a license to fit an exact Pareto from the sample minimum:
empirical tail thresholds are rarely known and must be diagnosed separately.

The [symmetric shifted double Pareto model](./incerto-double-pareto.md)
extends the tail construction to signed observations.

(pareto-threshold-quantities)=
## Exceedance probability and exceedance size

For $u\ge x_m$ and $y\ge0$, cancellation of Pareto survival probabilities
proves the conditional excess law:

$$
P(X-u>y\mid X>u)=\frac{\bar F(u+y)}{\bar F(u)}
=\left(1+\frac yu\right)^{-\alpha}.
$$

This is an exact [generalized Pareto](incerto-generalized-pareto.md) survival
with shape $\xi=1/\alpha$ and scale $u/\alpha$. It holds at every threshold
$u\ge x_m$ for this model; no limiting approximation is required.
Integrating the nonnegative excess survival gives

$$
e(u)=\mathbb E[X-u\mid X>u]=\frac{u}{\alpha-1},\qquad
\mathbb E[X\mid X>u]=\frac{\alpha u}{\alpha-1},\quad \alpha>1.
$$

Both expectations are infinite for $0<\alpha\le1$, since the integral of
$(1+y/u)^{-\alpha}$ diverges. The first expectation is the
[mean excess](incerto-mean-excess-function.md); the second includes the
threshold itself. In particular, their ratio to $u$ is different.

We also prove the distinction between tail probability and tail contribution
to the mean. For any nonnegative $X$ and $u\ge0$, the pointwise identity
$X\mathbf1_{\{X>u\}}=u\mathbf1_{\{X>u\}}+(X-u)_+$ and Tonelli's theorem
give

$$
\mathbb E[X\mathbf1_{\{X>u\}}]
=u\bar F(u)+\int_u^\infty\bar F(v)\,dv.
$$

The equality holds in the extended nonnegative reals, including atoms at
$u$. For Pareto with $\alpha>1$ it becomes
$\bar F(u)\alpha u/(\alpha-1)$. For $\alpha=3/2$, $x_m=1$, and $u=4$,
the probability is $1/8$, the mean excess is $8$, the conditional mean is
$12$, and the tail contribution is $3/2$. Replacing that contribution by
$u\bar F(u)=1/2$ would discard the entire excess integral.

Use [`pareto_tail_quantities`](incerto-api-reference.md#api-pareto-tail-quantities).

:::{dropdown} Code

```python
from xmath.incerto.glossary import pareto_tail_quantities

for alpha in (1, 1.5, 3):
    print(f"alpha={alpha:.4g}")
    for name, value in pareto_tail_quantities(alpha, x_m=1, threshold=4).items():
        print(f"  {name}: {value:.4g}")
```

**Output**

```text
alpha=1
  log_survival: -1.386
  mean_excess: inf
  conditional_mean: inf
  tail_first_moment: inf
alpha=1.5
  log_survival: -2.079
  mean_excess: 8
  conditional_mean: 12
  tail_first_moment: 1.5
alpha=3
  log_survival: -4.159
  mean_excess: 2
  conditional_mean: 6
  tail_first_moment: 0.09375
```
:::



These definitions clarify SCoFT v4 §2.2.4, pp. 10–11, and §§2.2.23–24,
pp. 17–18 [@taleb2025scoftv4]. The exact Pareto calculation here does not
classify arbitrary tails from a finite mean-excess plot. Reproduce it with
`uv run python demos/incerto/glossary_worked_cases.py`; the script also reports
the infinite-mean cases without replacing infinity by a finite cutoff.

(pareto-known-cutoff-likelihood)=
## Why the known-cutoff likelihood formula needs conditions

We prove the estimator stated above for $n\ge1$ iid exact Pareto observations,
known $x_m>0$, and observed $x_i\ge x_m$. Let
$D=\sum_i\log(x_i/x_m)$. Terms independent of $\alpha$ can be omitted from
the log likelihood, leaving

$$
\ell(\alpha)=n\log\alpha-\alpha D+\text{constant},\qquad
\ell'(\alpha)=\frac n\alpha-D,\qquad
\ell''(\alpha)=-\frac n{\alpha^2}<0.
$$

If $D>0$, the unique maximizer is $n/D$. If $D=0$, every observation equals
$x_m$ and $\ell$ increases without bound as $\alpha\to\infty$. Observations
below $x_m$ are outside this model's support. A threshold estimated from the
same sample changes the problem; this proof supplies no guarantee for that fit.

## Sums and moment orders

Let $X_1,X_2,\ldots$ be iid Pareto Type I variables with the same
$\alpha>0$ and $x_m>0$, and put $S_n=\sum_{i=1}^nX_i$ and
$M_n=\max_{1\le i\le n}X_i$. For a growing sample, the
[max-to-sum ratio](./incerto-max-to-sum-ratio.md) has three regimes:

- If $\alpha>1$, then $M_n/S_n\to0$ almost surely. The linked
  finite-mean proof gives $S_n/n\to\mathbb E[X]>0$ and $M_n/n\to0$
  almost surely, so their quotient tends to zero.
- If $0<\alpha<1$, then $M_n/S_n\xrightarrow{d}R_\alpha$, where
  $R_\alpha$ has a nondegenerate distribution on $(0,1)$. In particular,
  the ratio does not tend to zero in probability, nor does it tend to one.
  This limit theorem is cited rather than proved here
  [@darling1952influence, Theorem 5.1].
  [Kevei and Mason (2014), p. 632](https://alea.impa.br/articles/v11/11-30.pdf#page=2)
  restate the iid result and give the corresponding sum/maximum limit law;
  our ratio is its reciprocal.
- If $\alpha=1$, the mean is infinite but $M_n/S_n\to0$ in probability.
  More precisely, $M_n/(x_m n)$ is bounded in probability and
  $S_n/(x_m n\log n)\to1$ in probability. The calculation below proves
  these exact-Pareto assertions; it does not assert almost-sure convergence
  of the ratio.

For every $\alpha>0$, a different limit holds when the integer $n\ge1$
is fixed and the threshold grows:

$$
\mathbb P(S_n>x)\sim\mathbb P(M_n>x)
\sim n\mathbb P(X>x),\qquad x\to\infty.
$$

This is the [subexponential](./incerto-subexponentiality.md) rare-event
statement: a fixed-size sum exceeds a sufficiently high threshold mainly when
one summand exceeds it. It holds even when $\alpha>1$ and the largest share
vanishes along growing samples. For moments, the separate exact criterion is
$\mathbb E[X^p]<\infty$ if and only if $p<\alpha$, for $p>0$, as proved in
[Pareto Moment Existence](./incerto-pareto-moment-existence.md).

### Exact-Pareto boundary calculation

We prove the two normalizations used above for $\alpha=1$.
For $z>0$ and sufficiently large $n$, independence gives

$$
\mathbb P\!\left(\frac{M_n}{x_m n}\le z\right)
=\left(1-\frac{1}{nz}\right)^n\longrightarrow e^{-1/z}.
$$

For the sum, write $V_i=X_i/x_m$ and, for $n\ge3$, set $b_n=n\log n$ and
$T_{i,n}=V_i\mathbf1_{\{V_i\le b_n\}}$. Integration of the density
$v^{-2}$ on $[1,b_n]$ gives
$\mathbb E[T_{i,n}]=\log b_n$ and
$\mathbb E[T_{i,n}^2]=b_n-1$. Hence

$$
\operatorname{Var}\!\left(\frac{\sum_{i=1}^nT_{i,n}}{n\log n}\right)
\le\frac{b_n}{n(\log n)^2}=\frac{1}{\log n}\longrightarrow0.
$$

The normalized expectation $\log b_n/\log n$ tends to $1$.
Chebyshev's inequality gives convergence in probability for the truncated sum.
The union bound $\mathbb P(\max_{i\le n}V_i>b_n)\le n/b_n=1/\log n$
transfers that convergence to $S_n/(x_m n\log n)$. Dividing the tight
normalized maximum by this normalized sum and by $\log n$ proves
$M_n/S_n\to0$ in probability.

### Fixed-size rare-event calculation

We extend the two-summand bound on the subexponentiality page to prove the
displayed fixed-$n$ result. The case $n=1$ is immediate. For fixed $n\ge2$,
choose $h=x_m(x/x_m)^\gamma$ with $1/2<\gamma<1$. If $S_n>x$ but
$M_n\le x$, either one observation lies in $(x-(n-1)h,x]$, or at least two
observations exceed $h$. Otherwise the sum is at most $x$. Thus

$$
\mathbb P(S_n>x,\ M_n\le x)
\le n\bigl[\bar F(x-(n-1)h)-\bar F(x)\bigr]
  +\binom n2\bar F(h)^2.
$$

After division by $\bar F(x)$, the first bracket tends to zero because
$h/x\to0$; the second tail ratio is
$(x/x_m)^{\alpha(1-2\gamma)}\to0$.
Finally, $\mathbb P(M_n>x)=1-(1-\bar F(x))^n\sim n\bar F(x)$.
Nonnegativity makes $\{M_n>x\}$ a subset of $\{S_n>x\}$, and the
negligible residual probability proves the sum-tail equivalence.

(pareto-sums-products)=
## Positive sums and independent products

The sum and product discussions in SCoFT v4 §§5.2.1–5.2.2, pp. 98–100
[@taleb2025scoftv4], require different assumptions. Here we prove a bounded
claim about sums and exact formulas for independent Pareto products.

Fix $n<\infty$, positive weights $w_i$, and nonnegative $X_i$ with
$P(X_i>x)=x^{-\alpha_i}L_i(x)$ eventually, where $\alpha_i>0$
and each $L_i$ is slowly varying.
Dependence is unrestricted. For $S=\sum_{i=1}^n w_iX_i$,

$$
\max_i P(X_i>x/w_i)\le P(S>x)
\le\sum_{i=1}^nP(X_i>x/(nw_i)).
$$

The lower bound uses nonnegativity; the upper bound holds because a sum
above $x$ has some term above $x/n$. Put $a=\min_i\alpha_i$.
The [logarithmic regular-variation consequence](incerto-regular-variation.md#regular-variation-representation)
makes both bounds $x^{-a+o(1)}$, so
$\log P(S>x)/\log x\to-a$. These bounds establish a logarithmic
exponent, not a limiting survival constant or full regular variation of
every dependent sum. If one allows signed variables, take a symmetric
power-tailed $V$ and $(X_1,X_2)=(V,-V)$: both have right power tails but
their sum is identically zero.

(pareto-independent-products)=
For products, let $X_1,X_2$ be **independent** unit-cutoff Pareto variables
with exponents $a,b>0$. The symbols $a,b$ are local exponents in this
two-law calculation. For $z\ge1$, condition on $X_1$ to obtain

$$
P(X_1X_2>z)=z^{-a}+a z^{-b}\int_1^z t^{b-a-1}\,dt
=\begin{cases}
\dfrac{b z^{-a}-a z^{-b}}{b-a},&a\ne b,\\[6pt]
z^{-a}(1+a\log z),&a=b.
\end{cases}
$$

For unequal exponents the smaller one determines the leading power.
Equal exponents introduce a logarithmic factor, which is slowly varying
but never constant. With $a=1$, $b=2$, the survival is $2/z-1/z^2$;
with $a=b=1$, it is $(1+\log z)/z$.
For other Pareto minima, apply the formula to
$X_1X_2/(x_{m,1}x_{m,2})$.

Independence cannot be suppressed: $X_2=X_1$ gives the square, with
exponent $a/2$. Multiplication by a dependent bounded variable can even
remove the tail: $X_2=1/X_1$ gives product one. Claims that multiplication
always preserves the heavier tail therefore need additional hypotheses.
The [power-transformation proof](incerto-regular-variation.md#regular-variation-powers)
handles deterministic powers separately.

Use [`pareto_product_logsf`](incerto-api-reference.md#api-pareto-product-logsf).

```python
import numpy as np
from xmath.incerto.power_laws import pareto_product_logsf

for a, b in ((1, 2), (1, 1)):
    survival = np.exp(pareto_product_logsf(np.log([1, 10, 100]), a, b))
    print(f"a={a:d}, b={b:d}: [" + ", ".join(f"{v:.4g}" for v in survival) + "]")
```

**Output**

```text
a=1, b=2: [1, 0.19, 0.0199]
a=1, b=1: [1, 0.3303, 0.05605]
```

Equal exponents introduce the logarithmic factor derived above.

Figure regeneration: `uv run python demos/incerto/power_laws.py` for these product survivals,
the [moment-integral figure](incerto-pareto-moment-existence.md#pareto-moments-demo),
and exact population checks. This adds no new growing-sample dominance claim;
the earlier three max-to-sum regimes remain unchanged.

## Caveats

- Empirical data rarely follows an exact Pareto law from its minimum value.  A
  tail model needs a threshold choice, diagnostic plots, and sensitivity checks.
- A finite theoretical mean can still be practically hard to estimate when
  $\alpha$ is close to 1.  The issue is pre-asymptotic behavior, not merely the
  formal existence of $\mathbb E[X]$.
- The exact failure mode depends on the moment order.  At $p=\alpha$, the
  cutoff moment grows like $\log b$; for $p>\alpha$, it grows like
  $b^{p-\alpha}$.
- Do not estimate high moments of heavy-tailed samples without checking whether
  those moments are implied by the fitted tail exponent.

(pareto-regime-mixture)=
## A rare Pareto regime retains its power tail

The regime question in SCoFT v4 appendix A, §A.1, pp. 122–124
[@taleb2025scoftv4], does not by itself specify a Pareto law.
Here is an example that does. With probability
$w\in(0,1]$, draw a Pareto variable with cutoff $x_m>0$ and exponent
$\alpha>0$; otherwise draw a uniform variable on $[0,x_m]$.
For $u\ge x_m$, conditioning on the regime proves

$$
P(X>u)=w(x_m/u)^\alpha.
$$

For every positive order $q$, the mixture moment is
$(1-w)x_m^q/(q+1)+w\mathbb E[X_{\rm Pareto}^q]$.
The [existing moment proof](incerto-pareto-moment-existence.md) therefore
implies finiteness exactly when $q<\alpha$. A small positive mixture
weight changes the survival constant but not the exponent or moment gate.
At $w=0$ the model becomes bounded and all positive moments exist.

With $w=0.02$, $x_m=1$, $\alpha=3/2$, survival at $u=4$ is
$0.0025$, the mean is $0.55$, and the second moment is infinite.
These are population values; they do not predict which tail observations
will occur in a small sample. The [catalog's normal location mixture](incerto-tail-class-catalog.md#tail-catalog-regimes)
has a different tail class even when its regimes are widely separated.
The chapter-6 demo prints this example without sampling or clipping it.

## References

- @taleb2025scoftv4, §§5.1.1 and 5.2, pp. 95–100: scaling, sums,
  products and powers. The new finite-sum bounds and product formulas above
  are proved with explicit support and dependence assumptions.
- Darling, "The Influence of the Maximum Term in the Addition of Independent
  Random Variables," *Transactions of the American Mathematical Society*
  **73** (1952), 95–107, Theorem 5.1
  [DOI](https://doi.org/10.1090/S0002-9947-1952-0048726-0).
- Kevei and Mason, "The limit distribution of ratios of jumps and sums of
  jumps of subordinators," *ALEA* **11** (2014), 631–642;
  p. 632, Theorem 1.1 and the discussion of the iid result
  [PDF](https://alea.impa.br/articles/v11/11-30.pdf#page=2).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).

## Backlinks

- Generalizes to: [Regular Variation](./incerto-regular-variation.md); moment consequences are covered by
  [Karamata's theorem](./incerto-karamata.md).
- Used by: [Double Pareto Distribution](./incerto-double-pareto.md),
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md),
  [Pre-Asymptotic LLN Behavior](./incerto-lln-preasymptotic.md),
  [Max-to-Sum Ratio](./incerto-max-to-sum-ratio.md),
  and [Hill Estimator](./incerto-hill-estimator.md).
