---
title: Pickands-Balkema-de Haan Theorem
options:
  concept:
    id: pickands-balkema-de-haan
    type: theorem
    depends_on:
      - regular-variation
      - extreme-value-index
      - generalized-pareto
      - generalized-extreme-value
    tags:
      - extreme-value-theory
      - peaks-over-threshold
---

## Statement

Let $F$ be a distribution function with finite or infinite right endpoint
$x_F=\sup\{x:F(x)<1\}$.  For a high threshold $u<x_F$, define the conditional
excess distribution

$$
\begin{gathered}
F_u(y)=P(X-u\le y\mid X>u)\\
=\frac{F(u+y)-F(u)}{1-F(u)},\\
0\le y<x_F-u.
\end{gathered}
$$

If $F$ is in the maximum domain of attraction of an extreme-value law with
extreme-value index $\xi$, then there is a positive scale function $\beta(u)$
such that

$$
\begin{gathered}
\sup_{0\le y<x_F-u}
\left|F_u(y)-G_{\xi,\beta(u)}(y)\right|\to0,\\
u\uparrow x_F.
\end{gathered}
$$

The comparison is over the excess support, using the full GPD CDF: for
$\xi<0$, set $G_{\xi,\beta}(y)=1$ at and beyond $-\beta/\xi$.  The generalized Pareto term
has scale $\beta>0$ and is

$$
\begin{gathered}
G_{\xi,\beta}(y)
=1-\left(1+\frac{\xi y}{\beta}\right)^{-1/\xi},\\
\xi\ne0,
\end{gathered}
$$

on the support where $1+\xi y/\beta>0$, and

$$
G_{0,\beta}(y)=1-e^{-y/\beta}
$$

for $\xi=0$.  The theorem is the asymptotic justification for the
peaks-over-threshold model: sufficiently high threshold exceedances are modeled
by a [generalized Pareto distribution](./incerto-generalized-pareto.md).

$X$ denotes a draw from $F$, $u$ the threshold, $x_F$ its right endpoint,
$\xi$ the extreme-value index, and $\beta(u)>0$ the excess scale.
The maximum-domain-of-attraction assumption means that for iid draws,
some $a_n>0$ and $b_n$ give $(M_n-b_n)/a_n\Rightarrow H_{\xi,0,1}$.
Here $M_n$ is the sample maximum and $\Rightarrow$ is convergence in
distribution. The GEV page defines $H$. See [Shared notation](https://xshi19.github.io/math/notation/).

## Threshold intuition

Block-maxima theory says that properly normalized maxima have only a small
number of possible limiting shapes, collected in the
[Generalized Extreme-Value Distribution](./incerto-generalized-extreme-value.md).
The Pickands-Balkema-de Haan theorem says the matching threshold view has the
same discipline: once we condition on being far enough into the tail, the
remaining excess has an approximately
[Generalized Pareto Distribution](./incerto-generalized-pareto.md).

For fat-tail work, the case $\xi>0$ is the main bridge.  It corresponds to a
[regularly varying](./incerto-regular-variation.md) right tail with exponent
$\alpha=1/\xi$.  The theorem does not say every high observation is generated
by a clean [Pareto law](./incerto-pareto.md).  It says the excess
distribution over moving high thresholds has a universal limit shape under the
same domain-of-attraction assumptions used in extreme-value theory.

## Exact Pareto excess special case

The full Pickands-Balkema-de Haan theorem is cited here, not proved.  The
calculation below proves the exact Pareto special case: once the threshold is
above the lower cutoff, the excess distribution is already generalized Pareto,
not merely asymptotically close to it.

The full theorem is a classical result of Balkema and de Haan and,
independently, Pickands.  Its proof uses the equivalence between convergence of
normalized maxima and convergence of normalized threshold excesses.  We
record the statement and use exact Pareto algebra as a checkable special
case.

If $X$ is Pareto Type I with lower cutoff $x_m$ and tail exponent $\alpha$,
then for $u\ge x_m$,

$$
\begin{gathered}
P(X-u>y\mid X>u)\\
=\frac{P(X>u+y)}{P(X>u)}\\
=\left(1+\frac{y}{u}\right)^{-\alpha}.
\end{gathered}
$$

This is exactly a generalized Pareto survival function with

$$
\xi=\frac1\alpha,
\qquad
\beta(u)=\frac{u}{\alpha}.
$$

Thus exact Pareto tails do not merely approach the generalized Pareto form;
they have it at every threshold above $x_m$.

## Exact threshold calibration

(pbdh-threshold-excess-fit)=
For $\alpha=1.7=17/10$, the population excess parameters are
$\xi=10/17$ and $\beta(u)=10u/17$. The scaled excess
$Z=(X-u)/u$, conditional on $X>u$, has survival $(1+z)^{-1.7}$.

| Threshold $u\ge x_m$ | Shape $\xi$ | Scale $\beta(u)$ | $\mathbb P(Z>1\mid X>u)$ |
| --- | --- | --- | --- |
| $x_m$ | $10/17$ | $10x_m/17$ | $2^{-1.7}$ |
| $2x_m$ | $10/17$ | $20x_m/17$ | $2^{-1.7}$ |
| $4x_m$ | $10/17$ | $40x_m/17$ | $2^{-1.7}$ |

These are exact population identities, not fitted estimates. In finite data,
estimated shape and scale fluctuate, and real tails may approach the GPD
family slowly. [Threshold selection](./incerto-tail-threshold-selection.md)
examines that modeling choice; the theorem supplies no finite-sample cutoff.

(pbdh-chapter-nine)=
## What the limit permits

SCoFT v4 §9.1.3, pp. 176–177, introduces conditional excesses
[@taleb2025scoftv4]. In the theorem above, membership in a maximum domain
of attraction and an **absolute** CDF difference are essential. Approaching
the endpoint of an arbitrary CDF does not by itself establish a GPD limit.
The full theorem is cited from @balkema1974residual and
@pickands1975statistical; this page proves only special cases and consequences.

For a standard normal parent, excesses also have a limit, but on a shrinking
scale. For $u\to\infty$ and fixed $y\ge0$, the normal-tail bounds in the
[Gaussian maxima proof](incerto-generalized-extreme-value.md#gev-gaussian-maxima)
give

$$
\begin{gathered}
P(u(X-u)>y\mid X>u)\\
=\frac{\bar\Phi(u+y/u)}{\bar\Phi(u)}\longrightarrow e^{-y}.
\end{gathered}
$$

Thus $\xi=0$ with asymptotic excess scale $1/u$, whereas exact Pareto
has $\xi=1/\alpha$ and scale $u/\alpha$. A bounded example is uniform
on $[0,b]$: above $0\le u<b$ its excess is exactly uniform on $[0,b-u]$,
a GPD with $\xi=-1$ and $\beta(u)=b-u$.

These three cases can be compared with
[`generalized_pareto`](incerto-api-reference.md#api-generalized-pareto).
The normal row is a limit comparison; the other two are exact.

```python
from scipy.stats import norm
from xmath.incerto.distributions import generalized_pareto

for u in (2, 4, 8):
    probability = norm.sf(u+1/u)/norm.sf(u)
    print(f"normal u={u}: scaled excess survival={probability:.4g}")
for xi, scale in ((0, 1), (.5, .5), (-1, 1)):
    print(f"GPD xi={xi:g}: sf(1)={generalized_pareto(xi, beta=scale).sf(1):.4g}")
```

**Output**

```text
normal u=2: scaled excess survival=0.273
normal u=4: scaled excess survival=0.3375
normal u=8: scaled excess survival=0.3596
GPD xi=0: sf(1)=0.3679
GPD xi=0.5: sf(1)=0.25
GPD xi=-1: sf(1)=0
```

Uniform closeness of CDFs does not control unbounded moments: the
[contamination counterexample](incerto-hidden-tail.md#hidden-tail-empirical)
shows why. For the excesses of one fixed parent in a maximum domain of
attraction, however, the tail structure supplies the required control when
$\xi<1$. This gives the following stronger conclusion.

(pbdh-mean-excess-corollary)=
## Corollary: convergence of the mean excess

Let $F\in\operatorname{MDA}(H_{\xi,0,1})$ with $\xi<1$, and let
$x_F=\sup\{x:F(x)<1\}$ be its finite or infinite right endpoint. Assume
$\mathbb E[(X-u_0)^+]<\infty$ for some finite $u_0<x_F$, where
$r^+=\max(r,0)$. For $u_0\le u<x_F$, write $\bar F(u)=1-F(u)>0$ and
define the [mean excess](incerto-mean-excess-function.md) by

$$
e(u)=\frac{\mathbb E[(X-u)^+]}{\bar F(u)}.
$$

For any positive PBdH scale $\beta(u)$ giving the GPD limit of shape
$\xi$ in the statement above,

$$
\frac{e(u)}{\beta(u)}\longrightarrow\frac{1}{1-\xi},
\qquad u\uparrow x_F.
$$

No continuity, density, or differentiability of $F$ is required beyond
the domain-of-attraction assumption. That assumption excludes an atom at
the right endpoint. For $0<\xi<1$, $x_F=\infty$ and
$\beta(u)\sim\xi u$. For $\xi<0$, $x_F<\infty$ and
$\beta(u)\sim-\xi(x_F-u)$. The case $\xi=0$ is included, with limit 1;
its endpoint may be finite or infinite. The excess-integrability condition
concerns only the right tail, not the existence of a two-sided mean of $X$.

### Proof of the mean-excess corollary

We prove the corollary using the domain-of-attraction characterizations
and Gumbel tail representation cited from @embrechts1997modelling.
Those general representation results are not proved here. The integrations
below explain how their tail control gives convergence of means.

By survival integration,

$$
e(u)=\frac{\int_u^{x_F}\bar F(t)\,dt}{\bar F(u)}.
$$

If $0<\xi<1$, the attraction assumption makes $\bar F$ regularly varying
at infinity with index $-1/\xi<-1$.
[Karamata's integral theorem](incerto-karamata.md) gives
$e(u)\sim\xi u/(1-\xi)$. Division by $\beta(u)\sim\xi u$
proves the claim in this case.

If $\xi<0$, put $v=x_F-u$ and $R(v)=\bar F(x_F-v)$.
The finite-endpoint characterization makes $R$ regularly varying at zero
with index $-1/\xi>0$. Karamata's theorem at zero gives

$$
e(u)=\frac{\int_0^v R(s)\,ds}{R(v)}
\sim\frac{v}{1-1/\xi}.
$$

Since $\beta(u)\sim-\xi v$, this again gives $1/(1-\xi)$.

For $\xi=0$, the cited Gumbel representation supplies an equivalent
smooth tail $g$ and a positive auxiliary function $a$: near $x_F$,
$\bar F(u)/g(u)\to1$, $a(u)\sim\beta(u)$, and, for a constant $c>0$,

$$
g(u)=c\exp\!\left(-\int_{u_1}^u\frac{dt}{a(t)}\right).
$$

Here $u_1<x_F$ is fixed; $a$ is locally absolutely continuous with
$a'(u)\to0$ (where defined). This smoothness belongs to the equivalent
tail, and is not an extra assumption on $F$. Set $Q=ag$. Then
$Q'=(a'-1)g$ almost everywhere. Also $Q(u)\to0$: at a finite endpoint,
$a'$ bounded makes $a$ bounded and $g\to0$; at infinity, $a(u)=O(u)$
and a positive limit of the eventually decreasing $Q$ would force
$g(u)\ge C/u$ for some $C>0$, contradicting the assumed finite tail
integral.

Write $I(u)=\int_u^{x_F}g(t)\,dt$. Integration of the identity for $Q'$
therefore gives, for every $\varepsilon>0$ and all sufficiently high $u$,

$$
\begin{gathered}
I(u)=a(u)g(u)+\int_u^{x_F}a'(t)g(t)\,dt,\\
|I(u)-a(u)g(u)|\le\varepsilon I(u).
\end{gathered}
$$

Thus $I(u)\sim a(u)g(u)$. Tail equivalence bounds $\bar F/g$ uniformly
between $1-\varepsilon$ and $1+\varepsilon$ beyond sufficiently high $u$,
so it also gives equivalence of their tail integrals. Hence
$e(u)\sim a(u)\sim\beta(u)$, proving the zero-shape case.

The result concerns the population scale and a threshold tending to the
endpoint. A fitted shape below 1 at a chosen finite threshold does not
establish these hypotheses. The corollary supplies no convergence rate,
finite-sample threshold, or mean-error bound.

### Numerical check

For a standard normal, integrate the scaled excess survival directly,
using log survivals to avoid dividing tiny probabilities. The ratio tends
to the unit-scale exponential mean returned by
[`generalized_pareto`](incerto-api-reference.md#api-generalized-pareto).
The same quadrature checks the exact GPD cases with shapes $1/2$ and $-1$.
These few numerical values illustrate the corollary; they do not prove it
or test extreme-parameter accuracy.

:::{dropdown} Code and output: mean-excess limits
```python
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from xmath.incerto.distributions import generalized_pareto

target = generalized_pareto(0, beta=1).mean()
for u in (2, 4, 8):
    ratio = quad(lambda z: np.exp(norm.logsf(u+z/u)-norm.logsf(u)),
                 0, np.inf)[0]
    print(f"normal u={u}: e/beta={ratio:.4g}, limit={target:g}")
for xi in (.5, -1):
    law = generalized_pareto(xi, beta=1)
    ratio = quad(law.sf, 0, law.support()[1])[0]
    print(f"exact GPD xi={xi:g}: e/beta={ratio:.4g}, mean={law.mean():.4g}")
```

**Output**

```text
normal u=2: e/beta=0.7464, limit=1
normal u=4: e/beta=0.9024, limit=1
normal u=8: e/beta=0.9709, limit=1
exact GPD xi=0.5: e/beta=2, mean=2
exact GPD xi=-1: e/beta=0.5, mean=0.5
```
:::

## Caveats

- The theorem is asymptotic in the threshold.  It does not choose the threshold
  for a finite dataset.
- Dependence, seasonality, volatility clustering, rounding, truncation, and
  mixtures can all distort threshold exceedances.
- A generalized Pareto fit is a tail model, not a proof that the full
  distribution is Pareto.
- When $\xi\ge1$, the fitted generalized Pareto model has no finite mean.  When
  $\xi\ge1/2$, it has no finite variance.  These moment boundaries are often
  the practical reason the shape estimate matters.
- Raising the threshold can reduce tail-approximation bias but leaves fewer
  exceedances; it does not guarantee a monotone improvement in a fitted model.

## References

- Balkema and de Haan, "Residual Life Time at Great Age"
  @balkema1974residual.
- Pickands, "Statistical Inference Using Extreme Order Statistics"
  @pickands1975statistical.
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  @embrechts1997modelling.
- Coles, *An Introduction to Statistical Modeling of Extreme Values*
  @coles2001introduction.

## Backlinks

- Depends on: [Regular Variation](./incerto-regular-variation.md),
  [Extreme-Value Index](./incerto-extreme-value-index.md),
  [Generalized Pareto Distribution](./incerto-generalized-pareto.md),
  [Generalized Extreme-Value Distribution](./incerto-generalized-extreme-value.md),
  and [Shared notation](https://xshi19.github.io/math/notation/).
- Used by: [Mean Excess Function](./incerto-mean-excess-function.md) and
  [S&P 500 Tail Diagnostics](./incerto-sp500-tail.md).
