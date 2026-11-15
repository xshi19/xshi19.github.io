---
title: Redundancy, size, and decentralized exposure
options:
  concept:
    id: decentralization
    type: method
    prerequisites: [pareto]
    related: [ruin, asymmetric-payoffs]
    tags: [dependence, aggregation]
---

Splitting exposure can change both the size of a loss and the chance of
simultaneous losses. These are separate mechanisms. We compare systems
with the same total exposure, explicitly specifying their response to a
shock and their dependence. None of the calculations identifies a real
institution's failure law.

Use $P$, $\mathbb E$, $\operatorname{Var}$ and the Pareto exponent
$\alpha$ from the [shared notation](https://xshi19.github.io/math/notation/).
The local symbol $D_m$ denotes total damage with $m$ modules; $L$ remains
reserved for a slowly varying function in tail formulas.

(decentralization-convex-scale)=
## Splitting a fixed input to a convex damage function

Convexity gives the comparison when the same response is applied in each
module and the resulting damages add.

:::{prf:proposition} Equal splitting under an unchanged convex response
:label: prop-decentralization-convex-splitting

Let $h:[0,\infty)\to[0,\infty)$ be convex with $h(0)=0$.
For integer $m\ge1$ and $z\ge0$, suppose splitting divides input $z$
equally among $m$ modules, uses the unchanged damage function $h$ in each,
and adds their damages without extra costs or propagation. Then

$$
m h(z/m)\le h(z).
$$
:::

:::{prf:proof}
Convexity at
$z/m=(1/m)z+(1-1/m)0$ gives $h(z/m)\le h(z)/m$.
Multiplying by $m$ proves the claim.
:::

For a power response, the reduction can be calculated on every path.

:::{prf:proposition} Power damage under a common shock
:label: prop-decentralization-common-shock-power

Let $a>0$, $r>0$, $m\ge1$ be an integer, and $X\ge0$ be a finite
random shock shared by all modules. With response $h(z)=z^r$ and additive
module damages, define $D_1=(aX)^r$ and $D_m=m(aX/m)^r$. Then

$$
D_1=(aX)^r,\qquad
D_m=m(aX/m)^r=a^r m^{1-r}X^r.
$$

In particular, $D_m=m^{1-r}D_1$ pathwise. For $m>1$, the factor is less
than one if $r>1$, equals one if $r=1$, and exceeds one if $0<r<1$.
:::

:::{prf:proof}
Factoring $m^{-r}$ out of each module's power gives
$m(aX/m)^r=m^{1-r}(aX)^r$. The sign of $1-r$ gives the three comparisons.
:::

**Remarks.** The factor holds even with this perfectly common shock;
independence is not required.
If the response is applied after pooling all inputs, the result is again
$(aX)^r$ and there is no reduction. For a linear response ($r=1$),
splitting has no effect; for $0<r<1$ and $m>1$ the factor instead exceeds one.

(decentralization-pareto-tail)=
## The tail exponent need not improve

The power-payoff law gives the tail of the common-shock model.

:::{prf:corollary} Splitting rescales Pareto damage without changing its exponent
:label: cor-decentralization-pareto-damage

Let $X$ be Pareto Type I with cutoff $x_m>0$ and exponent $\alpha>0$.
For $a>0$, $r>1$, and integer $m\ge1$, set
$D_m=m(aX/m)^r$ and $d_m=a^r m^{1-r}x_m^r$. Then

$$
P(D_m>u)=\left(\frac{d_m}{u}\right)^{\alpha/r},
\qquad u\ge d_m.
$$

Below $d_m$ the survival is one. Above both minima, the split-to-unsplit
tail-probability ratio $P(D_m>u)/P(D_1>u)$ is
$m^{-\alpha(r-1)/r}$. The exponent remains $\alpha/r$ for every finite
$m$, and $\mathbb ED_m$ is finite exactly when $\alpha>r$.
:::

:::{prf:proof}
By [](#prop-decentralization-common-shock-power),
$D_m=a^rm^{1-r}X^r$. Solving $D_m>u$ for $X$ and substituting the
Pareto survival gives the displayed formula for $u\ge d_m$; the support
gives survival one below it. Dividing the two survivals gives
$(d_m/d_1)^{\alpha/r}=m^{-\alpha(r-1)/r}$. The
[power-payoff calculation](incerto-asymmetric-payoffs.md#payoffs-tail-transform)
gives the moment boundary, unchanged by the positive finite factor.
:::

**Remark.** If $\alpha\le r$, every finite split still has infinite mean
in this model.

With $a=x_m=1$, $\alpha=3/2$, $r=2$ and $m=16$, damage at each
fixed shock is reduced by $1/16$; for $u\ge1$ the survival is reduced
by $1/8$. Both damage laws still have infinite mean. A reduction in scale
does not by itself change the moment-existence boundary.

(decentralization-dependence)=
## Equal marginals, different aggregate failures

The bounded Bernoulli model isolates the effect of dependence.

:::{prf:proposition} Independent and common module failures
:label: prop-decentralization-bernoulli-failures

Let $a>0$, $q\in(0,1)$, and integer $m\ge1$. Each of $m$ equal modules
loses its entire exposure $a/m$ if an indicator $B_i$ is one, with
$P(B_i=1)=q$. Total damage is
$D_m=(a/m)\sum_{i=1}^m B_i$.

If the indicators are independent, then

$$
\mathbb E D_m=aq,\quad
\operatorname{Var}(D_m)=\frac{a^2q(1-q)}m,\quad
P(D_m=a)=q^m,\quad P(D_m>0)=1-(1-q)^m.
$$

For perfect dependence, $B_i=B$ for all $i$, we instead have $D_m=aB$,
the same mean, variance $a^2q(1-q)$, and probability $q$ of either a
positive loss or total failure.

More generally, if $k$ divides $m$, the modules form $k$ equal-size blocks,
and failures are identical within each block and independent between blocks,
the independent-case formulas hold with $m$ replaced by $k$.
:::

:::{prf:proof}
Each indicator has mean $q$ and variance $q(1-q)$. Linearity of expectation
gives $aq$ under either dependence model. For independent indicators,
variances add, giving $a^2q(1-q)/m$; the all-fail and all-survive
probabilities are $q^m$ and $(1-q)^m$. Under perfect dependence,
$D_m=aB$, so its variance is $a^2q(1-q)$ and both positive loss and total
failure occur exactly when $B=1$.
In the block model, each block carries exposure $a/k$ and one Bernoulli
indicator of mean $q$. Summing over these $k$ independent indicators gives
the same formulas with $k$ in place of $m$.
:::

For $m=10$, $q=0.02$, independent total failure has probability
$0.02^{10}=1.024\times10^{-17}$, while some loss has probability
$1-0.98^{10}\approx0.182927$. Under a common shock both probabilities
are $0.02$. Splitting reduces one event probability and increases another;
the event being protected against must be specified. All these finite
Bernoulli sums are bounded, even when dependent.

Adding nominal modules inside an unchanged block does not increase the
number of independent risks in the block model above.
This is a finite dependence comparison, not a model of contagion dynamics.

## Worked calculation and sources

```{figure} ../assets/figures/incerto/lecture-scale-dependence.svg
:alt: Three vertically stacked plots show Pareto damage survival, any-loss probability on a linear axis, and total-loss probability on a logarithmic axis.

Top: exact survivals for $\alpha=3/2$, $r=2$, $a=x_m=1$ and
$m=1,4,16$. Middle and bottom: bounded block failures at $q=0.02$.
A shared logarithmic axis would compress the any-loss and common-shock
curves near its upper edge because total-loss probabilities reach about
$10^{-68}$. Such apparent near-overlap is a scale effect: at $k=40$,
any loss has probability about $0.554$, versus $0.02$ under a common shock.
They agree exactly at $k=1$. Separate linear and logarithmic panels retain
both comparisons. The Bernoulli model is bounded and is not heavy-tailed.
```

Use [`split_power_damage_logsf`](incerto-api-reference.md#api-split-power-damage-logsf), [`block_failure_summary`](incerto-api-reference.md#api-block-failure-summary).

```python
import numpy as np
from xmath.incerto.lecture_risk import split_power_damage_logsf, block_failure_summary

for modules in (1, 16):
    log_tail = split_power_damage_logsf(4, alpha=1.5, power=2, modules=modules)
    print(f"modules={modules:d}: {np.exp(log_tail):.4g}")
for blocks in (1, 10):
    print(f"blocks={blocks:d}")
    for name, value in block_failure_summary(0.02, blocks).items():
        print(f"  {name}: {value:.4g}")
```

**Output**

```text
modules=1: 0.3536
modules=16: 0.04419
blocks=1
  mean: 0.02
  variance: 0.0196
  any_failure: 0.02
  total_failure: 0.02
blocks=10
  mean: 0.02
  variance: 0.00196
  any_failure: 0.1829
  total_failure: 1.024e-17
```

The split-damage calculation assumes equal modules; the block calculation
assumes independent failures and equal exposures.

Figure regeneration: `uv run python demos/incerto/lecture_risk.py`. The script evaluates
161 thresholds from $0.01$ to $100$ and 40 block counts, prints the
worked cases, and generates the figure without sampling or external data.

The aggregation and response questions arise in @taleb2025scoftv4,
ch. 3, §3.6.1, p. 53, and §3.12, pp. 66–67.
The [attached precaution paper, v1](https://arxiv.org/pdf/1410.5787v1),
§VII.A–C, p. 6, discusses nonlinear harm and dependence when splitting
exposure. The convexity inequality, exact Pareto scaling and finite
Bernoulli comparisons above are proved here under their stated models;
they establish no general institutional or policy ranking.
