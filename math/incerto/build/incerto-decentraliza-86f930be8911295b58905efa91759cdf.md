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

Let $h:[0,\infty)\to[0,\infty)$ be convex with $h(0)=0$.
For integer $m\ge1$ and $z\ge0$,

$$
m h(z/m)\le h(z).
$$

We prove the inequality directly: convexity at
$z/m=(1/m)z+(1-1/m)0$ gives $h(z/m)\le h(z)/m$.
It applies to a model where total input $z$ is divided equally and the
resulting module damages add. It assumes that division does not change
$h$, add costs, or cause damage to propagate between modules.

For $h(z)=z^r$, $r>1$, total exposure $a>0$ and a common nonnegative
shock $X$, compare

$$
D_1=(aX)^r,\qquad
D_m=m(aX/m)^r=a^r m^{1-r}X^r.
$$

The reduction factor $m^{1-r}$ holds on every path, even with this perfectly
common shock. It comes from the local nonlinear response, not independence.
If the response is applied after pooling all inputs, the result is again
$(aX)^r$ and there is no reduction. For a linear response ($r=1$),
splitting has no effect; for $0<r<1$ and $m>1$ the factor instead exceeds one.

(decentralization-pareto-tail)=
## The tail exponent need not improve

Suppose now that $X$ is exact [Pareto](incerto-pareto.md) with
$x_m>0$ and $\alpha>0$. Put $d_m=a^r m^{1-r}x_m^r$.
By solving the event $D_m>u$ for $X$,

$$
P(D_m>u)=\left(\frac{d_m}{u}\right)^{\alpha/r},
\qquad u\ge d_m.
$$

Below $d_m$ the survival is one. Above both minima, the split-to-unsplit
tail-probability ratio is $m^{-\alpha(r-1)/r}$. The exponent remains
$\alpha/r$ for every finite $m$. Consequently $\mathbb E D_m$ is finite
exactly when $\alpha>r$, by the
[power-payoff calculation](incerto-asymmetric-payoffs.md#payoffs-tail-transform).
If $\alpha\le r$, every finite split still has infinite mean in this model.

With $a=x_m=1$, $\alpha=3/2$, $r=2$ and $m=16$, damage at each
fixed shock is reduced by $1/16$; for $u\ge1$ the survival is reduced
by $1/8$. Both damage laws still have infinite mean. A reduction in scale
does not by itself change the moment-existence boundary.

(decentralization-dependence)=
## Equal marginals, different aggregate failures

Consider a second, bounded model. Each of $m$ equal modules loses its
entire exposure $a/m$ if an indicator $B_i$ is one, with
$P(B_i=1)=q\in(0,1)$. Total damage is
$D_m=(a/m)\sum_{i=1}^m B_i$.

If the indicators are independent, finite sums and independence give

$$
\mathbb E D_m=aq,\quad
\operatorname{Var}(D_m)=\frac{a^2q(1-q)}m,\quad
P(D_m=a)=q^m,\quad P(D_m>0)=1-(1-q)^m.
$$

For perfect dependence, $B_i=B$ for all $i$, we instead have $D_m=aB$,
the same mean, variance $a^2q(1-q)$, and probability $q$ of either a
positive loss or total failure. These formulas follow from Bernoulli
moments and the all-fail/all-survive intersections; no limit theorem is used.

For $m=10$, $q=0.02$, independent total failure has probability
$0.02^{10}=1.024\times10^{-17}$, while some loss has probability
$1-0.98^{10}\approx0.182927$. Under a common shock both probabilities
are $0.02$. Splitting reduces one event probability and increases another;
the event being protected against must be specified. All these finite
Bernoulli sums are bounded, even when dependent.

Intermediate cases can be represented by $k$ independent equal-size
blocks, with perfect dependence within each block. Replacing $m$ by $k$
in these formulas gives the total damage law. Adding nominal modules
inside an unchanged block does not increase the number of independent risks.
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
