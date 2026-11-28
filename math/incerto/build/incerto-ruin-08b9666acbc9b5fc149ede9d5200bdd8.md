---
title: Ruin and repeated exposure
options:
  concept:
    id: ruin
    type: theorem
    prerequisites: []
    related: [asymmetric-payoffs, decentralization, inference-asymmetry]
    tags: [ruin, dependence]
---

An absorbing state changes which future payoffs a path can receive. Let $T$
be the first ruin time, taking values in $\{1,2,\ldots\}\cup\{\infty\}$;
the process stays ruined after $T$. The probability law $P$ and expectation
$\mathbb E$ follow the [shared notation](https://xshi19.github.io/math/notation/).

(ruin-constant-hazard)=
## Independent repeated exposure

:::{prf:proposition} Independent exposure with constant ruin probability
:label: prop-ruin-constant-hazard

Let independent trials each fail with probability $q\in(0,1]$, and let
$T\in\{1,2,\ldots\}\cup\{\infty\}$ be the first failure time. For every
integer $N\ge0$,

$$
P(T>N)=(1-q)^N,\qquad P(T\le N)=1-(1-q)^N,
$$

where survival at $N=0$ is one, including when $q=1$. Moreover,
$P(T<\infty)=1$. In the separate case $q=0$, $P(T>N)=1$ for every
$N\ge0$ and $P(T=\infty)=1$.
:::

:::{prf:proof}
Suppose trials are independent and each causes ruin with the same
probability $q\in(0,1]$ if reached. For every integer $N\ge0$,

$$
P(T>N)=(1-q)^N,\qquad P(T\le N)=1-(1-q)^N,
$$

with survival at $N=0$ defined as one. Survival through $N$ is the
intersection of $N$ independent
nonruin events, hence the product. The events $\{T>N\}$ decrease to
$\{T=\infty\}$, so continuity of probability and $(1-q)^N\to0$ give
$P(T<\infty)=1$. At $q=0$, survival is one at every horizon instead.

:::

For $q=0.02$, ruin probabilities at horizons $10$, $100$ and $1000$ are
approximately $0.182927$, $0.867380$ and $0.9999999983$.
No heavy-tail assumption enters this calculation.

(ruin-varying-hazard)=
## What the dependence assumption can be replaced by

:::{prf:theorem} Conditional hazards and eventual ruin
:label: thm-ruin-conditional-hazards

Let $T\in\{1,2,\ldots\}\cup\{\infty\}$ be a ruin time, and define
$h_k=P(T=k\mid T\ge k)$ whenever $P(T\ge k)>0$. For every integer
$N\ge1$ with $P(T\ge N)>0$,

$$
P(T>N)=\prod_{k=1}^N(1-h_k).
$$

If $P(T\ge k)>0$ and $h_k<1$ for every $k\ge1$, then
$P(T<\infty)=1$ if and only if $\sum_{k=1}^\infty h_k=\infty$.
If a reached hazard $h_k=1$, then $P(T\le k)=1$; later hazards need not
be defined. No independence assumption is required.
:::

:::{prf:proof}
When $P(T\ge k)>0$, define the conditional hazard
$h_k=P(T=k\mid T\ge k)$. For an integer horizon $N\ge1$ with
$P(T\ge N)>0$, the chain rule gives

$$
P(T>N)=\prod_{k=1}^N(1-h_k).
$$

By continuity from above, $P(T=\infty)=\lim_{N\to\infty}P(T>N)$.
If $0\le h_k<1$ for all
$k$, eventual ruin has probability one **if and only if**
$\sum_{k=1}^{\infty}h_k=\infty$. If the sum
diverges, $\log(1-h_k)\le-h_k$ forces the product to zero. If the sum
converges, eventually $h_k\le1/2$ and
$\log(1-h_k)\ge-2h_k$. The tail log-sum then has a finite lower bound;
the finite initial product is positive, so the infinite product is positive.
If some reached hazard is one, ruin is certain by that step separately;
all later survival probabilities are zero, and no later hazards are needed.
:::

:::{prf:corollary} Positive hazards with positive survival
:label: cor-ruin-summable-hazards

If a ruin time has conditional hazards $h_k=1/(k+1)^2$ for every $k\ge1$,
then $P(T>N)=(N+2)/(2(N+1))$ for all integers $N\ge0$, and
$P(T<\infty)=1/2$.
:::

:::{prf:proof}
Applying [](#thm-ruin-conditional-hazards) and cancelling successive factors gives

$$
P(T>N)=\prod_{k=1}^N\frac{k(k+2)}{(k+1)^2}
=\frac{N+2}{2(N+1)}\longrightarrow\frac12.
$$

Continuity from above gives the infinite-horizon survival $1/2$.
:::

A positive hazard at every step is therefore insufficient for certain ruin.
Marginal failure probabilities also cannot replace conditional hazards.

:::{prf:proposition} Identical marginal risks under complete dependence
:label: prop-ruin-shared-failure

Let every potential trial-failure indicator equal one Bernoulli variable $B$
with $P(B=1)=q\in[0,1]$, and let $T$ be the first failure time.
Each marginal failure probability is $q$, but $P(T\le N)=q$ for every
integer $N\ge1$.
:::

:::{prf:proof}
If $B=1$, then $T=1$; if $B=0$, then $T=\infty$. Thus
$\{T\le N\}=\{B=1\}$ for every $N\ge1$.
:::

Surviving the first trial reveals $B=0$. Marginals alone do not determine repeated risk.

(ruin-ensemble-time)=
## Growing ensemble expectation and a terminating path

:::{prf:corollary} Growing mean with an eventually zero resource path
:label: cor-ruin-growing-mean

Under the independent constant-hazard model of
[](#prop-ruin-constant-hazard), let $q\in(0,1)$, $a>1$, $W_0>0$, and
$W_k=W_0a^k\mathbf1_{\{T>k\}}$ for integers $k\ge0$. Then
$\mathbb E W_k=W_0[a(1-q)]^k$, while $W_k=0$ eventually almost surely
and $N^{-1}\sum_{k=1}^N W_k\to0$ almost surely. If $a(1-q)>1$, the
expectation tends to infinity despite these pathwise limits.
:::

:::{prf:proof}
Take $q\in(0,1)$, $a>1$, and starting resource $W_0>0$. Let

$$
W_k=W_0a^k\mathbf1_{\{T>k\}}
$$

under the constant-hazard model. This gives
$\mathbb E W_k=W_0[a(1-q)]^k$. If $a(1-q)>1$, the expectation grows
without bound. Nevertheless $T<\infty$ almost surely, so on almost every
path $W_k=0$ eventually and

$$
\frac1N\sum_{k=1}^NW_k\longrightarrow0.
$$

By [](#prop-ruin-constant-hazard), $T<\infty$ almost surely. After the finite
ruin time the sum has only finitely
many nonzero, finite terms, while its denominator grows.
:::

Here $W_k$ denotes resources at step $k$, not a mixing variable.
There is no contradiction with the expectation, which is carried by the increasingly
rare surviving paths. With $W_0=1$, $q=0.02$ and $a=1.04$, the ensemble
factor per step is $1.0192$. This example proves a particular failure of
interchanging a path limit and expectation; it is not a general ergodic theorem.

An absorbing barrier can also make the order of additive payoffs matter.

:::{prf:proposition} Equal increment sums with different ruin outcomes
:label: prop-ruin-increment-order

Start with resource $3$, add successive increments while resources remain
positive, and set resources permanently to zero on first reaching a
nonpositive value. The increments $(-4,4)$ give the path $(3,0,0)$;
the reversed increments $(4,-4)$ give $(3,7,3)$. Both increment sums are zero.
:::

:::{prf:proof}
In the first order, $3-4=-1$ triggers absorption immediately. In the second,
$3+4=7$ and $7-4=3$ remain positive. The increment sum is $-4+4=0$.
:::

## Worked calculation and source scope

```{figure} ../assets/figures/incerto/lecture-ruin.svg
:alt: Two vertically stacked plots show survival under different hazards above and growing ensemble resources with shrinking survival below.

Exact finite-horizon curves. The summable-hazard model retains positive
infinite-horizon survival. The lower panel uses $q=0.02$, $a=1.04$;
it compares ensemble expectation with survival, not with simulated paths.
```

Use [`survival_from_hazards`](incerto-api-reference.md#api-survival-from-hazards), [`absorbing_resource_path`](incerto-api-reference.md#api-absorbing-resource-path).

```python
from xmath.incerto.lecture_risk import survival_from_hazards, absorbing_resource_path

survival = survival_from_hazards([0.02] * 3)
print("Survival through step 3: [" + ", ".join(f"{v:.4g}" for v in survival) + "]")
for increments in ([-4, 4], [4, -4]):
    path = absorbing_resource_path(3, increments)
    print(f"{increments}: [" + ", ".join(f"{v:.4g}" for v in path) + "]")
```

**Output**

```text
Survival through step 3: [1, 0.98, 0.9604, 0.9412]
[-4, 4]: [3, 0, 0]
[4, -4]: [3, 7, 3]
```

Ruin is absorbing.

Figure regeneration: `uv run python demos/incerto/lecture_risk.py`. The grids contain
$N=0,\ldots,400$ and $k=0,\ldots,200$; the two ordered increment
examples are printed separately. There is no RNG or external data.

The chapter context is @taleb2025scoftv4, ch. 3, §3.11, pp. 63–66.
The attached [precaution paper, v1](https://arxiv.org/pdf/1410.5787v1),
§II.B and §III, pp. 2–3, discusses irreversible failure and repeated risk.
Our proofs supply explicit hazard assumptions and counterexamples to an
unqualified eventual-ruin claim. The paper's policy and infinite-cost
arguments are not conclusions of these probability calculations.
