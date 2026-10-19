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

Suppose trials are independent and each causes ruin with the same
probability $q\in(0,1]$ if reached. For every integer $N\ge0$,

$$
P(T>N)=(1-q)^N,\qquad P(T\le N)=1-(1-q)^N,
$$

with survival at $N=0$ defined as one. We prove both the finite formula and
the limit: survival through $N$ is the intersection of $N$ independent
nonruin events, hence the product. The events $\{T>N\}$ decrease to
$\{T=\infty\}$, so continuity of probability and $(1-q)^N\to0$ give
$P(T<\infty)=1$. At $q=0$, survival is one at every horizon instead.

For $q=0.02$, ruin probabilities at horizons $10$, $100$ and $1000$ are
approximately $0.182927$, $0.867380$ and $0.9999999983$.
No heavy-tail assumption enters this calculation.

(ruin-varying-hazard)=
## What the dependence assumption can be replaced by

When $P(T\ge k)>0$, define the conditional hazard
$h_k=P(T=k\mid T\ge k)$. The chain rule gives

$$
P(T>N)=\prod_{k=1}^N(1-h_k).
$$

This identity needs no independence assumption. If $0\le h_k<1$ for all
$k$, eventual ruin has probability one **if and only if**
$\sum_{k=1}^{\infty}h_k=\infty$. We prove the equivalence. If the sum
diverges, $\log(1-h_k)\le-h_k$ forces the product to zero. If the sum
converges, eventually $h_k\le1/2$ and
$\log(1-h_k)\ge-2h_k$. The tail log-sum then has a finite lower bound;
the finite initial product is positive, so the infinite product is positive.
If some reached hazard is one, ruin is certain by that step separately.

A positive hazard at every step is insufficient. For
$h_k=1/(k+1)^2$,

$$
P(T>N)=\prod_{k=1}^N\frac{k(k+2)}{(k+1)^2}
=\frac{N+2}{2(N+1)}\longrightarrow\frac12.
$$

The formula follows by cancelling successive factors. It gives eventual
ruin probability $1/2$, although every conditional hazard is positive.
For a different boundary, let all potential trial-failure indicators equal
one Bernoulli variable $B$ with $P(B=1)=q$. Each marginal failure
probability is $q$, but $P(T\le N)=q$ for every $N\ge1$: surviving
the first trial reveals $B=0$. Marginals alone do not determine repeated risk.

(ruin-ensemble-time)=
## Growing ensemble expectation and a terminating path

Take $q\in(0,1)$, $a>1$, and starting resource $W_0>0$.
Here $W_k$ denotes resources at step $k$, not a mixing variable. Let

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

The proof is pathwise: after the finite ruin time the sum has only finitely
many nonzero, finite terms, while its denominator grows. There is no
contradiction with the expectation, which is carried by the increasingly
rare surviving paths. With $W_0=1$, $q=0.02$ and $a=1.04$, the ensemble
factor per step is $1.0192$. This example proves a particular failure of
interchanging a path limit and expectation; it is not a general ergodic theorem.

An absorbing barrier can also make the order of additive payoffs matter.
Starting at $3$, the increments $(-4,4)$ lead to permanent ruin at the
first step. Reversing them gives resources $(3,7,3)$ with no ruin.
Both increment sums are zero. The rule here sets resources to zero on
first reaching a nonpositive value and ignores all later increments.

## Worked calculation and source scope

```{figure} ../assets/figures/incerto/lecture-ruin.svg
:alt: Two vertically stacked plots show survival under different hazards above and growing ensemble resources with shrinking survival below.

Exact finite-horizon curves. The summable-hazard model retains positive
infinite-horizon survival. The lower panel uses $q=0.02$, $a=1.04$;
it compares ensemble expectation with survival, not with simulated paths.
```

Run `uv run python demos/incerto/lecture_risk.py`. The grids contain
$N=0,\ldots,400$ and $k=0,\ldots,200$; the two ordered increment
examples are printed separately. There is no RNG or external data.

The chapter context is @taleb2025scoftv4, ch. 3, §3.11, pp. 63–66.
The attached [precaution paper, v1](https://arxiv.org/pdf/1410.5787v1),
§II.B and §III, pp. 2–3, discusses irreversible failure and repeated risk.
Our proofs supply explicit hazard assumptions and counterexamples to an
unqualified eventual-ruin claim. The paper's policy and infinite-cost
arguments are not conclusions of these probability calculations.
