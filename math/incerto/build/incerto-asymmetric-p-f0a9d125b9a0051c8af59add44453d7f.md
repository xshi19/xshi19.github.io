---
title: Asymmetric payoffs and exposure
options:
  concept:
    id: asymmetric-payoffs
    type: method
    prerequisites: [pareto-moment-existence]
    related: [inference-asymmetry, ruin, decentralization]
    tags: [payoffs, convexity]
---

A forecast of an event and the payoff attached to that event answer different
questions. Let $X$ be a real random variable with CDF $F$ and let $g(X)$
be its payoff. We use $g$ for the transformation, reserving $F$ for the CDF
as in the [shared notation](https://xshi19.github.io/math/notation/).
An ordinary finite expected payoff requires $\mathbb E|g(X)|<\infty$.

(payoffs-hit-rate)=
## The frequency of gains does not determine their mean

Consider this finite payoff table, in arbitrary payoff units:

| Outcome probability | Payoff A | Payoff B |
| --- | --- | --- |
| $0.98$ | $1$ | $-1$ |
| $0.02$ | $-80$ | $80$ |

Direct summation gives $\mathbb E[A]=-0.62$ and $\mathbb E[B]=0.62$.
The higher probability of a gain belongs to the lower expected payoff.
Both payoffs are bounded; fat tails are unnecessary for this distinction.
More generally, a payoff equal to $a>0$ with probability $1-q$ and to
$-b<0$ with probability $q$ has mean $(1-q)a-qb$.

This is a one-period expectation calculation. A repeated strategy with a
finite resource constraint needs a model of [ruin](incerto-ruin.md) as well.

(payoffs-jensen)=
## Curvature and Jensen's inequality

Suppose $X$ takes values in an open interval $I$, its mean
$\mu=\mathbb E X$ belongs to $I$, and $\mathbb E|X|<\infty$.
If $g:I\to\mathbb R$ is differentiable and convex, and
$\mathbb E|g(X)|<\infty$, then

$$
\mathbb E g(X)\ge g(\mathbb E X).
$$

We prove this differentiable case. Convexity gives the supporting-line
inequality $g(x)\ge g(\mu)+g'(\mu)(x-\mu)$ for every $x\in I$:
the secant slopes on either side of $\mu$ bracket $g'(\mu)$.
Taking expectations of this integrable inequality proves the claim.
Applying it to $-g$ reverses the inequality for concave $g$.

For a symmetric two-point perturbation $X=\mu\pm s$, each with probability
$1/2$, the mean remains $\mu$. With $\mu=4$ and $s=2$,

$$
\mathbb E X^2=20>16=(\mathbb E X)^2,
\qquad
\mathbb E\sqrt X=\frac{\sqrt2+\sqrt6}{2}<2.
$$

For the quadratic payoff $g(x)=x^2$, the gain in expectation is exactly
$s^2$; for $g(x)=-x^2$, it is $-s^2$. These are changes relative to the
constant input $\mu$. Comparing two arbitrary distributions requires a
specified ordering of their spreads; variance alone does not order
expectations of every convex function.

For example, compare the symmetric law at $\pm1$ with the law assigning
probability $3/4$ to zero and $1/8$ to each of $\pm2$. Both have mean
zero and variance one. Their expectations of $|X|$ are $1$ and $1/2$,
whereas their expectations of $X^4$ are $1$ and $4$. Both functions are
convex, so neither law gives the larger expectation for every convex payoff.

(payoffs-tail-transform)=
## A transformation changes the tail and its moment requirements

For exact Pareto $X$ with $x_m>0$, $\alpha>0$, and a power payoff
$g(x)=x^r$ with $r>0$, direct inversion of the event gives

$$
P(g(X)>y)=P(X>y^{1/r})
=\left(\frac{x_m^r}{y}\right)^{\alpha/r},
\qquad y\ge x_m^r.
$$

The payoff exponent is $\alpha/r$. Its mean is finite exactly when
$r<\alpha$, by the [Pareto moment proof](incerto-pareto-moment-existence.md).
For example, $\alpha=3/2$ gives a finite mean for $X$ and an infinite mean
for $X^2$. The finite-expectation version of Jensen above cannot be applied
to that quadratic payoff with both sides finite.

If instead $g(x)=\min(x,c)$ on $x\ge0$ with $c>0$, the payoff lies in
$[0,c]$, so $\mathbb E e^{t g(X)}\le e^{|t|c}$ for every real $t$.
For $X$ Pareto with $x_m=1$, $\alpha=3/2$, and $c=4$, integration of
the survival gives

$$
\mathbb E\min(X,4)=\int_0^4P(X>v)\,dv
=1+\int_1^4v^{-3/2}\,dv=2,
\qquad \mathbb E X=3.
$$

This cap bounds the realized payoff and changes its mean. It does not
estimate the uncapped mean. The
[inference note](incerto-inference-asymmetry.md#inference-bounded-payoffs)
shows why bounded payoffs have a different contamination sensitivity.

Convexity alone does not bound downside: $g(x)=e^x+x$ has
$g''(x)=e^x>0$ but tends to $-\infty$ as $x\to-\infty$. An explicit bound such as
$g(x)=\max(x-u,0)-d\ge-d$, for fixed $u$ and $d\ge0$, is a separate
property. Repetition, costs and the distribution of $X$ still matter.

## Worked calculation and sources

Use [`finite_payoff_mean`](incerto-api-reference.md#api-finite-payoff-mean).

```python
from xmath.incerto.lecture_risk import finite_payoff_mean

probabilities = [0.98, 0.02]
print(f"A: {finite_payoff_mean(probabilities, [1, -80]):.4g}")
print(f"B: {finite_payoff_mean(probabilities, [-1, 80]):.4g}")
print(f"Convex gap: {finite_payoff_mean([0.5, 0.5], [2**2, 6**2]) - 4**2:.4g}")
```

**Output**

```text
A: -0.62
B: 0.62
Convex gap: 4
```

A’s higher gain frequency does not imply a higher expected payoff.

The lower panel of the [shared figure](incerto-inference-asymmetry.md#worked-calculation)
plots $\mathbb E[(4\pm s)^2]-16=s^2$ and its negative for
$0\le s\le3$. Reproduce the figure and finite payoff table with
`uv run python demos/incerto/lecture_risk.py`. These are exact finite
sums on 121 spread values, without simulation or external data.

The source context is @taleb2025scoftv4, ch. 3, §3.4.1, pp. 41–43,
and §3.10, pp. 60–63. The chapter motivates distinguishing state from
exposure. The finite-table calculation, differentiable Jensen argument,
power-tail transformation and capped-mean calculation are proved here.
