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
questions. Let $X$ be a real random variable and let $g(X)$ be its payoff.
Probability notation follows the
[shared notation](https://xshi19.github.io/math/notation/).
An ordinary finite expected payoff requires $\mathbb E|g(X)|<\infty$.

(payoffs-hit-rate)=
## The frequency of gains does not determine their mean

The two-outcome calculation separates gain frequency from expected payoff.

:::{prf:proposition} Mean of a two-outcome payoff
:label: prop-asymmetric-payoffs-two-outcome-mean

Let $a,b>0$ and $q\in[0,1]$. If a payoff equals $a$ with probability
$1-q$ and $-b$ with probability $q$, its gain probability is $1-q$ and
its mean is $(1-q)a-qb$.
:::

:::{prf:proof}
Only the outcome $a$ is positive. Summing each payoff times its probability
gives $(1-q)a+q(-b)=(1-q)a-qb$.
:::

Consider this finite payoff table, in arbitrary payoff units:

| Outcome probability | Payoff A | Payoff B |
| --- | --- | --- |
| $0.98$ | $1$ | $-1$ |
| $0.02$ | $-80$ | $80$ |

Direct summation gives $\mathbb E[A]=-0.62$ and $\mathbb E[B]=0.62$.
The higher probability of a gain belongs to the lower expected payoff.
Both payoffs are bounded; fat tails are unnecessary for this distinction.

This is a one-period expectation calculation. A repeated strategy with a
finite resource constraint needs a model of [ruin](incerto-ruin.md) as well.

(payoffs-jensen)=
## Curvature and Jensen's inequality

We prove the differentiable, integrable case of Jensen's inequality.

:::{prf:proposition} Jensen's inequality for differentiable payoffs
:label: prop-asymmetric-payoffs-jensen

Suppose $X$ takes values in an open interval $I$, its mean
$\mu=\mathbb E X$ belongs to $I$, and $\mathbb E|X|<\infty$.
If $g:I\to\mathbb R$ is differentiable and convex, and
$\mathbb E|g(X)|<\infty$, then

$$
\mathbb E g(X)\ge g(\mathbb E X).
$$

For differentiable concave $g$ under the same integrability assumptions,
the inequality reverses.
:::

:::{prf:proof}
Convexity gives the supporting-line
inequality $g(x)\ge g(\mu)+g'(\mu)(x-\mu)$ for every $x\in I$:
the secant slopes on either side of $\mu$ bracket $g'(\mu)$.
Taking expectations of this integrable inequality proves the claim.
Applying it to $-g$ reverses the inequality for concave $g$.
:::

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

The following comparison makes the limitation of variance explicit.

:::{prf:proposition} Equal variance does not order convex payoffs
:label: prop-asymmetric-payoffs-equal-variance

Let $X$ assign probability $1/2$ to each of $\pm1$, and let $Y$ assign
probability $3/4$ to zero and $1/8$ to each of $\pm2$. Both have mean
zero and variance one, but

$$
\mathbb E|X|=1>\tfrac12=\mathbb E|Y|,\qquad
\mathbb EX^4=1<4=\mathbb EY^4.
$$

Thus neither law has the larger expectation for every convex payoff.
:::

:::{prf:proof}
Symmetry gives the zero means. The second moments are $1$ and
$(1/8)4+(1/8)4=1$. Summing absolute values and fourth powers gives the
displayed expectations. Both $x\mapsto|x|$ and $x\mapsto x^4$ are convex,
and their expectation comparisons go in opposite directions.
:::

(payoffs-tail-transform)=
## A transformation changes the tail and its moment requirements

We obtain the power-payoff law by inverting its exceedance event.

:::{prf:proposition} Power payoffs of an exact Pareto variable
:label: prop-asymmetric-payoffs-power-tail

Let $X$ be Pareto Type I with lower cutoff $x_m>0$ and exponent
$\alpha>0$, so $P(X>x)=(x_m/x)^\alpha$ for $x\ge x_m$.
For $r>0$ and $g(x)=x^r$, the payoff has survival

$$
P(g(X)>y)=\left(\frac{x_m^r}{y}\right)^{\alpha/r},
\qquad y\ge x_m^r.
$$

Its survival is one below $x_m^r$, its tail exponent is $\alpha/r$, and
its mean is finite exactly when $r<\alpha$.
:::

:::{prf:proof}
For $y\ge x_m^r$, monotonicity of the positive power gives
$P(X^r>y)=P(X>y^{1/r})=(x_m/y^{1/r})^\alpha$.
The support gives the survival below $x_m^r$. The
[Pareto moment proof](incerto-pareto-moment-existence.md) gives the stated
mean boundary.
:::

For example, $\alpha=3/2$ gives a finite mean for $X$ and an infinite mean
for $X^2$. The finite-expectation version of Jensen above cannot be applied
to that quadratic payoff with both sides finite.

The capped payoff has a different integrability requirement.

:::{prf:proposition} Exponential moments and a mean under capping
:label: prop-asymmetric-payoffs-capping

Let $X\ge0$ be a finite random variable and $c>0$. For
$g(x)=\min(x,c)$ and every $t\in\mathbb R$,
$\mathbb E e^{t g(X)}\le e^{|t|c}<\infty$.
If $X$ is Pareto Type I with $x_m=1$ and $\alpha=3/2$, then
$\mathbb E\min(X,4)=2$, whereas $\mathbb EX=3$.
:::

:::{prf:proof}
Since $0\le g(X)\le c$, the exponential bound holds pointwise and hence
in expectation. In the specified Pareto case, integration of survival gives

$$
\mathbb E\min(X,4)=\int_0^4P(X>v)\,dv
=1+\int_1^4v^{-3/2}\,dv=2,
\qquad \mathbb E X=3.
$$
The uncapped value follows from $1+\int_1^\infty v^{-3/2}\,dv=3$.
:::

This cap bounds the realized payoff and changes its mean. It does not
estimate the uncapped mean. The
[inference note](incerto-inference-asymmetry.md#inference-bounded-payoffs)
shows why bounded payoffs have a different contamination sensitivity.

Convexity also does not supply a lower bound on the payoff.

:::{prf:proposition} A convex payoff with unbounded downside
:label: prop-asymmetric-payoffs-unbounded-downside

The function $g:\mathbb R\to\mathbb R$ given by $g(x)=e^x+x$ is
strictly convex and satisfies $\lim_{x\to-\infty}g(x)=-\infty$.
Consequently, convexity alone does not imply a finite lower payoff bound.
:::

:::{prf:proof}
The second derivative is $g''(x)=e^x>0$. As $x\to-\infty$, the
exponential term tends to zero and the linear term tends to $-\infty$.
:::

**Remark.** An explicit bound such as
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
