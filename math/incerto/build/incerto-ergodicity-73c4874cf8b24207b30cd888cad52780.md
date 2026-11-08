---
title: Time averages and ensemble averages
options:
  concept:
    id: ergodicity
    type: theorem
    depends_on: [ruin]
    tags: [law-of-large-numbers, growth, dependence]
---

## Which observable is averaged?

For a stationary process $(X_k)$, an ensemble mean is $\mathbb E[g(X_0)]$;
a time average is $n^{-1}\sum_{k=0}^{n-1}g(X_k)$ along one realization.
The observable $g$ and its integrability are part of the question.
Stationarity means that shifting every time index preserves joint laws.
Ergodicity means that events invariant under that shift have probability
zero or one.

For an integrable observable of a stationary process, Birkhoff's theorem
gives the almost-sure limit $\mathbb E[g(X_0)\mid\mathcal I]$, where
$\mathcal I$ is the invariant sigma-field. Under ergodicity this is the
constant $\mathbb E[g(X_0)]$. This general theorem is cited, not proved:
@durrett2019probability, Theorem 6.2.1, p. 335. The following finite examples
are proved directly.

For a stationary nonergodic example, draw $B\in\{0,1\}$ once with equal
probabilities and put $X_k=B$ at every time. The time average is $B$,
whereas the ensemble mean is $1/2$. The event $\{B=1\}$ is invariant
and has probability $1/2$. Independence fails even though all moments exist.

(ergodicity-growth)=
## Multiplicative growth: two different targets

Let $A_1,A_2,\ldots$ be iid positive multipliers with
$\mathbb E|\log A_1|<\infty$ and $\mathbb E A_1<\infty$.
For initial wealth $w_0>0$, define $W_n=w_0\prod_{i=1}^n A_i$.
Independence and the strong law give, respectively,

$$
\begin{gathered}
\mathbb E W_n=w_0(\mathbb E A_1)^n,\\
\frac1n\log(W_n/w_0)\xrightarrow{\mathrm{a.s.}}\mathbb E\log A_1.
\end{gathered}
$$

The first formula follows by factoring the expectation of a finite product.
The second follows by taking logarithms and averaging integrable iid
increments. Concavity of log gives
$\mathbb E\log A_1\le\log\mathbb E A_1$, strictly for a nonconstant
positive multiplier. These statements concern expected wealth and realized
log growth. They are not two competing values for the same time average.
Wealth itself generally is not stationary, so the stationary ergodic theorem
cannot be applied to $W_n$ without additional structure.

Take $A_i=1.8$ or $0.6$, each with probability $1/2$. Then

$$
\begin{gathered}
\mathbb E A_i=1.2,\\
\mathbb E\log A_i=\tfrac12\log(1.08)>0.
\end{gathered}
$$

Both targets grow, at different exponential rates. For the equally likely
pair $1.5,0.6$, expected wealth grows at rate $\log(1.05)>0$, while
the path log-growth rate is $\tfrac12\log(0.9)<0$, so $W_n\to0$ almost
surely. A positive expected payoff therefore does not determine long-run
multiplicative growth. Choosing expected utility, expected wealth or growth
as an objective is a further decision, not an ergodic theorem.

```{figure} ../assets/figures/incerto/ergodicity-growth.svg
:alt: Log wealth from four paths and sampling quantiles is compared with log expected wealth and expected log wealth for equally likely multipliers 1.5 and 0.6.

The $1.5/0.6$ example with $w_0=1$, 200 steps, 800 independent paths,
PCG64 seed 20261073. Four paths and pointwise 10th/50th/90th sampling
quantiles use log wealth, avoiding numerical overflow. The rising line is
$\log\mathbb EW_n=n\log1.05$; the falling line is
$\mathbb E\log W_n=(n/2)\log0.9$. Neither line is the empirical mean of
the displayed paths. The finite ensemble does not estimate rare wealth
contributions accurately merely because its median is stable.
```

[`finite_payoff_mean`](incerto-api-reference.md#api-finite-payoff-mean)
computes both finite expectations; [`running_mean`](incerto-api-reference.md#api-running-mean)
can average realized log increments.

```python
import numpy as np
from xmath.incerto.lecture_risk import finite_payoff_mean

a = np.array([1.5, .6])
print(f"Mean multiplier: {finite_payoff_mean([.5, .5], a):.4g}")
print(f"Mean log multiplier: {finite_payoff_mean([.5, .5], np.log(a)):.4g}")
```

**Output**

```text
Mean multiplier: 1.05
Mean log multiplier: -0.05268
```

Regenerate with `uv run python demos/incerto/limit_distributions.py`.
If $P(A_i=0)>0$, the finite-log assumption fails; iid repetition then
hits zero almost surely, and zero is absorbing. See the separate
[ruin calculation](incerto-ruin.md).

(ergodicity-growth-transform)=
## A thin-tailed rate can produce a power-tailed outcome

For a fixed horizon $T>0$, initial size $x_m>0$, and a random constant
rate $R\sim\mathrm{Exp}(\lambda)$, let $X=x_m e^{TR}$. For $x\ge x_m$,

$$
\begin{aligned}
P(X>x)&=P\!\left(R>\frac{\log(x/x_m)}T\right)\\
&=\left(\frac{x_m}{x}\right)^{\lambda/T}.
\end{aligned}
$$

Thus the outcome is exactly Pareto with $\alpha=\lambda/T$, despite the
exponential rate having all power moments. Integrating the exponential
density gives, for $p>0$,

$$
\mathbb EX^p=
\begin{cases}
x_m^p\lambda/(\lambda-pT),&pT<\lambda,\\
\infty,&pT\ge\lambda.
\end{cases}
$$

This is a change of variables at a fixed horizon. It does not establish
stationarity, ergodicity, or a model of epidemics. Specifying a right tail
alone cannot determine a two-sided rate density. More general mixing
assumptions are considered in [parameter uncertainty](incerto-mixture-uncertainty.md#mixture-growth-rate).

## Sources

@taleb2025scoftv4, appendix C, §C.1, pp. 183–185, is the growth-transform
context. The stationary and multiplicative examples above set out separate
assumptions. The historical data and policy claims of §C.2, pp. 185–186,
are outside these mathematical examples.
