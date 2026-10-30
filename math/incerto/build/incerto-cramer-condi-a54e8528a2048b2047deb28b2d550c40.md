---
title: Cramér Exponential-Moment Condition
options:
  concept:
    id: cramer-condition
    type: theorem
    depends_on:
      - regular-variation
    tags:
      - thin-tails
      - large-deviations
      - exponential-moments
---

## Statement

A random variable $X$ satisfies a right-tail Cramér condition when there is
some $\theta>0$ such that

$$
\mathbb E[e^{\theta X}]<\infty.
$$

For two-sided large-deviation statements, one usually asks for the moment
generating function $M(\theta)=\mathbb E[e^{\theta X}]$ to be finite in an
open neighborhood of $0$.  Equivalently, the cumulant generating function
$\Lambda(\theta)=\log \mathbb E[e^{\theta X}]$ is finite near zero, and
the Cramér rate function is

$$
I(x)=\sup_\theta\{\theta x-\Lambda(\theta)\}.
$$

We use the one-sided version when the question is upper-tail
concentration; the classical two-sided Cramér large-deviation theorem requires
stronger mgf control.

The condition gives the elementary Chernoff bound

$$
\mathbb P(X>x)\le e^{-\theta x}\mathbb E[e^{\theta X}],
$$

so the upper-tail probability is bounded above by an exponentially decaying
function.  Equivalently, $\bar F(x)=O(e^{-\theta x})$ for that value of
$\theta$.

Here $\bar F(x)=\mathbb P(X>x)$ is the survival function, $\mathbb E$
is expectation, and $\theta$ is the exponential-tilting parameter. All symbols needed here are defined locally; see also
[Shared notation](https://xshi19.github.io/math/notation/).

## Thin-tail intuition

The Cramér condition is a thin-tail gate.  If exponential moments exist, then
exponential tilting, Chernoff bounds, and classical large-deviation rates have
room to operate.  If no positive exponential moment exists, those tools may
give a false sense of security.

This is the clean contrast with [Pareto-type](./incerto-pareto.md) fat
tails.  A [regularly varying](./incerto-regular-variation.md) tail can have many finite
ordinary moments, but multiplying by $e^{\theta X}$ eventually overwhelms
every power-law decay. Failure of the Cramér condition does not require
infinite variance: it can also occur when both the mean and variance are finite.

## Chernoff bound and examples

We prove the displayed Chernoff bound from Markov's inequality.  The right-tail
and two-sided Cramér conditions are definitions or standard large-deviation
hypotheses; the larger Cramér theorem is cited rather than proved here.  The
normal, exponential, and Pareto cases below are direct checks of the condition.

The Chernoff bound follows from Markov's inequality applied to the nonnegative
random variable $e^{\theta X}$:

$$
\begin{aligned}
\mathbb P(X>x)&=\mathbb P(e^{\theta X}>e^{\theta x})\\
&\le e^{-\theta x}\mathbb E[e^{\theta X}].
\end{aligned}
$$

A standard normal random variable satisfies the two-sided version because
$\mathbb E[e^{\theta X}]=e^{\theta^2/2}$ for all real $\theta$.  An exponential
random variable with rate $\lambda>0$ has mgf
$\lambda/(\lambda-\theta)$ for $\theta<\lambda$, so it also satisfies
the two-sided neighborhood condition. Its positive exponential moments exist
exactly for $0<\theta<\lambda$.

For a [Pareto Type I](./incerto-pareto.md) random variable,

$$
\begin{aligned}
&\mathbb E[e^{\theta X}]\\
&\quad=\int_{x_m}^{\infty} e^{\theta x}\alpha x_m^\alpha x^{-(\alpha+1)}\,dx.
\end{aligned}
$$

For every $\theta>0$, the exponential factor dominates the polynomial decay,
so the integral diverges.  Thus Pareto tails fail the right-tail Cramér
condition for every positive $\theta$.

## Truncated exponential moment check

(cramer-condition-truncated-moment-check)=
For an exponential variable of rate $\lambda$ and $c\ge0$,

$$
\begin{gathered}
\mathbb E[e^{\theta X}\mathbf 1_{\{X\le c\}}]\\
=\frac{\lambda}{\lambda-\theta}
  \left(1-e^{-(\lambda-\theta)c}\right),\\
0<\theta<\lambda.
\end{gathered}
$$

For example, $\lambda=1$ and $\theta=0.12$ give a finite limit
$1/0.88\approx1.13636$ as $c\to\infty$.

For Pareto Type I with $\alpha>0$, every $\theta>0$ instead gives a
lower bound that diverges. For $c\ge2x_m$, retain only the interval $(c/2,c]$:

$$
\begin{aligned}
&\mathbb E[e^{\theta X}\mathbf 1_{\{X\le c\}}]\\
&\quad\ge e^{\theta c/2}\mathbb P(c/2<X\le c)\\
&\quad=(2^\alpha-1)\left(\frac{x_m}{c}\right)^\alpha e^{\theta c/2}\\
&\quad\longrightarrow\infty.
\end{aligned}
$$

These are population calculations, independent of simulation. Even a Pareto
law with $\alpha=2.5$, finite mean, and finite variance has no positive
exponential moment.

(cramer-bernoulli)=
## An optimized bound for Bernoulli averages

For iid $X_i\sim\mathrm{Bernoulli}(p)$, $0<p<1$, independence gives
$\mathbb Ee^{tS_n}=(1-p+pe^t)^n$. Markov therefore proves, for $p<q<1$,

$$
\begin{aligned}
&P(S_n\ge nq)\\
&\quad\le\inf_{t>0}\exp\Bigl\{-n\bigl[tq\\
&\qquad-\log(1-p+pe^t)\bigr]\Bigr\}\\
&\quad=e^{-nD(q\Vert p)},
\end{aligned}
$$

where

$$
\begin{aligned}
D(q\Vert p)&=q\log(q/p)\\
&\quad+(1-q)\log\frac{1-q}{1-p}.
\end{aligned}
$$

To verify the optimum, differentiate the bracket: its derivative vanishes
at $e^t=q(1-p)/(p(1-q))>1$, and its second derivative is negative.
Substitution yields the displayed expression. At $q=1$ the bound is
$p^n$, exactly the all-success probability. At $q\le p$ we use the trivial
upper bound one. If $nq$ is not an integer, the event is
$S_n\ge\lceil nq\rceil$; the bound at the unrounded $q$ is still valid.

For $p=1/2$, $q=3/4$, $D(q\Vert p)\approx0.130812$.
This is an upper bound, not a finite-$n$ equality or a tail-probability
approximation with a proved relative error. Appendix D, pp. 187–188 of
@taleb2025scoftv4 is the context; the optimization here is proved directly.
It also avoids identifying a two-sided Chernoff bound with a one-sided
probability or dropping its prefactor.

[`bernoulli_chernoff_log_bound`](incerto-api-reference.md#api-bernoulli-chernoff-log-bound)
returns the log upper bound, which remains useful when probabilities underflow.

```python
import numpy as np
from scipy.stats import binom
from xmath.incerto.limits import bernoulli_chernoff_log_bound

for n in (20, 100, 1000):
    exact = binom.sf(int(np.ceil(.75*n))-1, n, .5)
    bound = np.exp(bernoulli_chernoff_log_bound(.5, .75, n))
    print(f"n={n}: exact={exact:.4g}, bound={bound:.4g}")
```

**Output**

```text
n=20: exact=0.02069, bound=0.07308
n=100: exact=2.818e-07, bound=2.084e-06
n=1000: exact=6.738e-59, bound=1.545e-57
```

The general large-deviation principle instead gives logarithmic upper bounds
on closed sets and lower bounds on open sets. It does not generally assert
$P(\bar X_n>x)/e^{-nI(x)}\to1$. Our Bernoulli calculation needs only
Markov, independence, and elementary optimization, not that general theorem.

## Caveats

- The condition is direction-specific unless the moment generating function is
  finite around both sides of zero.
- Failure of the Cramér condition does not by itself prove regular variation
  or subexponentiality.  A lognormal distribution has every positive polynomial
  moment finite but no positive exponential moment, so "all moments finite" is
  not enough for the Cramér condition.
- Empirical samples cannot prove that an exponential moment exists; a finite
  sample always has a finite empirical exponential average.
- Large-deviation results need more than the one-line condition here.  This
  page records the gate and the thin-tail contrast, not the full theorem.

## References

- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [1971, 2nd ed., Wiley](https://www.wiley-vch.de/de/fachgebiete/mathematik-und-statistik/an-introduction-to-probability-theory-and-its-applications-volume-2-978-0-471-25709-7).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  @embrechts1997modelling.
- Taleb, *Statistical Consequences of Fat Tails*, third edition (2025),
  [arXiv:2001.10488v4](https://arxiv.org/abs/2001.10488v4), ch. 1,
  pp. 3–5: motivation for distinguishing tail assumptions. The elementary
  Chernoff and model calculations are proved here; this locator is not a
  citation for the general large-deviation theorem.

## Backlinks

- Depends on: [Regular Variation](./incerto-regular-variation.md).
- Used by: [Subexponentiality](./incerto-subexponentiality.md) and thin-tail contrast
  pages.
