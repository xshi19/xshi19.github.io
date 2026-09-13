---
title: Subexponentiality
options:
  concept:
    id: subexponentiality
    type: theorem
    depends_on:
      - regular-variation
    tags:
      - fat-tails
      - asymptotics
      - one-big-jump
---

## Statement

Let $X_1,X_2,\dots$ be iid nonnegative random variables with distribution $F$
and unbounded right support, so $\bar F(x)=\mathbb P(X>x)>0$ for every
finite $x$. The distribution is
subexponential when, for two independent copies,

$$
\frac{\mathbb P(X_1+X_2>x)}{2\mathbb P(X>x)} \to 1,
\qquad x\to\infty.
$$

Equivalently, for each fixed integer $n\ge1$, with
$S_n=X_1+\cdots+X_n$ and $M_n=\max_{1\le i\le n}X_i$,

$$
\mathbb P(S_n>x)
\sim
\mathbb P(M_n>x)
\sim n\bar F(x).
$$

Equivalently,

$$
\mathbb P(S_n>x,\; M_n\le x)=o(\bar F(x)).
$$

This is the one-big-jump principle: for a fixed number of summands, a very
large sum is usually caused by one summand being very large, not by all
summands moving moderately together.

The maximum-tail asymptotic
$\mathbb P(M_n>x)\sim n\bar F(x)$ is not the distinctive part of the theorem:
it follows for every iid distribution with an unbounded right tail and fixed
$n$.  The subexponential content is that the sum tail has the same first-order
asymptotic as the maximum tail.

[Regularly varying](./incerto-regular-variation.md) survival tails with positive
exponent, $\bar F\in RV_{-\alpha}$ for $\alpha>0$, are a standard
subexponential class.  We cite that theorem and use the exact
[Pareto family](./incerto-pareto.md) as a checkable
example.

Here $F$ is the common CDF, $n$ is fixed while $x$ grows, and
$\alpha>0$ and $x_m>0$ are the exponent and cutoff in the Pareto example.
The notation $a(x)\sim b(x)$ means $a(x)/b(x)\to1$; $o(b(x))$
denotes a term negligible after division by $b(x)$.

## One-big-jump intuition

Subexponential does not mean "lighter than exponential," and it is unrelated
to the concentration-theory phrase "sub-exponential random variable" for a
light-tailed $\psi_1$ condition.  In this context it means the opposite: the
tail is heavy enough that convolution barely changes the leading-order tail
probability.

For two iid summands, a sufficiently large sum is about twice as likely as
one summand exceeding the same threshold. For $n$ fixed summands, the leading
factor is $n$.

This is the mathematical version of a recurring Incerto warning.  In thin-tail
settings, a large aggregate deviation is often the result of many small
coordinated deviations.  In subexponential settings, the aggregate tail is
usually dominated by a single large term.  That changes how sums, ruin events,
insurance losses, and sample moments should be read.

## What we can prove directly

The statement collects several equivalent forms and one important sufficient
class.  We prove the elementary maximum-tail asymptotic and an exact Pareto
two-summand check.  The full equivalence theory and the theorem that regularly
varying tails are subexponential are cited, because their proofs need more
convolution-tail machinery than we develop here.

The theorem that regularly varying tails are subexponential goes back to
Chistyakov's convolution-tail criterion for sums of independent positive random
variables [1964](https://doi.org/10.1137/1109088).  Standard modern references with proofs are
Embrechts, Klueppelberg, and Mikosch, Appendix A3
[1997](https://doi.org/10.1007/978-3-642-33483-2), and Foss, Korshunov, and
Zachary, Chapter 3 [2013](https://doi.org/10.1007/978-1-4614-7101-1).
We do not reprove the full theorem, but the exact [Pareto](./incerto-pareto.md)
case gives the right calibration.

### Universal maximum tail

For any distribution with unbounded right tail, $\bar F(x)\to0$ as
$x\to\infty$.  For fixed $n$, iid independence gives

$$
\mathbb P(M_n\le x)
=\mathbb P(X_1\le x,\dots,X_n\le x)
=F(x)^n
=\left(1-\bar F(x)\right)^n.
$$

Therefore

$$
\mathbb P(M_n>x)
=1-\left(1-\bar F(x)\right)^n
$$

Set $u=\bar F(x)$.  Since $u\to0$ and $n$ is fixed, the binomial expansion
gives

$$
1-(1-u)^n
=nu-\binom{n}{2}u^2+O(u^3),
$$

and therefore

$$
\frac{1-(1-u)^n}{nu}
=1-\frac{n-1}{2}u+O(u^2)
\to1.
$$

Substituting back,

$$
\mathbb P(M_n>x)
\sim n\bar F(x).
$$

This maximum formula only uses iid independence, fixed $n$, and
$\bar F(x)\to0$.  For nonnegative variables, $M_n>x$ implies
$S_n=X_1+\cdots+X_n>x$, so it is a lower bound for the sum tail.  What is
special about subexponential tails is the matching upper bound:

$$
\mathbb P(S_n>x,\;M_n\le x)=o(\bar F(x)).
$$

Those are the configurations where several observations are moderately large
but none exceeds $x$.  Subexponentiality says this residual event is negligible
at the scale of one tail probability.

### Pareto upper-bound check

For the Pareto tail $\bar F(x)=(x_m/x)^\alpha$ with $x\ge x_m$, the missing
upper-bound step can be checked directly for $n=2$.  Let

$$
A_x=\{X_1+X_2>x,\;M_2\le x\}.
$$

Choose $h=x_m(x/x_m)^\gamma$ with $1/2<\gamma<1$. This keeps the units
consistent and gives $h/x\to0$ and $h\to\infty$.

(subexponentiality-cover-diagram)=
The residual event $A_x$ is the region in $[0,x]^2$ above $X_1+X_2=x$.
It is covered by two thin strips and a region where both coordinates exceed
$h$. The following table replaces the geometric plot.

| Cover event | Probability or bound |
| --- | --- |
| $x-h<X_1\le x$ | $\bar F(x-h)-\bar F(x)$ |
| $x-h<X_2\le x$ | $\bar F(x-h)-\bar F(x)$ |
| $X_1>h$ and $X_2>h$ | $\bar F(h)^2$ by independence |

Outside the cover region, a point has neither coordinate in $(x-h,x]$ and not
both coordinates above $h$.  Thus at least one coordinate is $\le h$, and the
other is $\le x-h$, so $X_1+X_2\le x$.  Therefore any point in $A_x$ must lie
in the cover: either one summand lies in $(x-h,x]$, or both summands
exceed $h$.  Hence

$$
\mathbb P(A_x)
\le
2\mathbb P(x-h<X\le x)
+\mathbb P(X_1>h,\;X_2>h).
$$

For the first term,

$$
\frac{\mathbb P(x-h<X\le x)}{\bar F(x)}
=
\frac{\bar F(x-h)-\bar F(x)}{\bar F(x)}
=
\left(\frac{x}{x-h}\right)^\alpha-1
\to0,
$$

because $h/x\to0$.  For the second term, independence gives

$$
\frac{\mathbb P(X_1>h,\;X_2>h)}{\bar F(x)}
=
\frac{\bar F(h)^2}{\bar F(x)}
=
(x/x_m)^{\alpha(1-2\gamma)}
\to0,
$$

because $\gamma>1/2$.  Thus $\mathbb P(A_x)=o(\bar F(x))$.  Since
$\{S_2>x\}$ is the disjoint union of $\{M_2>x\}$ and $A_x$,

$$
\mathbb P(X_1+X_2>x)\sim 2\bar F(x).
$$

The regularly varying theorem cited above generalizes this Pareto calculation:
the maximum formula is universal, while the negligible residual event is the
heavy-tail property.

A useful comparison table is:

| Distribution | Subexponential? | Regularly varying? |
| --- | --- | --- |
| Pareto | Yes | Yes |
| Lognormal | Yes | No |
| Weibull with shape $0<\beta<1$ | Yes | No |
| Exponential | No | No |

A broader Tail Class Catalog is planned. The lognormal and stretched
Weibull entries are cited examples, not consequences of the Pareto proof.

The exponential is an exact nonexample.  If
$X_i\sim\operatorname{Exp}(\lambda)$, then the universal maximum formula still
holds:

$$
\mathbb P(M_2>x)=2e^{-\lambda x}-e^{-2\lambda x}
\sim2e^{-\lambda x}.
$$

But the sum tail is much larger:

$$
\mathbb P(X_1+X_2>x)=e^{-\lambda x}(1+\lambda x),
$$

so

$$
\frac{\mathbb P(X_1+X_2>x)}{2\mathbb P(X>x)}
=\frac{1+\lambda x}{2}\to\infty,
$$

not $1$.

The fixed-$n$ assumption is load-bearing.  If $n$ grows with $x$, additional
large-deviation regimes can appear.

## A static bound on the residual tail

(subexponentiality-one-big-jump-check)=
The proof supplies a finite-threshold bound. Put $t=x/x_m$ and take
$\alpha=1.5$, $\gamma=0.75$. For sufficiently large $t$ with $x-h\ge x_m$,

$$
0\le\frac{\mathbb P(A_x)}{\bar F(x)}
\le 2\left[(1-t^{-1/4})^{-3/2}-1\right]+t^{-3/4}.
$$

| $t=x/x_m$ | Upper bound (rounded upward) |
| --- | --- |
| $10^4$ | $0.343428$ |
| $10^8$ | $0.030381$ |
| $10^{12}$ | $0.003004$ |

Together with $\mathbb P(M_2>x)=2\bar F(x)-\bar F(x)^2$, this bounds the
sum tail around its leading term. It illustrates how an asymptotic statement
can require large thresholds even when the model is exactly Pareto. These
are conservative analytic bounds, not measured convolution probabilities.

The fixed-$n$, large-threshold limit differs from the growing-sample
[max-to-sum ratio](./incerto-max-to-sum-ratio.md). A finite-mean Pareto law is
subexponential even though $M_n/S_n\to0$ almost surely as $n\to\infty$.

## Caveats

- The definition above is for nonnegative iid summands.  Two-sided or
  dependent data need extra assumptions before the one-big-jump reading is
  valid.
- The equivalence for $n$ summands holds for fixed $n$.  It is not a uniform
  statement over arbitrary growing horizons.
- Subexponentiality is a tail property.  It does not say the body of the
  distribution is Pareto, nor does it choose an empirical threshold.
- A simulation ratio near one is only a diagnostic.  A durable mathematical
  claim needs a theorem, a checked distributional assumption, or a citation.

## References

- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Bingham, Goldie, and Teugels, *Regular Variation* [1987](https://doi.org/10.1017/CBO9780511721434).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).
- Taleb, *Statistical Consequences of Fat Tails* [2020](https://arxiv.org/abs/2001.10488).

## Backlinks

- Depends on: [Regular Variation](./incerto-regular-variation.md).
- Related catalog: Tail Class Catalog (planned).
- Related: [Max-to-Sum Ratio](./incerto-max-to-sum-ratio.md).
- Related to: LLN Failure Under Infinite Mean (planned).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/theorems/subexponentiality.md`, revision `9717c9c`
(2026-09-13 Batch 2 import). Copyright (c) 2023 xshi19. Licensed under MIT.
Links, notation, and qualifications were adapted for this site; executable
figures and simulations were replaced with static calculations. No upstream
execution or formal-proof verification is claimed for this adaptation.

:::{dropdown} MIT permission notice

MIT License

Copyright (c) 2023 xshi19

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
:::
