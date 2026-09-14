---
title: Generalized Central Limit Theorem
options:
  concept:
    id: generalized-central-limit-theorem
    type: theorem
    depends_on:
      - regular-variation
      - pareto-moment-existence
    proof_depends_on: [pareto]
    tags:
      - stable-laws
      - sums
      - infinite-variance
---

(generalized-clt-statement)=
## Statement

The ordinary central limit theorem uses finite variance and normalizes sums by
$\sqrt{n}$.  The substantive generalized central limit theorem is a
classification theorem: a nondegenerate distribution can occur as a weak limit
of centered and normalized iid sums if and only if it is stable.  In other
words, if there are constants $a_n>0$ and $b_n$ such that

$$
\frac{X_1+\cdots+X_n-b_n}{a_n}
\Rightarrow Z,
$$

for a nondegenerate limit $Z$, then $Z$ must be stable; conversely, stable laws
arise as such limits for suitable iid summands.  The Gaussian law is the stable
case $\alpha=2$.  The heavy-tailed stable cases have index $0<\alpha<2$.

For a two-sided regularly varying sufficient condition, assume

$$
\mathbb P(|X|>x)=x^{-\alpha}L(x),\qquad 0<\alpha<2,
$$

where $L$ is slowly varying at infinity, with tail balance

$$
\frac{\mathbb P(X>x)}{\mathbb P(|X|>x)}\to p,\qquad
\frac{\mathbb P(X<-x)}{\mathbb P(|X|>x)}\to q,\qquad p+q=1.
$$

Choose $a_n$ so that $n\mathbb P(|X|>a_n)\to1$.  Then, with suitable
centering constants $b_n$,

$$
\frac{X_1+\cdots+X_n-b_n}{a_n}
\Rightarrow Z_\alpha,
$$

where $Z_\alpha$ is an $\alpha$-stable law whose skewness is determined by
$p-q$.  For nonnegative Pareto-type examples this reduces to a strongly
right-skewed stable limit, and the scaling is of order $n^{1/\alpha}$ up to a
slowly varying factor.  When $1<\alpha<2$, the mean exists but the variance is
infinite, so a common centering choice is $b_n=n\mathbb E[X]$.

A common centering summary is

$$
b_n=
\begin{cases}
0, & 0<\alpha<1\quad\text{in the usual nonnegative case},\\
n\mathbb E[X\mathbf 1_{\{|X|\le a_n\}}], & \alpha=1\quad\text{as a standard choice},\\
n\mathbb E[X], & 1<\alpha<2.
\end{cases}
$$

Here $S_n=\sum_{i=1}^n X_i$ is the sum, $M_n=\max_{1\le i\le n}X_i$
is the maximum, $\mathbb E$ is expectation, and $\Rightarrow$ denotes
convergence in distribution. The tail exponent $\alpha$, norming constants
$a_n$, centering constants $b_n$, tail-balance weights $p,q$, and stable limit
$Z_\alpha$ are defined above. Shared notation is planned.

We record the stable-limit contrast needed by the site's heavy-tail
pages.  We do not classify every possible domain of attraction or every
parameterization of stable laws.

The classification and sufficient tail condition above are cited from
Feller [1971, 2nd ed., Wiley](https://www.wiley-vch.de/de/fachgebiete/mathematik-und-statistik/an-introduction-to-probability-theory-and-its-applications-volume-2-978-0-471-25709-7) and Resnick [2007](https://doi.org/10.1007/978-0-387-45024-7); the local
argument below addresses only exact Pareto maximum scaling.

## Why $\sqrt{n}$ fails

The normal law is not the only possible attractor for sums.  It is the
finite-variance attractor. For the regularly varying tails with $0<\alpha<2$
considered here, the largest observations remain visible at the scale of the
centered sum, and $\sqrt{n}$ is no longer the right normalization.

For a [Pareto-type tail](./incerto-pareto.md) with exponent
$\alpha<2$, the natural scale of the maximum is about $n^{1/\alpha}$.  Stable
normalization puts the centered sum on that same order. This is why infinite-variance
sums can keep producing large jumps instead of smoothing into Gaussian-looking
noise.

## Scaling argument

The full generalized central limit theorem is a stable-law classification
result, so we cite it rather than reproduce the proof.  What we prove
directly is the exact Pareto maximum scaling calculation, which shows why the
usual $\sqrt n$ scale is too small when $\alpha<2$.

For a [Pareto Type I](./incerto-pareto.md) variable with survival
$\bar F(x)=(x_m/x)^\alpha$, choose $a_n=x_m n^{1/\alpha}$.  Then for $y>0$ and sufficiently large $n$ such that $a_n y\ge x_m$,

$$
\mathbb P(M_n/a_n\le y)
=
\left(1-\frac{1}{ny^\alpha}\right)^n
\to
e^{-y^{-\alpha}}.
$$

The maximum remains of order $a_n$.  Since $a_n$ grows faster than
$\sqrt{n}$ when $\alpha<2$, the Gaussian finite-variance scaling is not the
right asymptotic scale for such tails.  This maximum calculation identifies the
right order of extreme observations; it is a scaling heuristic, not a proof of
stable convergence for sums.

## Pareto scaling comparison

(generalized-clt-scaling-simulation)=
Take exact Pareto Type I with $x_m=1$ and $\alpha=1.5$. Its mean is $3$
and its variance is infinite. With $S_n=\sum_{i=1}^n X_i$, compare

$$
T_n=\frac{S_n-3n}{n^{2/3}},
\qquad
G_n=\frac{S_n-3n}{\sqrt n}=n^{1/6}T_n.
$$

The identity holds for every sample. Consequently, whenever its interquartile
range is nonzero, the spread of $G_n$ is exactly $n^{1/6}$ times that of
$T_n$. Here are the normalization factors, rounded to three decimals:

| Sample size $n$ | Stable scale $n^{2/3}$ | Gaussian scale $\sqrt n$ | Ratio $n^{1/6}$ |
| --- | --- | --- | --- |
| $100$ | $21.544$ | $10$ | $2.154$ |
| $1000$ | $100$ | $31.623$ | $3.162$ |
| $10000$ | $464.159$ | $100$ | $4.642$ |

The cited stable-limit theorem implies convergence of the central quantiles
of $T_n$ to those of a nondegenerate continuous stable law. Thus the central
spread of $G_n$ grows on the $n^{1/6}$ scale. The table evaluates the
normalizations; it is neither simulated evidence nor a proof of stable
convergence.

## Caveats

- Stable-law parameterizations vary across books and software.  We use
  only the stable index $\alpha$ and avoid detailed skew/location notation.
- The displayed regular-variation and tail-balance condition covers the
  non-Gaussian stable range $0<\alpha<2$; the Gaussian boundary needs a
  different criterion.
- Right-tail regular variation alone is not enough for a general two-sided
  variable; the left tail can change the scale or skewness of the limit.
- Some infinite-variance distributions can still be in the Gaussian domain of
  attraction when their truncated second moment is slowly varying.
- For $0<\alpha<1$, the absolute first moment is infinite. At $\alpha=1$,
  its finiteness depends on the slowly varying factor: the tail-integral
  criterion is $\int^\infty L(x)/x\,dx<\infty$. For example, a nonnegative
  variable with survival $1/[x(\log x)^2]$ for $x\ge e$ has a finite mean,
  since that tail integral equals $1$. Exact Pareto tails have infinite mean
  at $\alpha=1$. See the boundary discussion in [Karamata's theorem](./incerto-karamata.md).
  The truncated centering above remains a standard choice at this boundary;
  finite mean alone does not justify replacing it by $n\mathbb E[X]$ without
  checking the resulting location shift on the $a_n$ scale.
- Finite samples can look calmer than the asymptotic theory suggests until a
  large observation arrives.

## References

- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [1971, 2nd ed., Wiley](https://www.wiley-vch.de/de/fachgebiete/mathematik-und-statistik/an-introduction-to-probability-theory-and-its-applications-volume-2-978-0-471-25709-7).
- Resnick, *Heavy-Tail Phenomena* [2007](https://doi.org/10.1007/978-0-387-45024-7).
- Taleb, *Statistical Consequences of Fat Tails* [2020](https://arxiv.org/abs/2001.10488).

## Backlinks

- Depends on: [Regular Variation](./incerto-regular-variation.md) and
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md).
- Related to: [LLN Failure Under Infinite Mean](./incerto-lln-failure.md).
- Used by: Pre-Asymptotic LLN Behavior (planned).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/theorems/generalized-central-limit-theorem.md`, revision `9717c9c`
(2026-09-13 import). Copyright (c) 2023 xshi19. Licensed under MIT.
Links and notation were adapted for this site; executable figures and simulations
were replaced with static calculations. No upstream execution or formal-proof
verification is claimed for this adaptation.

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
