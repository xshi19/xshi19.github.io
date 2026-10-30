---
title: Karamata's Theorem
options:
  concept:
    id: karamata
    type: theorem
    depends_on:
      - regular-variation
    tags:
      - fat-tails
      - moments
---

## Statement

Let $L$ be eventually positive, measurable, locally bounded, and
[slowly varying](./incerto-regular-variation.md) at infinity.  If $\rho>-1$, then

$$
\int_a^x t^\rho L(t)\,dt
\sim
\frac{x^{\rho+1}L(x)}{\rho+1},
\qquad x\to\infty.
$$

If $\rho<-1$, then

$$
\int_x^\infty t^\rho L(t)\,dt
\sim
\frac{x^{\rho+1}L(x)}{-\rho-1},
\qquad x\to\infty.
$$

Here $a>0$ is fixed in a region where $L$ is locally bounded, and
$A(x)\sim B(x)$ means $A(x)/B(x)\to1$.
Both integral statements are cited from @bingham1987regular,
Theorem 1.5.11, p. 28 (direct half); the special case for $L$ and powers
also appears as Propositions 1.5.8 and 1.5.10, pp. 26–27.

For a nonnegative random variable $X$ with
[regularly varying](./incerto-regular-variation.md) survival function
$\bar F(x)=x^{-\alpha}L(x)$ and $\alpha>0$, integration by parts gives the
moment-test consequences

$$
\mathbb E[X^p\mathbf 1_{\{X>x\}}]
\sim
\frac{\alpha}{\alpha-p}x^p\bar F(x),
\qquad 0<p<\alpha,
$$

and

$$
\mathbb E[X^p\mathbf 1_{\{X\le x\}}]
\sim
\frac{\alpha}{p-\alpha}x^p\bar F(x),
\qquad p>\alpha.
$$

Thus a [Pareto-type tail](./incerto-pareto.md) has finite $p$th moment
for $0<p<\alpha$ and infinite $p$th moment for $p>\alpha$. The boundary case
$p=\alpha$ is not decided by this theorem alone; it depends on the slowly
varying factor $L$.

Here $\bar F(x)=\mathbb P(X>x)$, $\alpha$ is the positive tail exponent,
and $p$ is the moment order. The integration parameters $a$ and $\rho$
belong to the integral theorem.

## Intuition

Karamata's theorem says that, for regularly varying functions, integrals are
asymptotically governed by the endpoint where the mass accumulates.  When
$\rho>-1$, the integral up to $x$ is controlled by the upper endpoint.  When
$\rho<-1$, the remaining tail integral beyond $x$ is controlled by the lower
endpoint.

This is the bridge between tail shape and
[moment existence](./incerto-pareto-moment-existence.md).  Once the survival tail behaves
like $x^{-\alpha}L(x)$, multiplying by $x^p$ tests whether the $p$th moment is
still dominated by ordinary observations or by the far tail.

## Examples

- For an exact [Pareto tail](./incerto-pareto.md), $L$ is constant, so
  Karamata reduces the moment test to integrating a pure power.
- For $L(x)=\log x$, the same power boundary applies, but the endpoint
  asymptotic gains the slowly varying logarithmic factor.
- The boundary case $p=\alpha$ needs separate analysis.  Exact Pareto tails
  diverge logarithmically, as shown in
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md).  More generally,

  $$
  \mathbb E[X^\alpha]<\infty
  \Longleftrightarrow
  \int^\infty \frac{L(t)}{t}\,dt<\infty,
  $$

  when $\bar F(t)=t^{-\alpha}L(t)$ in the tail.  Thus $L(t)=1$ gives
  logarithmic divergence, $L(t)=1/\log t$ gives $\log\log t$ divergence,
  and $L(t)=1/(\log t)^2$ gives a finite boundary moment.

## Deriving the moment consequences

We use Karamata's integral asymptotics as a cited theorem, then derive the two
moment consequences by integration by parts.  The boundary case $p=\alpha$ is
not decided by Karamata alone; it depends on the slowly varying factor and is
handled through examples and caveats.

The integral asymptotics are the classical form of Karamata's theorem for
regularly varying functions.  The full proof uses uniform convergence of slowly
varying functions on compact multiplier intervals and a split of the integral
into a near-endpoint part and a negligible remainder; this page cites the
standard theorem rather than reproducing that argument here.

The moment consequences follow from integration by parts.  For $0<p<\alpha$,

$$
\mathbb E[X^p\mathbf 1_{\{X>x\}}]
=x^p\bar F(x)+p\int_x^\infty t^{p-1}\bar F(t)\,dt.
$$

Since $t^{p-1}\bar F(t)=t^{p-\alpha-1}L(t)$ and
$p-\alpha-1<-1$, Karamata's tail-integral form gives

$$
\int_x^\infty t^{p-1}\bar F(t)\,dt
\sim
\frac{x^{p-\alpha}L(x)}{\alpha-p}
=
\frac{x^p\bar F(x)}{\alpha-p}.
$$

Therefore

$$
\mathbb E[X^p\mathbf 1_{\{X>x\}}]
\sim
\left(1+\frac{p}{\alpha-p}\right)x^p\bar F(x)
=
\frac{\alpha}{\alpha-p}x^p\bar F(x).
$$

For $p>\alpha$,

$$
\mathbb E[X^p\mathbf 1_{\{X\le x\}}]
=p\int_0^x t^{p-1}\bar F(t)\,dt-x^p\bar F(x).
$$

The part of the integral over any fixed bounded interval is negligible relative
to $x^p\bar F(x)$.  Applying Karamata to
$t^{p-\alpha-1}L(t)$ gives

$$
p\int_0^x t^{p-1}\bar F(t)\,dt
\sim
\frac{p}{p-\alpha}x^p\bar F(x),
$$

so subtracting $x^p\bar F(x)$ leaves

$$
\mathbb E[X^p\mathbf 1_{\{X\le x\}}]
\sim
\frac{\alpha}{p-\alpha}x^p\bar F(x).
$$

For the exact Pareto special case, $L$ is constant and the same formulas are
obtained by direct power integration.

## Moment-test calculation

(karamata-moment-test-computation)=
For a slowly varying example take $L(t)=\log t$ on $t>1$ and
$\rho>-1$. Integration by parts gives the exact identity

$$
\int_1^x t^\rho\log t\,dt
=\frac{x^{\rho+1}\log x}{\rho+1}
-\frac{x^{\rho+1}-1}{(\rho+1)^2}.
$$

Dividing by Karamata's leading term yields

$$
\frac{\int_1^x t^\rho\log t\,dt}
     {x^{\rho+1}\log x/(\rho+1)}
=1-\frac{1-x^{-(\rho+1)}}{(\rho+1)\log x}\longrightarrow1.
$$

For exact Pareto with $x_m=1$ and $p>\alpha$, the ratio of the truncated
moment to its leading term is $1-x^{-(p-\alpha)}\to1$.
At $p=\alpha$, the truncated moment is $\alpha\log x$ instead.
These are exact calculations illustrating the cited theorem.

(karamata-conditional-mean)=
## Conditional size and the boundary moment

For $\alpha>1$, write $e(u)=\mathbb E[X-u\mid X>u]$ for the mean excess.
Set $p=1$ in the proved tail-moment consequence and divide by
$\bar F(u)>0$. It gives

$$
\frac{\mathbb E[X\mid X>u]}u\longrightarrow\frac{\alpha}{\alpha-1},
\qquad
\frac{e(u)}u\longrightarrow\frac1{\alpha-1}.
$$

This is a consequence under a regular-variation assumption, not a finite
threshold criterion for fitting that class. It gives the precise forward
implication relevant to SCoFT v4 §5.0.2, pp. 93–94 [@taleb2025scoftv4].
The [Pareto note](incerto-pareto.md#pareto-threshold-quantities) owns the
exact version. The chapter guide gives a subexponential Weibull example
whose conditional-mean ratio tends to one.

For a complete example with a finite boundary moment, take the survival
equal to one below $e$ and, for $x\ge e$, set
$\bar F(x)=e^\alpha x^{-\alpha}/(\log x)^2$.
It is continuous at $e$ and strictly decreasing afterwards. Tonelli gives

$$
\mathbb EX^\alpha
=\alpha\int_0^\infty t^{\alpha-1}\bar F(t)\,dt
=e^\alpha+\alpha e^\alpha\int_e^\infty\frac{dt}{t(\log t)^2}
=(1+\alpha)e^\alpha<\infty.
$$

This proves why the exact-Pareto boundary cannot be transferred to every
regularly varying tail. For $\alpha=2$ the answer is $3e^2$, whereas an
exact Pareto of exponent $2$ has infinite second moment.

The [chapter 5 integral figure](incerto-pareto-moment-existence.md#pareto-moments-demo)
compares the logarithmic integral in the earlier moment-test calculation
with its Karamata leading term. The demo also evaluates the
boundary example in logarithmic coordinates, integrating to infinity.

## Caveats

- The assumptions on $L$ matter.  Slow variation is an asymptotic regularity
  condition, not a finite-sample diagnostic.
- Karamata does not settle the boundary moment $p=\alpha$.  Exact Pareto tails
  diverge logarithmically there, while other slowly varying factors can change
  the boundary behavior.
- Moment existence is a tail statement.  Estimating a moment from data also
  depends on sample size, dependence, threshold choice, and whether the fitted
  tail model is credible.
- We treat nonnegative right tails.  Two-sided models need the same
  reasoning applied to $|X|$ or to each side separately.

## References

- @bingham1987regular, *Regular Variation*, Theorem 1.5.11, p. 28;
  Propositions 1.5.8 and 1.5.10, pp. 26–27. These support the cited integral
  asymptotics; the moment and conditional-mean consequences are derived here.
- @taleb2025scoftv4, §§5.0.2 and 5.1.1, pp. 93–94 and 95–97:
  conditional-size and scaling context, not the proof source for Karamata.
- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  (1971, 2nd ed., Wiley).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  @embrechts1997modelling.
- Resnick, *Heavy-Tail Phenomena* @resnick2007heavy.

## Backlinks

- Depends on: [Regular Variation](./incerto-regular-variation.md) and the canonical tail
  notation in [Shared notation](https://xshi19.github.io/math/notation/).
- Used by: [Pareto Distribution](./incerto-pareto.md),
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md),
  [LLN Failure Under Infinite Mean](./incerto-lln-failure.md),
  [Max-to-Sum Ratio](./incerto-max-to-sum-ratio.md), and
  [Hill Estimator](./incerto-hill-estimator.md).
