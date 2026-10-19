---
title: Chapter 5 — Subexponentials and power laws
---

Chapter 5 compares two different tail properties. [Subexponentiality](incerto-subexponentiality.md)
describes a large sum of a fixed number of independent observations.
[Regular variation](incerto-regular-variation.md) describes how survival changes
when we multiply a threshold. The latter implies the former for nonnegative
laws with positive tail exponent, but the converse fails.

We use @taleb2025scoftv4, third edition (2025), arXiv:2001.10488v4,
chapter 5, printed pp. 91–105, and its attached appendix B, pp. 127–133.
The notes develop their own examples and arguments. The source locators below
identify context; each concept page identifies its proved and cited results.

## Reading map

| V4 locator, printed pages | Question | Concept page |
| --- | --- | --- |
| §5.0.1, pp. 91–93 | Which limit makes one summand account for a large sum? | [Subexponentiality](incerto-subexponentiality.md), [tail class catalog](incerto-tail-class-catalog.md) |
| §§5.0.2–5.0.3, pp. 93–95 | What does the squared-survival ratio measure? | [Survival tail ratio](incerto-survival-tail-ratio.md#tail-ratio-borderline): exponential, logistic, Laplace, and Weibull calculations |
| §5.1.1, pp. 95–97 | What survives multiplicative rescaling? | [Pareto](incerto-pareto.md), [regular variation](incerto-regular-variation.md), and [Karamata integrals](incerto-karamata.md) |
| §5.1.2, pp. 97–98 | Does a convenient tail model identify the data-generating process? | [Student versus double Pareto](incerto-double-pareto.md#double-pareto-student-matching); no empirical fit is inferred here |
| §§5.2.1–5.2.2, pp. 98–100 | When does the smallest exponent determine a sum or product? | [Positive sums and independent products](incerto-pareto.md#pareto-sums-products) |
| §5.2.3, p. 100 | How does taking a positive power change the exponent? | [Power transformations](incerto-regular-variation.md#regular-variation-powers), [moment existence](incerto-pareto-moment-existence.md) |
| §5.3, p. 101 | Is matching tails the same as matching variance? | [Double-Pareto comparison](incerto-double-pareto.md#double-pareto-student-matching) |
| §5.4, p. 102 | What does a historical example establish about extrapolation? | [Inference limits](incerto-inference-asymmetry.md); the source's jobless-claims example is not reproduced or independently validated |
| §5.5, pp. 102–103 | Can a law have no positive power moment? | [Log-Pareto construction](incerto-tail-class-catalog.md#tail-catalog-log-pareto) |
| §5.6, pp. 103–105 | Can constant-scale observations produce unstable averages of squares? | [Moment estimation](incerto-moment-estimation.md#moment-estimation-gates); the squared-tail calculation below |
| §B.1, pp. 128–129 | Which constraints select a Gaussian, uniform, or Pareto law? | [Maximum entropy](incerto-maximum-entropy.md#maximum-entropy-shannon) |
| §B.2, pp. 129–133 | How do escort constraints and normalization restrict a q-exponential? | [Tsallis and escort calculations](incerto-maximum-entropy.md#maximum-entropy-tsallis) |

## Keep the three ratios separate

For iid nonnegative $X_1,X_2$ with survival $\bar F(u)=P(X>u)>0$, define
the following local ratios:

$$
A_t(u)=\frac{\bar F(tu)}{\bar F(u)},\qquad
Q_F(u)=\frac{\bar F(u)^2}{\bar F(2u)},\qquad
R(u)=\frac{P(X_1+X_2>u)}{2\bar F(u)}.
$$

Regular variation asks whether $A_t(u)\to t^{-\alpha}$ for every fixed
$t>0$. Subexponentiality asks whether $R(u)\to1$. The event comparison
$Q_F$ alone supplies neither definition. For an exponential law, $Q_F=1$
identically, while $R(u)=(1+\lambda u)/2$ diverges. The
[ratio note](incerto-survival-tail-ratio.md) gives exact calculations and
explains why translating a light-tailed variable changes the limit of $Q_F$.

For a Pareto law, the fixed-size rare-event limit holds for every $\alpha>0$.
It does not say that the maximum accounts for nearly all of an increasingly
long sample. The [existing Pareto regimes](incerto-pareto.md#sums-and-moment-orders)
separate $n\to\infty$ from $u\to\infty$ with $n$ fixed, including the
infinite-mean boundary $\alpha=1$.

## Exponents need support and dependence assumptions

For nonnegative variables and positive fixed weights, a union bound and a
one-coordinate lower bound preserve the smallest **logarithmic** tail
exponent of a finite sum. That statement does not require independence.
Signed variables can cancel: $(V,-V)$ has sum zero even when both marginals
have power tails. The [sum proof](incerto-pareto.md#pareto-sums-products)
states exactly what the bounds establish.

Products are more sensitive to dependence. Two independent unit-cutoff
Pareto variables with common exponent $\alpha$ have product survival

$$
P(X_1X_2>z)=z^{-\alpha}(1+\alpha\log z),\qquad z\ge1.
$$

The [product derivation](incerto-pareto.md#pareto-independent-products)
also handles unequal exponents. With identical observations $X_2=X_1$,
the product is a square and its exponent is $\alpha/2$ instead. Thus the
unqualified product principle in §5.2 cannot be used under arbitrary
dependence.

For a nonnegative variable with regularly varying survival and $r>0$,
$X^r$ has exponent $\alpha/r$. Applying this to $|T_\nu|$, where $T_\nu$
is a Student variable with degrees of freedom $\nu>0$, gives exponent
$\nu/2$ for $T_\nu^2$. Its mean exists for $\nu>2$ and its variance for
$\nu>4$. These moment conditions do not identify a stochastic-volatility
process from a fluctuating sample statistic. Chapter 4 already supplies
the [estimator-moment distinction](incerto-moment-estimation.md#moment-estimation-root).
The special-function expression and mean-deviation table in §5.6 are not
needed for this argument and are not claimed verified here.

## Qualifications to carry into the source

- Exponential and gamma laws fail the convolution-subexponential criterion;
  the [catalog's direct proof](incerto-tail-class-catalog.md#exact-light-tail-checks)
  takes precedence over treating them as members of that class.
- A slowly varying factor need not approach a positive constant. The
  [existing counterexample](incerto-tail-class-catalog.md#tail-class-catalog-slow-variation)
  and the equal-exponent product above retain a logarithmic factor forever.
- Matching a survival exponent, or even a leading survival constant, does
  not equate finite moments. Our double-Pareto density includes the required
  factor $1/2$ on each side; its normalization and variance convention are explicit.
- A limit $\mathbb E[X\mid X>u]/u=1$ does not establish a positive
  exponential moment. For survival $e^{-\sqrt u}$, direct integration gives
  $\int_u^\infty e^{-\sqrt t}\,dt=2(\sqrt u+1)e^{-\sqrt u}$.
  Dividing by survival gives $e(u)=2\sqrt u+2$, so the conditional-mean
  ratio tends to one despite subexponentiality.
- Maximum entropy is relative to a support, reference measure, and specified
  constraints. The appendix's escort-mean calculation requires positive
  normalization and an admissible entropic index; stationarity alone is not
  a proof of a global optimum for a nonlinear constraint.

Notation follows the [shared canon](https://xshi19.github.io/math/notation/).
The source's lower cutoff $L$ maps to $x_m$; $L(x)$ here is a slowly
varying function. We reserve $\alpha>0$ for a survival exponent, use $\nu$
for Student degrees of freedom, and use local $q$ for the entropic index.
The formula $\xi=1/\alpha$ is restricted to the positive regularly varying case.

(scoft-ch5-demo)=
## Reproduce the worked models

Run `uv run python demos/incerto/power_laws.py` after installing the repository
dependencies. It produces four original figures: tail diagnostics, products
and moment integrals, Student/double-Pareto matching, and entropy constraints.
The grids and one-dimensional quadrature are bounded; no random draws,
external data, fitted parameters, censoring, or sampling intervals are used.
The concept-page captions give model parameters and numerical limits.

The [maximum-entropy example](incerto-maximum-entropy.md#maximum-entropy-worked)
keeps $\mathbb E\log(X/x_m)=1/2$ fixed and compares an exact Pareto law
with a log-gamma alternative. The entropy gap is computed independently
from the defining integrals. This checks a worked case of the analytic
proof, not a general numerical search over densities.

## References

- @taleb2025scoftv4, ch. 5, pp. 91–105; appendix B, pp. 127–133, at the
  section locators above. Source context and notation only where qualified.
- The [regular-variation](incerto-regular-variation.md),
  [Karamata](incerto-karamata.md), and [subexponential](incerto-subexponentiality.md)
  notes give theorem-level references for their cited general results.
