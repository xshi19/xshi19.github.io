---
title: Survival Tail Ratio
options:
  concept:
    id: survival-tail-ratio
    type: method
    depends_on:
      - regular-variation
      - subexponentiality
    related:
      - max-to-sum-ratio
    tags:
      - diagnostics
      - fat-tails
      - tail-risk
---

## Statement

For a right-tail survival function $\bar F(u)=\mathbb P(X>u)$, the survival
tail-ratio diagnostic is

$$
Q_F(u)=\frac{\bar F(u)^2}{\bar F(2u)},
$$

where $\bar F(2u)>0$.  With iid copies $X_1$ and $X_2$, the numerator is

$$
\bar F(u)^2=\mathbb P(X_1>u,\ X_2>u).
$$

Thus $Q_F(u)$ compares a two-coordinate moderate-extreme event with a
one-coordinate double-threshold event.  It is a tail-geometry diagnostic, not
an estimator by itself.

Here $u>0$ is the comparison threshold, $F$ is the CDF, and
$\alpha>0$, $x_m>0$ are the Pareto exponent and lower cutoff.
$Q_F$ is distinct from the regular-variation ratio $\bar F(tu)/\bar F(u)$
for a fixed multiplier $t>0$. All limits below send $u\to\infty$.

## Calibration cases

The diagnostic separates three useful geometries:

| Tail model | $Q_F(u)$ behavior | Reading |
| --- | --- | --- |
| Pareto, $\bar F(u)=(x_m/u)^\alpha$ | $2^\alpha(x_m/u)^\alpha\to0$ | A single doubled observation is eventually more likely than two independent observations above $u$. |
| Exponential, $\bar F(u)=e^{-\lambda u}$, $\lambda>0$ | $1$ | Two $u$-exceedances and one $2u$-exceedance have the same exponential cost. |
| Standard normal | $\sim \sqrt{2/\pi}\,e^{u^2}/u\to\infty$ | Two moderate extremes are much more likely than one doubled extreme. |

The Pareto calculation is exact once $u\ge x_m$.  The normal calculation uses
Mills' ratio, $\bar\Phi(u)\sim e^{-u^2/2}/(\sqrt{2\pi}u)$, for the
standard Gaussian tail. This also follows from the complementary-error
function expansion in [NIST DLMF, Section 7.12](https://dlmf.nist.gov/7.12).

The [regular variation](./incerto-regular-variation.md) connection is simple.
If $\bar F\in RV_{-\alpha}$ with $\alpha>0$, then

$$
\frac{\bar F(2u)}{\bar F(u)}\to 2^{-\alpha}.
$$

Therefore

$$
Q_F(u)
=
\frac{\bar F(u)}{\bar F(2u)/\bar F(u)}
\sim
2^\alpha\bar F(u)
\to0.
$$

So regularly varying tails fall on the one-big-jump side of this diagnostic.
That agrees with [subexponentiality](./incerto-subexponentiality.md), though
$Q_F(u)\to0$ is only a diagnostic ratio and not the full convolution-tail
definition.

## Static calibration

(survival-tail-ratio-diagnostic)=
For an exact Pareto law with $x_m=1$ and $\alpha=1.5$,
$Q_F(u)=(2/u)^{3/2}$ for $u\ge1$. The exponential comparison is exactly one
at every positive threshold.

| Threshold $u$ | Pareto $Q_F(u)$ | Exponential $Q_F(u)$ |
| --- | --- | --- |
| $2$ | $1$ | $1$ |
| $4$ | $2^{-3/2}\approx0.353553$ | $1$ |
| $8$ | $1/8$ | $1$ |
| $16$ | $2^{-9/2}\approx0.044194$ | $1$ |

A heavy-tailed model can therefore have $Q_F(u)\ge1$ at finite thresholds;
the asymptotic direction and the range inspected both matter. For the normal,
substitution of Mills' ratio gives

$$
Q_\Phi(u)\sim
\frac{e^{-u^2}/(2\pi u^2)}{e^{-2u^2}/(2\sqrt{2\pi}u)}
=\sqrt{\frac2\pi}\frac{e^{u^2}}u.
$$

Student-t tails with positive degrees of freedom (including the Cauchy case)
are regularly varying, so the preceding regular-variation argument gives
$Q_F(u)\to0$. These are population calculations, not empirical fits.

(tail-ratio-borderline)=
## What a borderline ratio does and does not establish

SCoFT v4 §§5.0.2–5.0.3, pp. 93–95 [@taleb2025scoftv4], motivates
this event comparison with exponential and sigmoid survivals. We derive
their values here and keep them separate from a convolution-tail criterion.

For centered logistic survival $\bar F(u)=1/(1+e^{2ku})$, $k>0$,

$$
Q_F(u)=\frac{1+e^{4ku}}{(1+e^{2ku})^2}\longrightarrow1.
$$

This law is light-tailed. Its density at $x\in\mathbb R$ is bounded by a constant times
$e^{-2k|x|}$, so its MGF is finite for $|\theta|<2k$ by comparison
with an exponential integral on each half-line. The density is
$\tfrac{k}{2}\operatorname{sech}^2(kx)$, the [`hypsecant2`](incerto-api-reference.md#api-hypsecant2) helper's law
with `loc=0, scale=1/k`.
The name identifies this squared-secant convention, not the ordinary
hyperbolic-secant distribution.

For a centered Laplace law with scale $b>0$, the right survival is
$\tfrac12e^{-u/b}$ on $u\ge0$, giving $Q_F(u)=1/2$.
More generally, $\bar F(u)\sim c e^{-\lambda u}$ with $c,\lambda>0$
implies $Q_F(u)\to c$.
Translating $X$ by $d$ changes that constant to $ce^{\lambda d}$.
Thus the value one is not an invariant boundary between light and heavy
tails. It reflects the tail prefactor as well as its decay.

For Weibull survival $e^{-u^\beta}$ on $u\ge0$, with shape $\beta>0$,

$$
\log Q_F(u)=(2^\beta-2)u^\beta.
$$

It tends to $-\infty$, zero, or $+\infty$ as $\beta<1$, $\beta=1$,
or $\beta>1$. The [catalog](incerto-tail-class-catalog.md) supplies the
separate subexponential classification for the first range. A vanishing
$Q_F$ is not a proof of the full convolution condition.

```{figure} ../assets/figures/incerto/power-law-diagnostics.svg
:alt: Three vertical panels show a fully ticked log squared-survival ratio, a linear detail separating exponential zero from negative logistic values, and the iid two-sum ratio.

Top and middle: exact $\log Q_F$ at 241 thresholds between $1$ and $16$ for
Pareto $(3/2,1)$, Weibull shape $1/2$, rate-one exponential, centered
logistic with $k=1/2$, and standard normal. The top uses a symmetric-log
axis with explicit negative and positive ticks. The middle separates the
two curves near zero on a linear axis: exponential is exactly zero, while
logistic approaches zero from below. Bottom: numerical convolution
ratios at 81 thresholds from $3$ to $10^4$ for Pareto $(3/2,1)$,
Weibull shape $1/2$, and rate-one exponential. The finite plots illustrate
different questions; only the convolution ratio defines subexponentiality.
```

Use [`pareto_type1`](incerto-api-reference.md#api-pareto-type1), [`iid_two_sum_tail`](incerto-api-reference.md#api-iid-two-sum-tail).

```python
import numpy as np
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.tail_diagnostics import iid_two_sum_tail

law = pareto_type1(1.5)
u = np.array([3., 10., 100.])
ratio = np.exp(2 * law.logsf(u) - law.logsf(2 * u))
print("Squared-survival ratio: [" + ", ".join(f"{v:.4g}" for v in ratio) + "]")
result = iid_two_sum_tail(law, u)
assert result["success"].all()
print("Two-sum ratio: [" + ", ".join(f"{v:.4g}" for v in result["ratio"]) + "]")
```

**Output**

```text
Squared-survival ratio: [0.5443, 0.08944, 0.002828]
Two-sum ratio: [1.769, 1.391, 1.044]
```

The convolution ratio uses independent summands.

Figure regeneration: `uv run python demos/incerto/power_laws.py`. Log survival values prevent
division of underflowed probabilities. The convolution helper uses
one-dimensional adaptive quadrature; the demo stops on a failed error check.
No random sampling or empirical fitting is involved.

## Caveats

- $Q_F(u)$ is a distribution-level diagnostic.  A finite empirical estimate can
  be dominated by sampling noise because both the numerator and denominator
  involve rare events.
- The iid interpretation of $\bar F(u)^2$ fails under dependence.  Clustered
  extremes need their own dependence model before this ratio can be read as a
  joint-event probability.
- The ratio uses right tails.  Two-sided returns, losses, and absolute values
  require an explicit transformation before the diagnostic is applied.
- $Q_F(u)\to0$ is compatible with one-big-jump geometry, but it is not a
  replacement for the convolution-tail definition of
  [Subexponentiality](./incerto-subexponentiality.md).

## References

- @taleb2025scoftv4, §§5.0.2–5.0.3, pp. 93–95: event-ratio context.
  The limiting prefactor and translation calculations are proved here.
- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  (2nd ed., Wiley, 1971).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  @embrechts1997modelling.

## Backlinks

- Depends on: [Regular Variation](./incerto-regular-variation.md) and
  [Subexponentiality](./incerto-subexponentiality.md).
- Related catalog: [Tail Class Catalog](./incerto-tail-class-catalog.md).
- Related diagnostic: [Max-to-Sum Ratio](./incerto-max-to-sum-ratio.md).
- Used by: [Iso-Density Tail Geometry](./incerto-iso-density-tail-geometry.md).
