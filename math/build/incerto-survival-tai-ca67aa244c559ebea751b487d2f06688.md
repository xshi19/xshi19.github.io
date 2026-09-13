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

For a right-tail survival function $\bar F(x)=\mathbb P(X>x)$, the survival
tail-ratio diagnostic is

$$
Q_F(x)=\frac{\bar F(x)^2}{\bar F(2x)},
$$

where $\bar F(2x)>0$.  With iid copies $X_1$ and $X_2$, the numerator is

$$
\bar F(x)^2=\mathbb P(X_1>x,\ X_2>x).
$$

Thus $Q_F(x)$ compares a two-coordinate moderate-extreme event with a
one-coordinate double-threshold event.  It is a tail-geometry diagnostic, not
an estimator by itself.

Here $x>0$ is the comparison threshold, $F$ is the CDF, and
$\alpha>0$, $x_m>0$ are the Pareto exponent and lower cutoff.
$Q_F$ is distinct from the regular-variation ratio $\bar F(tx)/\bar F(x)$.

## Calibration cases

The diagnostic separates three useful geometries:

| Tail model | $Q_F(x)$ behavior | Reading |
| --- | --- | --- |
| Pareto, $\bar F(x)=(x_m/x)^\alpha$ | $2^\alpha(x_m/x)^\alpha\to0$ | A single doubled observation is eventually more likely than two independent observations above $x$. |
| Exponential, $\bar F(x)=e^{-\lambda x}$ | $1$ | Two $x$-exceedances and one $2x$-exceedance have the same exponential cost. |
| Standard normal | $\sim \sqrt{2/\pi}\,e^{x^2}/x\to\infty$ | Two moderate extremes are much more likely than one doubled extreme. |

The Pareto calculation is exact once $x\ge x_m$.  The normal calculation uses
Mills' ratio, $\bar\Phi(x)\sim e^{-x^2/2}/(\sqrt{2\pi}x)$, for the
standard Gaussian tail. This also follows from the complementary-error
function expansion in [NIST DLMF, Section 7.12](https://dlmf.nist.gov/7.12).

The [regular variation](./incerto-regular-variation.md) connection is simple.
If $\bar F\in RV_{-\alpha}$ with $\alpha>0$, then

$$
\frac{\bar F(2x)}{\bar F(x)}\to 2^{-\alpha}.
$$

Therefore

$$
Q_F(x)
=
\frac{\bar F(x)}{\bar F(2x)/\bar F(x)}
\sim
2^\alpha\bar F(x)
\to0.
$$

So regularly varying tails fall on the one-big-jump side of this diagnostic.
That agrees with [subexponentiality](./incerto-subexponentiality.md), though
$Q_F(x)\to0$ is only a diagnostic ratio and not the full convolution-tail
definition.

## Static calibration

(survival-tail-ratio-diagnostic)=
For an exact Pareto law with $x_m=1$ and $\alpha=1.5$,
$Q_F(x)=(2/x)^{3/2}$ for $x\ge1$. The exponential comparison is exactly one
at every positive threshold.

| Threshold $x$ | Pareto $Q_F(x)$ | Exponential $Q_F(x)$ |
| --- | --- | --- |
| $2$ | $1$ | $1$ |
| $4$ | $2^{-3/2}\approx0.353553$ | $1$ |
| $8$ | $1/8$ | $1$ |
| $16$ | $2^{-9/2}\approx0.044194$ | $1$ |

A heavy-tailed model can therefore have $Q_F(x)\ge1$ at finite thresholds;
the asymptotic direction and the range inspected both matter. For the normal,
substitution of Mills' ratio gives

$$
Q_\Phi(x)\sim
\frac{e^{-x^2}/(2\pi x^2)}{e^{-2x^2}/(2\sqrt{2\pi}x)}
=\sqrt{\frac2\pi}\frac{e^{x^2}}x.
$$

Student-t tails with positive degrees of freedom (including the Cauchy case)
are regularly varying, so the preceding regular-variation argument gives
$Q_F(x)\to0$. These are population calculations, not empirical fits.

## Caveats

- $Q_F(x)$ is a distribution-level diagnostic.  A finite empirical estimate can
  be dominated by sampling noise because both the numerator and denominator
  involve rare events.
- The iid interpretation of $\bar F(x)^2$ fails under dependence.  Clustered
  extremes need their own dependence model before this ratio can be read as a
  joint-event probability.
- The ratio uses right tails.  Two-sided returns, losses, and absolute values
  require an explicit transformation before the diagnostic is applied.
- $Q_F(x)\to0$ is compatible with one-big-jump geometry, but it is not a
  replacement for the convolution-tail definition of
  [Subexponentiality](./incerto-subexponentiality.md).

## References

- Taleb, *Statistical Consequences of Fat Tails* [2020](https://arxiv.org/abs/2001.10488).
- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  (2nd ed., Wiley, 1971).
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).

## Backlinks

- Depends on: [Regular Variation](./incerto-regular-variation.md) and
  [Subexponentiality](./incerto-subexponentiality.md).
- Related catalog: Tail Class Catalog (planned).
- Related diagnostic: [Max-to-Sum Ratio](./incerto-max-to-sum-ratio.md).
- Used by: Iso-Density Tail Geometry (planned).

## Source and adaptation

Adapted from `incerto-wiki`, `content/concepts/methods/survival-tail-ratio.md`, revision `9717c9c`
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
