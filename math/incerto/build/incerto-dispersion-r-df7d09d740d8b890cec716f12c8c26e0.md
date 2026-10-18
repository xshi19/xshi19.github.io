---
options:
  concept:
    id: dispersion-ratio
    type: empirical-example
    depends_on:
    - body-shoulder-tail
    tags:
    - robust-scale
    - tail-risk
title: Dispersion Ratio Under Fat Tails
---

## Statement

Let $x_1,\dots,x_n$ be observed data, and compare the sample standard
deviation with absolute-deviation scale estimates.  The repository uses
normal-scaled versions

$$
\operatorname{MeanAD}
=
\sqrt{\frac{\pi}{2}}\frac1n\sum_{i=1}^n |x_i-\bar x|,
$$

and

$$
\operatorname{MedianAD}
=
\frac{\operatorname{median}_i |x_i-\operatorname{median}(x)|}
{\Phi^{-1}(0.75)}.
$$

For Gaussian data, both are calibrated to estimate the same $\sigma$ as the
standard deviation.  Under fat-tailed or contaminated data, the ratios

$$
\frac{\operatorname{Std}}{\operatorname{MeanAD}},
\qquad
\frac{\operatorname{Std}}{\operatorname{MedianAD}}
$$

can rise because sample standard deviation is more sensitive to isolated extremes.
The mean absolute deviation uses all observations; the median absolute
deviation depends on their central order statistics.

The sample notation $X_1,\dots,X_n$ and expectation notation follow the shared
[notation table](https://xshi19.github.io/math/notation/).  The scale symbol $\sigma$ and the
names `MeanAD` and `MedianAD` are local conventions for this page.

## Examples

- For Gaussian data, the standard deviation, normal-scaled mean absolute
  deviation, and normal-scaled median absolute deviation target the same
  $\sigma$, so the ratios should be near $1$.
- For Student-t or Pareto samples, large observations can push the standard
  deviation much higher than the absolute-deviation scales.
- In a single-outlier sample, the median absolute deviation can be zero while
  the standard deviation is large, so the second ratio is recorded as infinite when the numerator is positive.
For a constant sample both numerator and denominator vanish, and the ratio is undefined.

## Calibration

We justify the Gaussian scale constants here; these identities do not
establish a tail class. Integrating the standard normal density on the
positive half-line gives $\mathbb E|Z|=2\int_0^\infty z\phi(z)\,dz=\sqrt{2/\pi}$.
The equation $P(|Z|\le m)=2\Phi(m)-1=1/2$ gives $m=\Phi^{-1}(0.75)$.
Here $Z$ is standard normal and $\phi$ is its density.

For $X\sim N(\mu,\sigma^2)$,

$$
\mathbb E|X-\mu|
=
\sigma\sqrt{\frac{2}{\pi}}.
$$

Thus multiplying the mean absolute deviation by $\sqrt{\pi/2}$ calibrates it
to $\sigma$ under a normal model.  Also,

$$
\operatorname{median}(|X-\mu|)
=
\sigma\Phi^{-1}(0.75),
$$

so dividing the raw median absolute deviation by $\Phi^{-1}(0.75)$ gives the
same normal calibration.

These identities are normal-model calibrations, not tail theorems.  If the
sample has extreme observations, the standard deviation's quadratic loss gives
those observations much more weight than either absolute-deviation statistic.

## Simulation

The simulation compares Gaussian, Student-t, and Pareto samples across sample
sizes.  It reports repeated-simulation medians of the two dispersion ratios.

Run `python demos/incerto/dispersion_ratio.py` after installing `xshi-math[incerto]`
to reproduce these static figures. The script records the sample sizes and
random seeds where simulation is used.

![dispersion ratio diagnostic 1](../assets/incerto/figures/dispersion-ratio-1.svg)


The Gaussian calibration explains why the simulated Gaussian ratios are near one.  The
Student-t and Pareto ratios are larger because the standard deviation is more
sensitive to extremes than the absolute-deviation scales.  The ratio is a
stress signal, not a fitted tail exponent.

## Caveats

- A high ratio is a diagnostic, not a fitted tail exponent.
- The mean absolute deviation and median absolute deviation are scaled here to
  match a Gaussian standard deviation.  Other conventions use unscaled values.
- Median absolute deviation can be too robust for payoff questions where a
  rare outlier dominates the quantity of interest.
- In an infinite-variance sample, the ratios are sample-path objects; the
  population variance does not exist.
- Dependence and volatility clustering can move these ratios even when the
  one-step marginal distribution is unchanged.

## References

- @rousseeuw1993alternatives, "Alternatives to the Median Absolute Deviation".

## Backlinks

- Depends on: [Variance Gamma Distribution](https://xshi19.github.io/math/normix-theory/normix-gh-family-tour/)
  and [Body, Shoulders, and Tails](incerto-body-shoulder-tail.md).
- Used by: robust-scale examples and tail-risk diagnostics.

## Recorded calculation

The committed demo produced the following on 2026-10-17. These are sample
calculations, with seeds and sample sizes specified in the script.

```text
normal; n=200; median Std/MeanAD=0.999; median Std/MedianAD=1.001
normal; n=1000; median Std/MeanAD=1.000; median Std/MedianAD=1.002
normal; n=5000; median Std/MeanAD=1.000; median Std/MedianAD=0.997
normal; n=20000; median Std/MeanAD=1.000; median Std/MedianAD=0.999
student-t(3); n=200; median Std/MeanAD=1.159; median Std/MedianAD=1.421
student-t(3); n=1000; median Std/MeanAD=1.202; median Std/MedianAD=1.464
student-t(3); n=5000; median Std/MeanAD=1.226; median Std/MedianAD=1.500
student-t(3); n=20000; median Std/MeanAD=1.231; median Std/MedianAD=1.497
Pareto(1.5); n=200; median Std/MeanAD=1.819; median Std/MedianAD=6.723
Pareto(1.5); n=1000; median Std/MeanAD=2.457; median Std/MedianAD=9.386
Pareto(1.5); n=5000; median Std/MeanAD=3.086; median Std/MedianAD=12.000
Pareto(1.5); n=20000; median Std/MeanAD=3.974; median Std/MedianAD=16.015
```

## Source and license

Adapted from `xshi19/incerto-wiki`, `content/concepts/examples/dispersion-ratio.md`,
at revision `9717c9cf83aee18b47f423a14bce2f0d14bb8bcb`.
The original author-controlled material is used under MIT; the cited books
and external datasets retain their own rights. Calculations are static and
are not executed by the site build.

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
