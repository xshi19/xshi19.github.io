---
kernelspec:
  name: python3
  display_name: Python 3
---

# Body, Shoulders, and Tails

## Statement

For a symmetric location-scale density

$$
f_\sigma(x)=\frac1\sigma p\left(\frac{x-\mu}{\sigma}\right),
$$

Taleb's body-shoulder-tail diagnostic asks where a small randomization of
$\sigma$ adds density and where it removes density.  A local version is the
sign of

$$
\frac{\partial^2 f_\sigma(x)}{\partial \sigma^2}.
$$

For the standard normal density, this second derivative changes sign at

$$
\pm\sqrt{\frac{5-\sqrt{17}}{2}},
\qquad
\pm\sqrt{\frac{5+\sqrt{17}}{2}}.
$$

These four points split the line into left tail, left shoulder, peak, right
shoulder, and right tail regions.  Positive scale curvature means variance
mixing raises density there; negative scale curvature means it lowers density
there.

## Intuition

If a Gaussian scale is randomized while keeping the center fixed, the resulting
mixture does not simply "spread out" everywhere.  It tends to add mass near the
peak and in the far tails, while taking mass from the shoulders.  The shoulders
are the moderate-deviation region that looks ordinary under a single scale but
is depleted when scale uncertainty is introduced.

This is a diagnostic for finite-sample geometry, not a tail-index estimator.
It explains why stochastic volatility can create both a sharper center and
fatter-looking tails without producing a Pareto tail.

## Proof

For the standard normal density $\phi$, write

$$
f_\sigma(x)=\frac1\sigma\phi\left(\frac{x}{\sigma}\right).
$$

Set $y=x/\sigma$.  A direct differentiation gives

$$
\frac{\partial f_\sigma(x)}{\partial \sigma}
=
\frac{f_\sigma(x)}{\sigma}(y^2-1),
$$

and a second differentiation gives

$$
\frac{\partial^2 f_\sigma(x)}{\partial \sigma^2}
=
\frac{f_\sigma(x)}{\sigma^2}(y^4-5y^2+2).
$$

At $\sigma=1$, the sign changes where

$$
x^4-5x^2+2=0.
$$

Solving the quadratic in $x^2$ gives

$$
x^2=\frac{5\pm\sqrt{17}}2.
$$

For the Student-t density, the same calculation produces different boundaries
that depend on the degrees of freedom.  The page's Python example uses the
closed form migrated from the Chapter 4 notebook.

## Python

The helper functions return the four boundaries in increasing order.

```{code-cell} python
:label: body-shoulder-tail-python-check

import numpy as np

from incerto.stats import normal_peak_shoulder_tail, t_peak_shoulder_tail

normal_bounds = np.round(normal_peak_shoulder_tail(), 3)
student_bounds = np.round(t_peak_shoulder_tail(df=3), 3)

print(f"Normal scale-curvature boundaries: {normal_bounds}")
print(f"Student-t(df=3) scale-curvature boundaries: {student_bounds}")
```

The printed boundaries show that a Student-t distribution with three degrees of
freedom has wider tail boundaries but a similar inner shoulder boundary.

## Caveats

- The diagnostic assumes a symmetric location-scale family.  Skewed or
  multimodal distributions need a different interpretation.
- The region labels are not universal definitions of "body" or "tail".  They
  are tied to scale perturbations.
- A shoulder/tail boundary is not a threshold for Pareto estimation.  It
  describes density geometry, not regular variation.
- The finite-difference version used in exploratory plots depends on the size
  of the scale perturbation.  The formulas here are the local limit.

## References

- Taleb, *Statistical Consequences of Fat Tails*, Chapter 4
  [@taleb2020scoft].

## Backlinks

- Depends on: [Normal Variance Mixture](../distributions/normal-mixture.md)
  and the canonical density notation in [Notation](../../notation/index.md).
- Used by: the planned Chapter 4 reading guide and
  [Dispersion Ratio Under Fat Tails](../examples/dispersion-ratio.md).

<!-- incerto-provenance:start -->
<footer class="incerto-provenance">
<p><strong>Provenance.</strong> Source: <code>content/concepts/methods/body-shoulder-tail.md</code>. Last verified: 2026-05-24. Checked against cited sources, page proof or computation, and executable examples.</p>
</footer>
<!-- incerto-provenance:end -->
