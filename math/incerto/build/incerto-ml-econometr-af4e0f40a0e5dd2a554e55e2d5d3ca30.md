---
title: Tail assumptions in statistical pipelines
options:
  concept:
    id: ml-econometrics
    type: method
    depends_on:
      - moment-estimation
      - hidden-tail
      - sp500
    tags:
      - estimation
      - extrapolation
---

A risk estimator needs a target, a sampling model, and an account of what
happens outside the observed range. Parametric extrapolation supplies an
assumption about that range. An empirical distribution puts no mass beyond
its largest observation. Neither choice settles whether its risk estimate
will be useful on the next sample.

SCoFT v4 Appendix F, §§F.1–F.2, printed pp. 208–212, discusses parametric
and nonparametric risk estimation. Appendix G, pp. 213–216, discusses
bounded and unbounded basis functions and calibration by angles
[@taleb2025scoftv4]. We develop limited mathematical statements and a
specified evaluation procedure. They do not imply a blanket failure of
econometrics or machine learning.

(pipeline-moments)=
## Which moment does the estimator need?

For iid $X_i$ with $\mathbb E[X^2]<\infty$, the raw second-moment estimator
is consistent by the strong law. If also $\mathbb E[X^4]<\infty$, then

$$
\begin{gathered}
A_n=\frac1n\sum_{i=1}^n X_i^2,\\
\operatorname{Var}(A_n)
=\frac{\operatorname{Var}(X^2)}n,\\
\operatorname{Var}(X^2)
=\mathbb E[X^4]-(\mathbb E[X^2])^2.
\end{gathered}
$$

The variance formula follows by independence and expansion of the square.
If the second moment is finite but the fourth is infinite, $A_n$ still
converges almost surely, while its sampling variance is infinite: its
nonnegative square dominates $X_1^4/n^2$. Thus consistency and finite
sampling variance are different requirements. This concerns a raw second
moment; estimating an unknown mean and then a central variance introduces
another step. The [moment-estimation note](incerto-moment-estimation.md)
treats the broader distinction.

Moment requirements depend on the estimator and loss. A sample fraction
for a fixed event averages bounded indicators. A sample mean of an
unbounded loss has a different integrability requirement. Likewise a
finite population covariance can coexist with difficult finite-sample
covariance estimation. High sample kurtosis alone cannot decide that all
methods using covariance are invalid.

```{figure} ../assets/figures/incerto/pipeline-contamination.svg
:name: pipeline-contamination-figure
:alt: A Gaussian fitted loss cutoff grows with one replacement loss, while the empirical 95th percentile stays level and the mean loss above it rises.

Controlled contamination: 255 fixed normal observations (PCG64 seed 1012)
plus one loss varied from 2 to 40 in normal scale units. The Gaussian cutoff
uses the sample mean and standard deviation; the empirical cutoff is the
95th percentile with the higher-order-statistic convention. Its flat segment
coexists with increasing tail severity. This deterministic sweep illustrates
these estimators on one sample; it is neither observed market evidence nor
an estimator performance theorem.
```

(pipeline-unseen)=
## What a nonparametric fit leaves unspecified

Let $P$ be a loss distribution supported on $[0,b]$, and let $\delta_B$
put all mass at $B>b$. For $0<\varepsilon<1$, consider

$$
\begin{gathered}
Q_B=(1-\varepsilon)P+\varepsilon\delta_B,\\
\sup_x|F_{Q_B}(x)-F_P(x)|\le\varepsilon,\\
\mathbb E_{Q_B}[X]-\mathbb E_P[X]\\
=\varepsilon(B-\mathbb E_P[X]).
\end{gathered}
$$

The CDF bound follows by writing the mixture CDF and using that both CDFs
lie in $[0,1]$. Linearity of expectation proves the last identity. For any
fixed $\varepsilon$, the mean difference grows without bound as $B$ grows,
although the CDF distance stays small. In $n$ iid mixture draws the
probability of no contamination is $(1-\varepsilon)^n$.

This proves a limitation without a bound on the unseen loss, not an
impossibility theorem for every nonparametric estimand. Bounded-payoff error
has different controls. An ordinary empirical bootstrap can resample only
observed values; additional tail modeling supplies additional assumptions.
See [hidden tails](incerto-hidden-tail.md) and
[plug-in tail estimation](incerto-plug-in-tail-estimation.md).

(pipeline-holdout)=
## Freeze the pipeline before the test period

The local S&P procedure uses the first half of chronological log returns
for fitting and the second half for evaluation. Loss is $\ell_t=-r_t$.
We compare a Gaussian 95% cutoff, fitted with the training mean and
standard deviation, with the training empirical 95th percentile. Both
are frozen before the first test observation.

Report strict exceedance counts, the test sample size, and the mean loss
among those exceedances. They measure frequency and severity separately.
For $n$ sorted training values and probability $q$, the empirical quantile
selects zero-based index $\lceil(n-1)q\rceil$ (`higher` interpolation).
Ties may put more mass at the cutoff. Neither cutoff is promised to yield exactly 5% test
exceedances. No binomial error bars are attached to dependent returns.

Use [`chronological_tail_check`](incerto-api-reference.md#api-chronological-tail-check).

:::{dropdown} Code: a fixed training set and two test periods

```python
import numpy as np
from xmath.incerto.diagnostics import chronological_tail_check

train = np.array([-0.03, -0.01, 0, 0.01, 0.03])
for label, test in (("quiet", [0, -0.01, 0.01]), ("stress", [-0.08, 0, -0.12])):
    result = chronological_tail_check(np.r_[train, test], train_size=5)
    for method in ("gaussian", "empirical"):
        row = result[method]
        print(f"{label} / {method}: cutoff={row['threshold']:.4f}, hits={row['exceedances']}, severity={row['mean_exceedance_loss']:.4f}")
```

**Output**

```text
quiet / gaussian: cutoff=0.0329, hits=0, severity=nan
quiet / empirical: cutoff=0.0300, hits=0, severity=nan
stress / gaussian: cutoff=0.0329, hits=2, severity=0.1000
stress / empirical: cutoff=0.0300, hits=2, severity=0.1000
```
:::

These are invented returns. The missing-severity value is exposed rather
than replaced by zero. The unchanged cutoffs demonstrate the frozen fit;
the stress period changes the realized hits and losses.

The [S&P diagnostics pipeline](incerto-sp500.md#sp500-local-run) applies
exactly this procedure to a local authorized cache. It reports dates and
uses all returns, with no training/test overlap in return observations.
Its overlapping aggregation comparison is descriptive and does not enter
the fit. No empirical S&P holdout result is asserted without that cache.
Choosing the probability level after inspecting test results, or repeatedly
trying windows and publishing only a favorable one, would invalidate the
interpretation as a single prespecified comparison. Rolling retraining,
dependence-aware inference, and competing conditional-volatility models
would require separately specified experiments.

(pipeline-hinges)=
## What finite basis expansions can extrapolate

A finite sum of bounded sigmoid functions with finite coefficients is
bounded. Indeed, if $|b_j(x)|\le1$, then

$$
\left|\sum_{j=1}^m w_j b_j(x)\right|
\le\sum_{j=1}^m|w_j|.
$$

It cannot equal an unbounded target on the whole real line. This says
nothing against approximation on a compact interval, where the target
may itself be bounded. Adding an unbounded linear term also changes the
class of functions.

A hinge $(x-K)_+=\max(x-K,0)$ has a linear right tail. We can represent
every continuous piecewise-linear function with finitely many knots
exactly by an affine term and a finite sum of hinges.
Let $K_1<\cdots<K_m$ be the knots, $s_0,\ldots,s_m$ the successive
slopes, and $a=f(K_1)$. Define slope changes $w_j=s_j-s_{j-1}$. Then

$$
\begin{gathered}
f(x)=a+s_0(x-K_1)\\
\quad+\sum_{j=1}^m w_j(x-K_j)_+.
\end{gathered}
$$

To prove the representation, differentiate away from the knots. On
segment $j$ the slope is $s_0+\sum_{i=1}^j w_i=s_j$ by telescoping.
The formula is continuous at every knot and equals $a$ at $K_1$.
Those slopes and one anchor value determine the specified function.

Finite hinge sums are piecewise linear, not exact representations of
arbitrary nonlinear functions. Even for this class, a chosen slope outside
the observed range remains an extrapolation assumption. Linearity outside
the knots can help when the target has linear tails, and fail when it does
not. A function's extrapolation and the input distribution's tail are
separate objects.

(pipeline-angles)=
## Angles determine slopes; slope changes determine weights

With equal axis units, a segment at angle $\vartheta_j$ has slope
$s_j=\tan\vartheta_j$. Its angle is $\arctan s_j$. Thus a hinge weight is

$$
w_j=\tan\vartheta_j-\tan\vartheta_{j-1}.
$$

It is not the arctangent of an angle. If display coordinates multiply
horizontal and vertical data by $a_x>0$ and $a_y>0$, the data slope is
$(a_x/a_y)\tan\vartheta_j$. Measuring a screenshot angle without its
axis scales cannot recover the coefficient. The local symbol $\vartheta$
avoids reusing the canon's model parameter $\theta$.

Use [`hinge_payoff`](incerto-api-reference.md#api-hinge-payoff).

```python
import numpy as np
from xmath.incerto.diagnostics import hinge_payoff

slopes = np.array([0.5, 1.5, -0.5])
print("Angles (degrees): " + ", ".join(f"{v:.2f}" for v in np.degrees(np.arctan(slopes))))
print("Hinge weights: " + ", ".join(f"{v:.2f}" for v in np.diff(slopes)))
values = hinge_payoff([-1, 0, 1, 2, 3], [0, 2], slopes)
print("Values: " + ", ".join(f"{v:.2f}" for v in values))
```

**Output**

```text
Angles (degrees): 26.57, 56.31, -26.57
Hinge weights: 1.00, -2.00
Values: -0.50, 0.00, 1.50, 3.00, 2.50
```

Smooth a hinge with $h_a(z)=a^{-1}\log(1+e^{az})$, $a>0$. Factoring
out $e^{az}$ when $z\ge0$, and using the original expression when $z<0$,
proves the uniform bound

$$
0\le h_a(z)-z_+\le\frac{\log2}a.
$$

For a weighted hinge sum with the same affine term, the triangle
inequality therefore bounds the smoothing error by
$a^{-1}\log2\sum_j|w_j|$. Smoothing retains linear asymptotic tails;
it does not establish the correct slope for an unseen outcome.

```{figure} ../assets/figures/incerto/hinge-calibration.svg
:alt: A continuous piecewise-linear function has slopes 0.5, 1.5 and minus 0.5; another panel compares a hinge with two softplus smoothings.

Exact model curves: knots at 0 and 2 give hinge weights 1 and −2.
The lower panel uses smoothing parameters 1 and 4. Angles are not read
from the rendered figure: display aspect ratios need not preserve data slopes.
```

The figures are regenerated by
`uv run python demos/incerto/sp500_diagnostics.py`. The
[chapter guide](incerto-scoft-ch10.md) maps the arguments to v4 locators.
The moment, contamination, hinge, angle, and smoothing statements above
have explicit arguments; the controlled examples are numerical
illustrations. No universal superiority of an activation function or
risk estimator follows from them.
