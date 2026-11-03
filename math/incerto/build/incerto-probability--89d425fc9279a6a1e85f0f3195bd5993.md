---
title: Tail probabilities, payoffs, and error propagation
options:
  concept:
    id: probability-conflation
    type: method
    prerequisites: [calibration-payoff, point-forecast]
    related: [risk-measures, asymmetric-payoffs, moment-estimation]
    tags: [probability, payoffs, uncertainty]
---

A forecast can accurately describe whether a threshold will be crossed
while leaving the cost of crossing it largely unknown. The
[calibration note](incerto-calibration-payoff.md) defines the event and its
payoffs; the [point-forecast note](incerto-point-forecast.md) explains how
the choice of loss changes the prediction target. Here we prove two
nonidentification results and quantify error propagation in an exact tail
model. They concern information supplied by forecasts, not the abilities
of particular forecasters.

The source context is SCoFT v4, chapter 13, §§13.2–13.3, pp. 253–258,
and §13.6, pp. 259–260 [@taleb2025scoftv4].
The [chapter guide](incerto-scoft-ch13.md) records the section map and
qualifications. All examples below are synthetic.

(conflation-score-gap)=
## A binary score does not identify a payoff mean

Let $X\ge0$ be a loss, $u>0$ a fixed threshold, and
$D=\mathbf1_{\{X>u\}}$. Write $p=P(D=1)$ and let $q\in[0,1]$
be a constant probability report. In the
[shared notation](https://xshi19.github.io/math/notation/), $p=\bar F(u)$.
The Brier loss is $(q-D)^2$. Its excess expected loss above the true
report is

$$
\begin{gathered}
\mathbb E[(q-D)^2]=p(1-p)+(q-p)^2,\\
R_B(q)-R_B(p)=(q-p)^2.
\end{gathered}
$$

Expanding the square and using $D^2=D$ proves both identities.
They measure an error in an event probability. They contain no conditional
severity. In particular, a calibrated report $q=p$ need not have a small
raw Brier score: the term $p(1-p)$ is the outcome's irreducible variation.

We now prove that even the complete iid binary-score sampling law cannot
bound expected excess-payoff error. Fix $0<p<1$ and $u>0$. For each
$\alpha>1$, take an exact Pareto loss with minimum
$x_m=u p^{1/\alpha}$. The existing
[tail-payoff derivation](incerto-calibration-payoff.md#calibration-tail-models)
then gives

$$
\begin{gathered}
P(X>u)=p,\\
C_\alpha(u)=\mathbb E[(X-u)_+],\\
C_\alpha(u)=\frac{up}{\alpha-1}.
\end{gathered}
$$

Every model yields the same Bernoulli law for $D$, hence the same score
law for any fixed report and any iid sample size. Yet $C_\alpha(u)$
can be arbitrarily large as $\alpha\downarrow1$. An amount estimate
based only on $p$ therefore has unbounded possible error across these
models, even with zero Brier regret. All means in this comparison are
finite; an infinite mean is not needed for the counterexample.
With the same fixed $p$ and $u$, the threshold-scaled tail-level mean is

$$
\frac{\mathbb E[X D]}{up}
=1+\frac1{\alpha-1}.
$$

This ratio increases as the exponent decreases toward one. It states
precisely what is held fixed when we say that thicker tails enlarge the
binary/continuous payoff gap. Across arbitrary families with changing
scales, means, or caps, there is no universal ordering by a tail label.

The example uses [`brier_score`](incerto-api-reference.md#api-brier-score)
and [`pareto_tail_bet`](incerto-api-reference.md#api-pareto-tail-bet).
The fixed outcome list illustrates the identical scores; the payoff means
are exact model evaluations, not estimates from that list.

```python
from xmath.incerto.calibration import brier_score, pareto_tail_bet

p, u = .1, 5
outcomes = [0] * 9 + [1]
for alpha in (3, 1.5, 1.05):
    payoff = pareto_tail_bet(alpha, u, xm=u * p**(1 / alpha))
    score = brier_score([p] * 10, outcomes)
    print(f"alpha={alpha:.2f}: Brier={score:.3f}, excess mean={payoff['excess_mean']:.2f}")
```

**Output**

```text
alpha=3.00: Brier=0.090, excess mean=0.25
alpha=1.50: Brier=0.090, excess mean=1.00
alpha=1.05: Brier=0.090, excess mean=10.00
```

(conflation-gap-figure)=
```{figure} ../assets/figures/incerto/conflation-payoff-gap.svg
:alt: The Brier score distribution stays fixed across three Pareto exponents while expected excess loss increases steeply as the exponent approaches one.

Exact models with $u=5$, $p=0.1$ and adjusted Pareto minima.
Top: the distribution of the average of 20 scores at $q=p$ is identical
for all three exponents; its masses are drawn once. Bottom: expected
excess in loss units, over $1.02\le\alpha\le4$, with the three models
marked. The panels use different units; their vertical values are not
comparable scores. No sample or fit enters these curves.
```

(conflation-whole-curve)=
## Even uniformly small probability errors need tail control

One threshold loses information. The entire survival function determines
the mean of a nonnegative loss through Tonelli's theorem:

$$
\mathbb E X=\int_0^\infty\bar F(v)\,dv.
$$

Knowing every probability exactly is therefore sufficient. Approximating
every probability to a small *absolute* error is a different claim.
We prove the distinction with a finite-support sequence.

Let the reference loss be identically zero. For each integer $n\ge2$,
let $X_n=n^2$ with probability $1/n$, and zero otherwise. If $F_0$
and $F_n$ are their CDFs, then

$$
\begin{gathered}
\sup_v|F_n(v)-F_0(v)|=\frac1n,\\
\mathbb E X_n=n.
\end{gathered}
$$

Indeed, the CDF difference equals $1/n$ on $[0,n^2)$ and zero
elsewhere. Multiplying the only positive value by its probability gives
the mean. For every nonnegative threshold below $n^2$, reporting the
reference exceedance probability zero has Brier regret $1/n^2$ under
the $X_n$ model. Thus even uniformly vanishing thresholdwise regret
does not control a mean over this class. Each individual law is bounded;
the class has no common cap or uniform tail-integrability bound.

The relevant extra assumption can be seen directly. For two nonnegative
laws $F,G$ with finite means and any $H>0$, split the mean integrals at $H$:

$$
\begin{aligned}
|\mathbb E_F X-\mathbb E_G X|
&\le H\|F-G\|_\infty\\
&\quad+\mathbb E_F[(X-H)_+]\\
&\quad+\mathbb E_G[(X-H)_+].
\end{aligned}
$$

The first term bounds the integral on $[0,H]$; positivity bounds the
remaining difference by the two tails. If the tail terms tend uniformly
to zero in a model class, uniform CDF convergence does imply convergence
of means. A common finite cap supplies this control immediately.
See [moment estimation](incerto-moment-estimation.md) for why a finite
sample alone does not establish the needed population moments.

Use [`finite_payoff_mean`](incerto-api-reference.md#api-finite-payoff-mean).

```python
from xmath.incerto.lecture_risk import finite_payoff_mean

for n in (10, 100, 1000):
    mean = finite_payoff_mean([1-1/n, 1/n], [0, n*n])
    print(f"n={n}: CDF error={1/n:.3f}, mean={mean:.1f}")
```

**Output**

```text
n=10: CDF error=0.100, mean=10.0
n=100: CDF error=0.010, mean=100.0
n=1000: CDF error=0.001, mean=1000.0
```

(conflation-inverse-probability)=
## A bounded probability can produce an unbounded quantile

For an exact Pareto loss with fixed $x_m>0$ and $\alpha>0$, the
threshold associated with tail probability $p\in(0,1)$ is
$K(p)=x_m p^{-1/\alpha}$. Its derivatives are

$$
\begin{gathered}
K'(p)=-\frac{K(p)}{\alpha p},\\
K''(p)=\frac{\alpha+1}{\alpha^2p^2}K(p)>0,\\
\frac{d\log K}{d\log p}=-\frac1\alpha.
\end{gathered}
$$

These follow by differentiating the power. At the same cutoff and
probability, decreasing $\alpha$ increases both the quantile and its
relative sensitivity. Absolute derivatives diverge near $p=0$ even
when $\alpha>1$. The inverse survival function in this model is
**convex**, not concave. Its curvature sign cannot be inferred from
boundedness of the probability axis.

To model uncertainty, introduce a random report $\Pi\sim\mathrm{Beta}(a,b)$,
with $a,b>0$, and set $K=x_m\Pi^{-1/\alpha}$. This is an assumed law
for reports; the probability integral transform does not imply it for a
fixed event. The transform applies to $F(X)$ for a continuous CDF,
not to the constant $\bar F(u)$ at a fixed threshold.

We derive the positive moments of $K$ directly from the beta density.
For order $r>0$,

$$
\begin{aligned}
\mathbb E K^r
&=\frac{x_m^r}{B(a,b)}\\
&\quad\cdot\int_0^1
  p^{a-1-r/\alpha}(1-p)^{b-1}\,dp.
\end{aligned}
$$

The endpoint at one is integrable because $b>0$. At zero the integral
is finite exactly when $a-r/\alpha>0$. Consequently,

$$
\begin{gathered}
\mathbb E K^r=x_m^r\frac{B(a-r/\alpha,b)}{B(a,b)},\\
r<a\alpha.
\end{gathered}
$$

At and above $r=a\alpha$, the nonnegative moment is infinite;
continuing a gamma-function expression beyond this domain cannot produce
a moment. The source's additional condition involving $a+b$ is redundant
when $b>0$. Changing variables $p=(x_m/k)^\alpha$ also gives

$$
\begin{gathered}
f_K(k)=\frac{\alpha x_m^{a\alpha}}{B(a,b)}
 k^{-a\alpha-1}\\
\qquad\cdot[1-(x_m/k)^\alpha]^{b-1},\\
k>x_m.
\end{gathered}
$$

Since the bracket tends to one, the density has tail power
$a\alpha+1$ and the survival has exponent $a\alpha$.
The report $\Pi$ has every positive moment, while $K$ may have neither
a finite mean nor a finite second moment. If $1<a\alpha\le2$, the
mean exists but the variance is infinite. If $a\alpha\le1$, ordinary
variance about a finite mean is undefined.

Use [`beta_pareto_quantile_moment`](incerto-api-reference.md#api-beta-pareto-quantile-moment).
For $a=2,b=3$, the report has mean $0.4$ and variance $0.04$ in
every row; the exponent alone changes the quantile-moment boundary.

```python
from xmath.incerto.risk_measures import beta_pareto_quantile_moment

for alpha in (.5, 1, 2):
    moments = [beta_pareto_quantile_moment(2, 3, alpha, r) for r in (1, 2)]
    print(f"alpha={alpha:g}: E[K]={moments[0]:.3f}, E[K²]={moments[1]:.3f}")
```

**Output**

```text
alpha=0.5: E[K]=inf, E[K²]=inf
alpha=1: E[K]=4.000, E[K²]=inf
alpha=2: E[K]=1.829, E[K²]=4.000
```

With a uniform report ($a=b=1$), $K$ itself has the exact Pareto
survival $(x_m/k)^\alpha$. This special case connects the transformed
moment boundary directly to the
[Pareto moment proof](incerto-pareto-moment-existence.md).
At $\alpha=1,x_m=1$, its indicator-truncated first moment is $\log H$.
Use [`pareto_truncated_moment`](incerto-api-reference.md#api-pareto-truncated-moment)
to evaluate that finite integral, keeping the infinite full moment distinct.

```python
from xmath.incerto.power_laws import pareto_truncated_moment

for cutoff in (10, 100, 1000):
    value = pareto_truncated_moment(cutoff, 1, 1)
    print(f"cutoff={cutoff}: truncated first moment={value:.2f}")
```

**Output**

```text
cutoff=10: truncated first moment=2.30
cutoff=100: truncated first moment=4.61
cutoff=1000: truncated first moment=6.91
```

(conflation-probability-figure)=
```{figure} ../assets/figures/incerto/conflation-probability-error.svg
:alt: Pareto inverse-survival curves amplify small tail probabilities; exact survival curves of beta-transformed quantiles become heavier as the Pareto exponent decreases.

Exact curves, $x_m=1$. Top: $K(p)$ for $\alpha=0.5,1,2$, with
both axes logarithmic. Bottom: $P(K>k)$ computed from the beta CDF
with $a=2,b=3$, using the same exponents. The transformed survival
exponents are $1,2,4$ respectively. A finite plot does not establish
moment divergence; the preceding integral does.
```

Small report variance alone is insufficient: for fixed $a$ and growing
$b$, the beta variance tends to zero while the moment boundary $a\alpha$
is unchanged. The report mean also tends to zero in this comparison.
This is not a fixed-mean, vanishing-noise theorem. Concentrating reports
away from zero, or imposing $\Pi\ge\varepsilon>0$, changes the conclusion.
Likewise, a fixed physical cap bounds the transformed loss quantile.

(conflation-decisions)=
## What an evaluation must establish

The mathematical issues raised in §§13.4–13.5, pp. 258–259, can be
stated without a judgment about a person or a trading record:

- Evaluation of one indicator does not identify an unbounded severity
  distribution. The model needs an exposure and conditional severity.
- One finite path cannot prove a universal distributional identity or
  exclude every rare-loss model. A specified iid sample and loss function
  can still supply statistical evidence; the sample size is not necessarily
  one merely because a record belongs to one strategy.
- A probability improvement has decision value only relative to actions,
  costs, and consequences. In a fixed-severity binary decision it can
  change the preferred action, as the
  [calibration extension](incerto-calibration-payoff.md#calibration-payoff-error)
  demonstrates. There is no general theorem that better probabilities
  are useless.

The [risk-measures note](incerto-risk-measures.md) treats a different
uncertainty model: a random tail exponent at a fixed probability level.
It separates expected model VaR/ES from the VaR/ES of a predictive mixture.
Neither construction follows just from the beta-report calculation.

## Sources and reproduction

Chapter 13 supplies the questions and locators. The attached article
[@taleb2023conflation, §§I–II, pp. 1–3] has the same probability-to-quantile
model; its discussion in §§III–IV, pp. 3–4 is context, not evidence for
a universal decision-performance claim. The arguments above prove their
stated finite/Pareto/beta cases. They do not cover every heavy-tailed family:
lognormal tails, for example, are not regularly varying power tails.

Generate both figures with
`uv run python demos/incerto/probability_conflation.py`.
