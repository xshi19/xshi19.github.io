---
title: VaR, expected shortfall, and tail uncertainty
options:
  concept:
    id: risk-measures
    type: method
    prerequisites: [pareto-moment-existence]
    related: [probability-conflation, calibration-payoff, hill-estimator, plug-in-tail-estimation]
    tags: [risk, quantiles, uncertainty]
---

A loss quantile specifies a threshold. Expected shortfall averages the
losses in an upper probability slice. Both depend on a probability law;
reporting either quantity without its model and estimation assumptions
leaves the uncertainty unresolved.

SCoFT v4, chapter 13, §13.3, pp. 257–258 [@taleb2025scoftv4], motivates
the error-propagation question. Its random-probability calculation is
derived in [probability conflation](incerto-probability-conflation.md#conflation-inverse-probability).
Here we define the risk measures precisely and prove additional results
for a random Pareto exponent. These two uncertainty models are distinct.

(risk-definitions)=
## Quantiles and expected shortfall

Let $X\ge0$ be a finite almost surely loss, with CDF $F$ and survival
$\bar F$ as in the [notation canon](https://xshi19.github.io/math/notation/).
Use $0<\tau<1$ for confidence and $p=1-\tau$ for tail mass;
$\alpha$ remains the tail exponent. Define

$$
\begin{gathered}
q_\tau=\inf\{x:F(x)\ge\tau\},\\
\operatorname{VaR}_\tau(X)=q_\tau,\\
\operatorname{ES}_\tau(X)
=\frac1p\int_\tau^1 q_s\,ds.
\end{gathered}
$$

The quantile is finite for each interior confidence level. The integral
may be infinite. ES selects the upper mass $p$, including the required
fraction of an atom at the quantile. Consequently the strict conditional
mean $\mathbb E[X\mid X>q_\tau]$ is not a general definition of ES.

We prove the equivalent excess-payoff formula, which also handles atoms.
Write $q=q_\tau$ and $C(t)=\mathbb E[(X-t)_+]$. Represent the loss
by its quantile function evaluated at a uniform random variable; this has
CDF $F$. Monotonicity gives $q_s\le q$ below $\tau$ and $q_s\ge q$
above $\tau$, so

$$
\begin{gathered}
C(q)=\int_\tau^1(q_s-q)\,ds,\\
\operatorname{ES}_\tau(X)=q+\frac{C(q)}p.
\end{gathered}
$$

These are nonnegative integrals and remain valid with infinity. If the
law is continuous at $q$ with $P(X>q)=p$, ES equals the conditional
tail mean. The unconditional tail-level quantity from the
[calibration note](incerto-calibration-payoff.md#calibration-contract) is
then $T(q)=p\operatorname{ES}_\tau(X)$, not ES itself. At an atom,
the exact expression is

$$
\begin{aligned}
p\operatorname{ES}_\tau(X)
&=\mathbb E[X\mathbf1_{\{X>q\}}]\\
&\quad+q\{p-P(X>q)\}.
\end{aligned}
$$

Use [`discrete_risk`](incerto-api-reference.md#api-discrete-risk).
Here the worst tenth of the law includes all the losses of 20 and some
losses of 5. Averaging only observations strictly above the quantile
would give a different answer.

```python
from xmath.incerto.risk_measures import discrete_risk

risk = discrete_risk([0, 5, 20], [.85, .1, .05], .1)
print(f"VaR={risk['var']:.1f}, ES={risk['es']:.1f}")
```

**Output**

```text
VaR=5.0, ES=12.5
```

VaR is also a minimizer of expected quantile loss when the first moment
is finite, as proved in the
[point-forecast note](incerto-point-forecast.md#point-forecast-quantiles).
Use [`quantile_loss`](incerto-api-reference.md#api-quantile-loss) and
[`finite_payoff_mean`](incerto-api-reference.md#api-finite-payoff-mean)
to compare three candidate thresholds under this same finite law.

```python
from xmath.incerto.forecasting import quantile_loss
from xmath.incerto.lecture_risk import finite_payoff_mean

for threshold in (0, 5, 20):
    losses = quantile_loss([0, 5, 20], threshold, tau=.9)
    risk = finite_payoff_mean([.85, .1, .05], losses)
    print(f"threshold={threshold}: expected quantile loss={risk:.2f}")
```

**Output**

```text
threshold=0: expected quantile loss=1.35
threshold=5: expected quantile loss=1.10
threshold=20: expected quantile loss=1.85
```

(risk-subadditivity)=
## A quantile can fail subadditivity

Consider mutually exclusive losses: $(X,Y)$ takes $(0,0)$ with
probability $0.92$, $(10,0)$ with probability $0.04$, and $(0,10)$
with probability $0.04$. At confidence $0.95$, each marginal has VaR
zero, while $X+Y$ has VaR 10. This finite example proves that VaR is
not subadditive over all joint loss laws. It is not a claim that every
portfolio violates subadditivity.

The quantile definition also applies to real integrable losses. On that
domain, ES is subadditive regardless of dependence. We prove this from
the excess formula. For any real $t$, let

$$
J_X(t)=t+\frac{\mathbb E[(X-t)_+]}p.
$$

Integrating $q_s-t\le(q_s-t)_+$ over the upper slice gives
$\operatorname{ES}_\tau(X)\le J_X(t)$. Equality holds at
$t=q_\tau$, so ES is the minimum of $J_X$. The pointwise inequality

$$
(X+Y-t-s)_+
\le(X-t)_++(Y-s)_+
$$

implies $J_{X+Y}(t+s)\le J_X(t)+J_Y(s)$. Choose marginal
quantile minimizers and minimize the left side to obtain

$$
\operatorname{ES}_\tau(X+Y)
\le\operatorname{ES}_\tau(X)+\operatorname{ES}_\tau(Y).
$$

Monotonicity, positive homogeneity, and translation by a constant follow
directly from the quantile definition, within the integrable loss domain.
These properties explain the coherence of ES. They do not bound the
sampling error of an estimated ES, nor make an infinite ES finite.

```python
from xmath.incerto.risk_measures import discrete_risk

for name, x in (("X", [0, 10, 0]), ("Y", [0, 0, 10]), ("X+Y", [0, 10, 10])):
    r = discrete_risk(x, [.92, .04, .04], .05)
    print(f"{name}: VaR={r['var']:.1f}, ES={r['es']:.1f}")
```

**Output**

```text
X: VaR=0.0, ES=8.0
Y: VaR=0.0, ES=8.0
X+Y: VaR=10.0, ES=10.0
```

(risk-pareto)=
## Exact Pareto risk and its moment boundary

For $X\sim\operatorname{Pareto}(\alpha,x_m)$ with $\alpha>0$,
$x_m>0$, inversion of $\bar F(x)=(x_m/x)^\alpha$ gives

$$
\begin{gathered}
q_\tau=x_m p^{-1/\alpha},\\
\operatorname{ES}_\tau(X)
=\frac{\alpha}{\alpha-1}q_\tau,\\
\alpha>1.
\end{gathered}
$$

To prove the second line, integrate the Pareto survival above $q_\tau$:
$C(q_\tau)=pq_\tau/(\alpha-1)$, then substitute into the excess
formula. For $0<\alpha\le1$ this integral diverges, hence ES is
infinite at every interior confidence level, while VaR remains finite.
These are exact-Pareto statements. The equality case for a general
regularly varying law can depend on its slowly varying factor; see
[moment existence](incerto-pareto-moment-existence.md).

Use [`pareto_risk`](incerto-api-reference.md#api-pareto-risk).

```python
from xmath.incerto.risk_measures import pareto_risk

for alpha in (.8, 1, 1.2, 2, 3):
    r = pareto_risk(alpha, .01)
    print(f"alpha={alpha:g}: VaR={r['var']:.2f}, ES={r['es']:.2f}")
```

**Output**

```text
alpha=0.8: VaR=316.23, ES=inf
alpha=1: VaR=100.00, ES=inf
alpha=1.2: VaR=46.42, ES=278.50
alpha=2: VaR=10.00, ES=20.00
alpha=3: VaR=4.64, ES=6.96
```

(risk-alpha-convexity)=
## Random exponents: averaging model risk

Hold the cutoff $x_m$ and tail mass $p$ fixed. Put $c=-\log p>0$
and view the quantile as the function $v(a)=x_m e^{c/a}$ of an
exponent $a>0$. Direct differentiation proves

$$
\begin{gathered}
v'(a)=-\frac{c}{a^2}v(a),\\
\frac{v''(a)}{v(a)}=\frac{2c}{a^3}+\frac{c^2}{a^4}>0.
\end{gathered}
$$

Thus $v$ is decreasing and strictly convex. If a positive random
exponent $A$ has finite mean $a_0$, Jensen gives
$\mathbb E[v(A)]\ge v(a_0)$, allowing an infinite left side.
Equality holds exactly when $A$ is almost surely constant, a degenerate
exponent with no parameter uncertainty. This averages the quantiles of
conditional models. It does not yet compute a predictive
mixture quantile, and it is not a bias claim about an arbitrary estimator:
if $A$ represents an estimator, its mean need not equal the true exponent.

For $a>1$, put $h(a)=a v(a)/(a-1)$ and $\ell(a)=\log h(a)$.
The ES curvature follows from

$$
\begin{gathered}
\ell'(a)=-\frac1{a(a-1)}-\frac{c}{a^2},\\
\ell''(a)=\frac{2a-1}{a^2(a-1)^2}+\frac{2c}{a^3}>0,\\
h''(a)=h(a)\{\ell''(a)+\ell'(a)^2\}>0.
\end{gathered}
$$

Therefore averaging model ES is at least ES at the mean exponent
when $A>1$ almost surely and $\mathbb E A<\infty$, with equality exactly
when $A$ is almost surely constant. Each component can have finite ES
while its average is infinite: for $A$ uniform on
$(1,2)$, $h(A)\ge x_m/(A-1)$ has infinite expectation. Positive
probability on $A\le1$ makes the average infinite immediately.

For a small perturbation $A=a_0+\varepsilon Z$, assume
$Z$ is bounded with mean zero. Taylor's theorem, on a compact interval
inside the relevant domain, proves for $f=v$ or $f=h$

$$
\begin{aligned}
\mathbb E f(A)-f(a_0)
&=\tfrac12\varepsilon^2f''(a_0)\mathbb E Z^2\\
&\quad+O(\varepsilon^3).
\end{aligned}
$$

The remainder is bounded by a constant times
$|\varepsilon|^3\mathbb E|Z|^3$. The ES expansion needs $a_0>1$
and a perturbation that stays above one. A variance alone, without these
support and remainder conditions, does not justify the approximation.
For comparisons between two uncertainty laws, convex order is sufficient;
larger variance alone is not.

(risk-alpha-mixture)=
## Mixing distributions is a different operation

Now specify $X\mid A\sim\operatorname{Pareto}(A,x_m)$ with the
same fixed $x_m$ in every component, $A>0$ almost surely, and finite
$a_0=\mathbb E A$. For $x>x_m$ put $t=\log(x/x_m)>0$.
Convexity of $a\mapsto e^{-at}$ proves

$$
\begin{gathered}
\bar F_{\rm mix}(x)=\mathbb E[e^{-At}],\\
\bar F_{\rm mix}(x)\ge e^{-a_0t}.
\end{gathered}
$$

This is first-order stochastic dominance over the Pareto law at $a_0$.
It implies larger or equal quantiles at every confidence level, and
larger or equal ES by integration of quantiles. For nonconstant $A$,
strict Jensen and continuous strictly decreasing mixture survival give
a strict quantile increase at every interior confidence level. The
inequality is specific to this common-cutoff exponent mixture. Holding
a median or mean fixed by changing $x_m$ is a different comparison.

The mixture's VaR is finite: dominated convergence gives
$\mathbb E[e^{-At}]\to0$ as $t\to\infty$. Its mean and ES can
still be infinite. With $A$ uniform on $(1,2)$, Tonelli gives an
infinite mixture mean, hence infinite ES; truncating the loss below any
finite quantile cannot remove that infinite tail mean.

There is also a sharp distinction between mixture VaR and average model
VaR. With $A$ uniform on $(0,1)$, $\mathbb E[e^{c/A}]=\infty$
for any $c>0$, while mixture VaR stays finite. The first claim follows
by substituting $z=1/A$: the resulting integral of $e^{cz}/z^2$
diverges. No identity interchanges averaging a quantile with mixing laws.

Use [`pareto_mixture_risk`](incerto-api-reference.md#api-pareto-mixture-risk)
for the predictive mixture and
[`pareto_risk`](incerto-api-reference.md#api-pareto-risk) for components.
The two exponents below have mean two. The mixture ES is evaluated at
its own quantile, not at either component quantile.

:::{dropdown} Compare three risk calculations

```python
from xmath.incerto.risk_measures import pareto_mixture_risk, pareto_risk

p = .01
components = [pareto_risk(a, p) for a in (1.2, 2.8)]
rows = {
    "mean exponent": pareto_risk(2, p),
    "average model risk": {k: sum(r[k] for r in components)/2 for k in ("var", "es")},
    "predictive mixture": pareto_mixture_risk([1.2, 2.8], [.5, .5], p),
}
for name, r in rows.items():
    print(f"{name}: VaR={r['var']:.2f}, ES={r['es']:.2f}")
```

**Output**

```text
mean exponent: VaR=10.00, ES=20.00
average model risk: VaR=25.80, ES=143.28
predictive mixture: VaR=26.17, ES=156.38
```

:::

(risk-mixture-figure)=
```{figure} ../assets/figures/incerto/risk-exponent-uncertainty.svg
:alt: At a fixed mean exponent of two, spreading two equally weighted Pareto exponents increases both averaged model risk and predictive-mixture risk; ES rises faster near the mean-existence boundary.

Exact-model evaluations: $x_m=1$, $p=0.01$, equally likely
$A=2-d$ and $A=2+d$, $0\le d\le0.95$. Separate panels show
VaR and ES, with logarithmic amount axes. The mixture quantile is found
numerically from its survival equation; ES uses the analytic tail integral
at that quantile. The flat reference uses exponent two. Lines compare
these specified models, not arbitrary uncertainty distributions.
```

(risk-estimation)=
## Estimation adds another source of uncertainty

The [Hill estimator](incerto-hill-estimator.md#hill-sampling-law) estimates
the extreme-value index $\xi=1/\alpha$ from a chosen number $k$ of
upper log spacings. Threshold choice and finite-sample error are additional
assumptions. In an exact Pareto model the fixed-$k$ reciprocal estimate
is biased; Jensen at its mean therefore does not by itself compare its
plug-in risk to the true risk. Nor does a distribution of repeated fitted
parameters automatically become a posterior mixture for future losses.

The example reuses
[`hill_alpha_estimator`](incerto-api-reference.md#api-hill-alpha-estimator)
and [`pareto_risk`](incerto-api-reference.md#api-pareto-risk).
It uses a known synthetic cutoff of one, so it isolates exponent error
instead of estimating the cutoff as well. At every $k$, it retains fits
with infinite ES. The intervals summarize sampling variation; they are
not confidence intervals.

:::{dropdown} Repeated fixed-k estimates

```python
import numpy as np
from xmath.incerto.estimators import hill_alpha_estimator
from xmath.incerto.risk_measures import pareto_risk

rng = np.random.default_rng(1307)
samples = 1 + rng.pareto(1.2, (400, 250))
for k in (25, 100):
    alphas = [hill_alpha_estimator(x, k) for x in samples]
    risks = [pareto_risk(a, .01) for a in alphas]
    lo, mid, hi = np.quantile([r['var'] for r in risks], [.1, .5, .9])
    infinite = sum(np.isinf(r['es']) for r in risks)
    print(f"k={k}: VaR 10/50/90%={lo:.1f}/{mid:.1f}/{hi:.1f}; infinite ES={infinite}/400")
```

**Output**

```text
k=25: VaR 10/50/90%=16.9/45.4/131.3; infinite ES=66/400
k=100: VaR 10/50/90%=29.6/46.4/76.3; infinite ES=13/400
```

:::

(risk-estimation-figure)=
```{figure} ../assets/figures/incerto/risk-estimation.svg
:alt: Repeated Pareto Hill fits have a wide sampling range of plug-in VaR at small tail counts; a separate panel shows the fraction of fitted exponents yielding infinite ES.

Synthetic iid Pareto $\alpha=1.2,x_m=1$, 2,000 independent samples
of size 500, PCG64 seed 1308. Each sample is reused at
$k=20,50,100,200,400$. Top: 10th–90th sampling quantiles and median
of plug-in VaR at $p=0.01$, with the exact population VaR as a dashed
line. Bottom: observed fraction with infinite plug-in ES. All one million
draws and all fits are retained; increasing $k$ is appropriate here because
the entire law is exact Pareto, not because more body observations are
always suitable for a tail fit.
```

At $k=400$, 0 of 2,000 fits have infinite ES, a finite-sample count that
does not establish zero population probability; the
rule of three gives an
approximate one-sided 95% upper confidence bound of $3/2000=0.15\%$.

The same moment distinction applies to direct sample ES: observations
can yield a finite numerical average even when the population ES is
infinite. The [point-forecast evaluation](incerto-point-forecast.md#point-forecast-evaluation)
and [plug-in mean](incerto-plug-in-tail-estimation.md) pages explain the
related loss-moment and boundary problems.

## Sources and reproduction

The definitions, finite counterexample, ES subadditivity, Pareto formulas,
and common-cutoff uncertainty comparisons are proved above. Chapter 13
supplies the error-propagation context; it is not cited as a proof of every
extension here. The two figures are generated by
`uv run python demos/incerto/probability_conflation.py`, through the shared
figure style and exporter. No observed financial data or investment
performance claim enters the examples.
