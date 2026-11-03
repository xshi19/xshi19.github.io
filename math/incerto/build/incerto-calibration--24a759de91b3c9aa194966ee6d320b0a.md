---
title: Probability calibration versus payoff accuracy
options:
  concept:
    id: calibration-payoff
    type: method
    prerequisites: [asymmetric-payoffs, moment-estimation]
    related: [point-forecast, ml-econometrics]
    tags: [calibration, scoring, payoffs]
---

A probability forecast describes an event. A loss forecast describes its
consequences. Even a perfectly calibrated event probability leaves the
conditional severity unspecified. The [asymmetric-payoff note](incerto-asymmetric-payoffs.md)
already separates gain frequency from expected gain; here we derive the
sampling laws of event scores and compare them with unbounded exposures.

The source context is SCoFT v4, chapter 11, §§11.1–11.5 and §11.7,
printed pp. 220–240 [@taleb2025scoftv4]. The calculations below are original
model arguments. They do not establish a result about participants in
psychology experiments, forecasters, or financial markets.

(calibration-contract)=
## Specify the event and its payoff

Let $X\ge0$ be a loss amount, $u>0$ a fixed threshold, and
$D=\mathbf1_{\{X>u\}}$. We use the strict event and the
[shared survival notation](https://xshi19.github.io/math/notation/)
$p_u=\bar F(u)$. A binary contract pays one unit when $D=1$;
a vanilla excess contract pays $(X-u)_+=\max(X-u,0)$.
Its expected payoff is not an option price unless a pricing measure,
discounting convention, and contract costs have also been specified.

Assume $p_u>0$. Write $T(u)$ for the unconditional tail-level mean
and $C(u)$ for the unconditional excess mean. We prove their decomposition:

$$
\begin{gathered}
T(u)=\mathbb E[X D],\\
C(u)=\mathbb E[(X-u)_+],\\
T(u)=u p_u+C(u),\\
C(u)=p_u e(u).
\end{gathered}
$$

Indeed, $XD=uD+(X-u)_+$ pointwise. Taking nonnegative expectations
proves the third line even when a mean is infinite. Conditioning the
excess on $D=1$ proves the fourth, with $e(u)$ the
[mean excess](incerto-mean-excess-function.md). Thus replacing $T(u)$ by
$up_u$ omits the entire excess contribution.

A threshold, horizon, units, and observation rule can define a disaster
indicator under any tail law. Fat tails do not make measurable events
undefined. Words such as “collapse” alone do not specify those choices,
and a specified indicator does not encode all losses above its threshold.
A fixed cash amount plus one binary payoff is constant on $X>u$;
it cannot equal $(X-u)_+$ on an unbounded support. This proves a static
nonreplication statement for those two instruments. It is not a theorem
about every dynamic trading strategy. Pointwise, a continuum of thresholds
can recover the excess:

$$
(x-u)_+=\int_u^\infty\mathbf1_{\{x>v\}}\,dv.
$$

Use [`tail_bet_payoffs`](incerto-api-reference.md#api-tail-bet-payoffs).

```python
from xmath.incerto.calibration import tail_bet_payoffs

payoffs = tail_bet_payoffs([1, 4, 5, 20], 4)
for name, values in payoffs.items():
    print(name + ": " + ", ".join(f"{v:.1f}" for v in values))
```

**Output**

```text
binary: 0.0, 0.0, 1.0, 1.0
excess: 0.0, 0.0, 1.0, 16.0
tail_level: 0.0, 0.0, 5.0, 20.0
```

(calibration-tail-models)=
## Equal event probabilities leave severity open

For an exact Pareto law with $u\ge x_m>0$ and $\alpha>1$, the
[Pareto moment calculation](incerto-pareto-moment-existence.md) gives

$$
\begin{gathered}
p_u=(x_m/u)^\alpha,\\
C(u)=\frac{u p_u}{\alpha-1},\\
\frac{T(u)}{u p_u}=\frac{\alpha}{\alpha-1}.
\end{gathered}
$$

The excess formula also follows directly by integrating
$C(u)=\int_u^\infty\bar F(v)\,dv$, justified by the preceding
indicator identity and Tonelli's theorem. For $\alpha\le1$, both payoff
means are infinite. The binary payoff remains bounded in every case.

For an exponential loss of mean $b>0$, the same integration yields
$C(u)=b p_u$ and $T(u)/(up_u)=1+b/u$. These exact models explain why
substituting the threshold for conditional severity can have very different
relative errors. The limiting ratio alone is not a definition of all thin
or heavy tails. The [Weibull example](incerto-scoft-ch5.md) with survival
$\exp(-\sqrt x)$ is subexponential, but its mean excess
$2\sqrt u+2$ gives $T(u)/(up_u)\to1$. See also the
[mean-excess corollary](incerto-pickands-balkema-de-haan.md#pbdh-mean-excess-corollary)
for the hypotheses needed to pass from tail-law limits to means.

For any fixed $0<p<1$ and $u>0$, choose $x_m=u p^{1/\alpha}$.
All these Pareto laws have the same event probability $p$, while
$C(u)=up/(\alpha-1)$ is unbounded as $\alpha\downarrow1$.
No statistic of this single indicator alone identifies that payoff mean.
Defining $p^*=T(u)/u$ converts an expected amount to threshold units;
$p^*$ can exceed one and is then certainly not a probability. Even when
it lies below one, it is not the probability of the original event.

Use [`pareto_tail_bet`](incerto-api-reference.md#api-pareto-tail-bet).
The three finite models below have exactly the same event probability.

```python
from xmath.incerto.calibration import pareto_tail_bet

for alpha in (1.1, 1.5, 3):
    result = pareto_tail_bet(alpha, 10, xm=10 * .02**(1 / alpha))
    print(f"alpha={alpha:.1f}: p={result['probability']:.3f}, excess={result['excess_mean']:.3f}")
print("Infinite-mean boundary:", pareto_tail_bet(1, 10)["excess_mean"])
```

**Output**

```text
alpha=1.1: p=0.020, excess=2.000
alpha=1.5: p=0.020, excess=0.400
alpha=3.0: p=0.020, excess=0.100
Infinite-mean boundary: inf
```

(calibration-payoff-figure)=
```{figure} ../assets/figures/incerto/calibration-tail-payoffs.svg
:alt: A step payoff and an excess payoff at threshold four; below them, Pareto excess means rise sharply near exponent one although the event probability stays at two percent.

Exact model curves. Top: strict binary and excess payoffs at $u=4$;
the open step endpoint excludes equality. Bottom: $u=10$, $p_u=0.02$,
$1.05\le\alpha\le4$, with $x_m$ adjusted as above. Probability is
held fixed; the excess mean is in loss units. No observed data or fitted
exponent enters either panel.
```

(calibration-lognormal)=
## A mean-preserving change can move probability and severity oppositely

Let $Z$ be standard normal, $m>0$, and $s>0$. Define a lognormal loss
with fixed mean $m$ by

$$
X=m\exp(sZ-s^2/2).
$$

At threshold $u=m$, completing the square in the normal density gives

$$
\begin{gathered}
P(X>m)=\Phi(-s/2),\\
T(m)=m\Phi(s/2),\\
C(m)=m\{2\Phi(s/2)-1\}.
\end{gathered}
$$

To prove the formulas, $X>m$ is equivalent to $Z>s/2$, and
$e^{sz-s^2/2}\varphi(z)=\varphi(z-s)$, where $\varphi$ is the
standard normal density. Integrating over $z>s/2$ gives $T(m)$;
subtracting $mP(X>m)$ gives $C(m)$. Thus increasing $s$ decreases
the event probability but increases both payoff means. Their limits are
zero, $m$, and $m$, respectively; neither expectation diverges in this
fixed-mean model. At $s=0$, $X=m$ surely, so the *strict* event has
probability zero. The binary and tail-level means are discontinuous at
that degenerate boundary.

Use [`lognormal_tail_bet`](incerto-api-reference.md#api-lognormal-tail-bet).

```python
from xmath.incerto.calibration import lognormal_tail_bet

for sigma in (0, .01, 1, 3, 8):
    r = lognormal_tail_bet(sigma)
    print(f"s={sigma:.2f}: p={r['probability']:.4f}, tail={r['tail_mean']:.4f}, excess={r['excess_mean']:.4f}")
```

**Output**

```text
s=0.00: p=0.0000, tail=0.0000, excess=0.0000
s=0.01: p=0.4980, tail=0.5020, excess=0.0040
s=1.00: p=0.3085, tail=0.6915, excess=0.3829
s=3.00: p=0.0668, tail=0.9332, excess=0.8664
s=8.00: p=0.0000, tail=1.0000, excess=0.9999
```

(calibration-uncertainty)=
## Distributional uncertainty needs a specified mixture

Jensen's inequality applies to a particular function of a particular
random parameter. It does not imply that every uncertainty change increases
every payoff discrepancy. Here is one precise case, proved directly.
Fix $u\ge x_m>0$ and let $A>1$ be a random Pareto exponent. Conditional
on $A=a$, use the Type-I law with that exponent and the same $x_m$.
Let $c(a)$ denote its excess mean, and $\ell=\log(u/x_m)\ge0$.

$$
\begin{gathered}
c(a)=\frac{u e^{-a\ell}}{a-1},\\
\begin{aligned}
\frac{c''(a)}{c(a)}&=\left(\ell+\frac1{a-1}\right)^2\\
&\quad+\frac1{(a-1)^2}>0.
\end{aligned}
\end{gathered}
$$

Logarithmic differentiation proves the second identity. Tonelli gives the
mixture excess mean $\mathbb E[c(A)]$. If $\mathbb EA<\infty$, convexity
therefore gives $\mathbb E[c(A)]\ge c(\mathbb EA)$, allowing an infinite
left side. A finite mean exponent above one does not ensure a finite
mixture payoff: take $A$ uniform on $(1,2)$ and $u=x_m=1$; then
$\mathbb E[1/(A-1)]=\infty$.

The mixture probability is $\mathbb E[e^{-A\ell}]$. Consequently the
ratio $T(u)/(up_u)$ in the mixture weights each exponent by its probability
of exceeding $u$. It is generally neither the ratio evaluated at
$\mathbb EA$ nor the unweighted average of the conditional ratios.
This distinction prevents a Jensen comparison of one quantity from being
silently transferred to another. The
[mixture-uncertainty note](incerto-mixture-uncertainty.md) treats other
parameter choices and moment gates.

For an equal mixture of exponents 1.2 and 2.8 at $u=x_m=1$, use the
same [`pareto_tail_bet`](incerto-api-reference.md#api-pareto-tail-bet).

```python
from xmath.incerto.calibration import pareto_tail_bet

mixed = sum(pareto_tail_bet(a, 1)["excess_mean"] for a in (1.2, 2.8)) / 2
plugin = pareto_tail_bet(2, 1)["excess_mean"]
print(f"Mixture excess: {mixed:.3f}; mean-exponent model: {plugin:.3f}")
```

**Output**

```text
Mixture excess: 2.778; mean-exponent model: 1.000
```

(calibration-definition)=
## Calibration and the Brier score

Let $Q\in[0,1]$ be the announced probability and $D\in\{0,1\}$ the
outcome. Here $Q$ is local forecast notation, not a probability measure.
Calibration means $\mathbb E[D\mid Q]=Q$ almost surely. It concerns
conditional frequencies among comparable forecasts. A constant forecast
$Q=p$ is calibrated whenever $P(D=1)=p$, even if it cannot distinguish
high-risk from low-risk cases.

Write $m(Q)=\mathbb E[D\mid Q]$ and $R=(Q-D)^2$ for one Brier score.
We use the binary convention on $[0,1]$, without the factor of two that
arises from summing the errors for both outcome categories. Expansion of
the square proves

$$
\begin{aligned}
\mathbb E[R\mid Q]
&=m(Q)\{1-m(Q)\}\\
&\quad+\{Q-m(Q)\}^2.
\end{aligned}
$$

For a fixed conditional event probability, the expected score is uniquely
minimized by reporting that probability. This is strict propriety; it does
not mean that each finite sample rewards the truthful report, or that
calibration identifies a severity distribution. Among calibrated forecasts,
$\mathbb E R=\mathbb E[Q(1-Q)]$: more informative forecasts can have lower
scores than the calibrated constant forecast.

For $n\ge1$ pairs define $B_n=n^{-1}\sum_i R_i$. Pointwise,
$0\le B_n\le1$, with **no independence assumption**. Every positive
moment and every real exponential moment is finite. Boundedness does not
imply small standardized kurtosis or fast averaging under dependence.
For example, if every $D_i$ equals the same Bernoulli variable and every
$Q_i=0$, then $B_n=D_1$ for all $n$; its variance never decreases.

Use [`brier_score`](incerto-api-reference.md#api-brier-score) and
[`calibration_curve`](incerto-api-reference.md#api-calibration-curve).
The latter bins forecasts and reports actual counts. Empty bins remain
visible as `nan`; a small bin cannot certify a population conditional mean.

```python
from xmath.incerto.calibration import brier_score, calibration_curve

q, d = [0, .2, .2, .8, 1], [0, 0, 1, 1, 1]
print(f"Brier: {brier_score(q, d):.3f}")
c = calibration_curve(q, d, bins=4)
for count, forecast, frequency in zip(c["count"], c["forecast_mean"], c["event_frequency"]):
    print(f"n={count}: forecast={forecast:.2f}, frequency={frequency:.2f}")
```

**Output**

```text
Brier: 0.144
n=3: forecast=0.13, frequency=0.33
n=0: forecast=nan, frequency=nan
n=0: forecast=nan, frequency=nan
n=2: forecast=0.90, frequency=1.00
```

(calibration-curve-figure)=
```{figure} ../assets/figures/incerto/calibration-curves.svg
:alt: Observed bin frequencies follow the diagonal for calibrated forecasts and fall below the diagonal for overforecasting; a separate panel shows counts in all ten bins.

Synthetic forecasts $Q$ uniform on $[0,1]$, with $n=4000$ per experiment.
The two generating conditional event probabilities are $Q$ and $Q^2$.
PCG64 seeds 1103 and 1104. Points are bin means, solid/dashed curves are
the generating functions, and error bars are $\pm2$ plug-in standard errors
for independent Bernoulli outcomes within the displayed bins. They are
pointwise descriptive intervals, not simultaneous coverage or a calibration
test. Counts are plotted separately; binning can conceal variation within bins.
```

(calibration-tally)=
## Exact tally and fixed-forecast score laws

Now assume $D_1,\ldots,D_n$ are iid Bernoulli with parameter $p$.
The tally $K_n=\sum_iD_i$ counts occurrences, or correct predictions
if $D_i$ was defined as correctness. Independence gives the generating
polynomial $(1-p+pt)^n$. Its coefficients prove

$$
\begin{gathered}
P(K_n=k)={n\choose k}p^k(1-p)^{n-k},\\
k=0,\ldots,n,\\
\mathbb E[K_n/n]=p,\\
\operatorname{Var}(K_n/n)=\frac{p(1-p)}n.
\end{gathered}
$$

This law is independent of the magnitude distribution used to define each
event, once $p$ is fixed. Unequal event probabilities give a different
law, and dependent indicators need their joint law. Differentiating
$n\log(1-p+pe^{t/n})$ gives the third and fourth cumulants of the
fraction, $p(1-p)(1-2p)/n^2$ and
$p(1-p)\{1-6p(1-p)\}/n^3$. For $0<p<1$, its excess kurtosis is
$\{1-6p(1-p)\}/\{np(1-p)\}$. At $p=0$ or $1$ the fraction is
constant and standardized kurtosis is undefined.

The variance identity and Chebyshev's inequality prove convergence in
probability to $p$. The iid finite-variance CLT
(@durrett2019probability, Theorem 3.4.1)
gives a standard normal limit for the centered fraction divided by
$\sqrt{p(1-p)/n}$ when $0<p<1$. An uncentered fraction converges to a
point mass. An $n^{-1}$ decay of excess kurtosis is not a general
$n^{-1}$ bound on normal-approximation error. See also the
[limit-law note](incerto-generalized-central-limit-theorem.md).

If every forecast equals the same $q\in[0,1]$, expand each score to obtain

$$
B_n=q^2+(1-2q)K_n/n.
$$

This proves the entire finite score distribution by transforming the
binomial support. In particular,

$$
\begin{aligned}
\mathbb E B_n&=p(1-p)+(q-p)^2,\\
\operatorname{Var}(B_n)&=\frac{(1-2q)^2p(1-p)}n.
\end{aligned}
$$

At $q=1/2$, $B_n=1/4$ for every outcome sequence, including sequences
from a miscalibrated model. For $q\ne1/2$ and $0<p<1$, its standardized kurtosis is
that of the tally. With $n=1,p=0.01,q=0$, it exceeds 98 despite
bounded support. Finite positive moments do not imply kurtosis below three.

Use [`binary_tally_distribution`](incerto-api-reference.md#api-binary-tally-distribution)
and [`brier_score_distribution`](incerto-api-reference.md#api-brier-score-distribution).

```python
from xmath.incerto.calibration import binary_tally_distribution, brier_score_distribution

fraction, mass = binary_tally_distribution(4, .2)
print(f"Tally mean={mass @ fraction:.3f}; probability of zero={mass[0]:.4f}")
for q in (.2, .5):
    score, weight = brier_score_distribution(4, .2, q)
    print(f"q={q:.1f}: mean={weight @ score:.3f}; atoms={len(score)}")
```

**Output**

```text
Tally mean=0.200; probability of zero=0.4096
q=0.2: mean=0.160; atoms=5
q=0.5: mean=0.250; atoms=1
```

(calibration-beta)=
## Random forecasts: two distinct beta experiments

Suppose $Q$ has beta density $f_{a,b}$ on $(0,1)$, with shapes $a,b>0$.
A beta marginal does not specify how $D$ depends on $Q$. Nor does the
probability integral transform imply that arbitrary forecasters' reports
have a beta distribution. We choose a beta model for calculation and
compare two joint laws explicitly.

First let $D$ be independent Bernoulli$(p)$. For $0<z<1$, put
$s=\sqrt z$. When $D=0$, the event $R\le z$ is $Q\le s$;
when $D=1$, it is $Q\ge1-s$. Changing variables proves the density

$$
\begin{gathered}
\begin{aligned}
h(z)&=\frac{(1-p)f_{a,b}(s)}{2s}\\
&\quad+\frac{p f_{a,b}(1-s)}{2s},
\end{aligned}\\
s=\sqrt z,\qquad 0<z<1.
\end{gathered}
$$

There are no endpoint atoms for positive beta shapes. Let $\mathrm B$
denote the beta function and $M_r=\mathbb E[R^r]$. Integrating the
powers against the joint density proves, for $r>0$,

$$
\begin{aligned}
M_r&=(1-p)\frac{\mathrm B(a+2r,b)}{\mathrm B(a,b)}\\
&\quad+p\frac{\mathrm B(a,b+2r)}{\mathrm B(a,b)}.
\end{aligned}
$$

This independent experiment is generally miscalibrated: its conditional
probability is the constant $p$, not the random $Q$.

For a calibrated experiment, draw $D\mid Q=q$ as Bernoulli$(q)$.
Replace the joint weights above by $1-q$ and $q$. The same changes of
variables and beta integrals give

$$
\begin{gathered}
h_{\rm cal}(z)=\frac{1-s}{2s}H(s),\\
H(s)=f_{a,b}(s)+f_{a,b}(1-s),\\
s=\sqrt z,\qquad 0<z<1,
\end{gathered}
$$

$$
\begin{aligned}
M_r&=\frac{\mathrm B(a+2r,b+1)}{\mathrm B(a,b)}\\
&\quad+\frac{\mathrm B(a+1,b+2r)}{\mathrm B(a,b)}.
\end{aligned}
$$

For iid *pairs* $(Q_i,D_i)$, either model gives
$\mathbb E B_n=M_1$ and
$\operatorname{Var}(B_n)=(M_2-M_1^2)/n$.
Writing $\phi_R(t)=\mathbb E[e^{itR}]$, independence also gives the
exact characteristic function

$$
\phi_{B_n}(t)=\{\phi_R(t/n)\}^n.
$$

This convolution identity is exact; a normal curve is an approximation to
the standardized average when its variance is positive, by the iid
finite-variance CLT (@durrett2019probability, Theorem 3.4.1).
The helpers evaluate the exact one-score CDF and moments. For averages of
random beta forecasts, they simulate the distribution instead of claiming
that a fitted Gaussian is the exact finite law.

As a check, take $a=b=1$. The independent model has CDF $\sqrt z$,
mean $1/3$, and variance $4/45$, for any $p$. The calibrated model has
CDF $2\sqrt z-z$, mean $1/6$, and variance $7/180$. These follow by
integrating the preceding densities. The identical forecast marginal hides
very different scoring behavior.

Use [`beta_brier_cdf`](incerto-api-reference.md#api-beta-brier-cdf),
[`beta_brier_moments`](incerto-api-reference.md#api-beta-brier-moments), and
[`simulate_brier_scores`](incerto-api-reference.md#api-simulate-brier-scores).
These are model simulations with 20,000 independent averages of 20 scores.
The reported Monte Carlo standard error uses the exact model variance.

:::{dropdown} Code: compare the two joint laws
```python
import numpy as np
from xmath.incerto.calibration import beta_brier_cdf, beta_brier_moments, simulate_brier_scores

for p, label in ((.5, "independent"), (None, "calibrated")):
    exact = beta_brier_moments(1, 1, p=p, n=20)
    draws = simulate_brier_scores(1, 1, 20, p=p, repetitions=20000, seed=1107)
    se = np.sqrt(exact["average_variance"] / len(draws))
    print(label)
    print(f"  one-score CDF at .25: {float(beta_brier_cdf(.25, 1, 1, p=p)):.3f}")
    print(f"  exact mean={exact['mean']:.4f}; simulated={draws.mean():.4f}; MC SE={se:.4f}")
    print(f"  exact average variance={exact['average_variance']:.5f}")
    print("  average quantiles: " + ", ".join(f"{v:.3f}" for v in np.quantile(draws, [.1, .5, .9])))
```

**Output**

```text
independent
  one-score CDF at .25: 0.500
  exact mean=0.3333; simulated=0.3337; MC SE=0.0005
  exact average variance=0.00444
  average quantiles: 0.249, 0.332, 0.421
calibrated
  one-score CDF at .25: 0.750
  exact mean=0.1667; simulated=0.1669; MC SE=0.0003
  exact average variance=0.00194
  average quantiles: 0.112, 0.164, 0.226
```
:::

(calibration-score-figure)=
```{figure} ../assets/figures/incerto/calibration-score-laws.svg
:alt: A fixed-forecast Brier distribution has binomially weighted atoms; two one-score CDFs and simulated twenty-score averages differ between independent and calibrated uniform forecasts.

Top: exact masses for $n=20,p=0.2,q=0.2$, with the deterministic
$q=0.5$ score marked separately. Middle: exact uniform-forecast CDFs
for independent outcomes ($p=0.5$) and calibrated outcomes. Bottom:
20,000 simulated averages per model, each of 20 scores; PCG64 seed 1107.
Vertical lines mark the exact means. The histograms estimate finite
sampling distributions; they are not proofs or observed forecasting records.
```

(calibration-nonlinear)=
## Evaluate the specified exposure

For any integrable payoff $g(X)$ and event $A$ with $P(A)>0$,
conditioning gives the exact identity

$$
\mathbb E[g(X)\mathbf1_A]
=P(A)\mathbb E[g(X)\mid A].
$$

Replacing the conditional expectation by $g(\mathbb E[X\mid A])$
requires additional assumptions and generally changes the answer.
For convex $g$, conditional Jensen gives a weak inequality when the
expectations are defined; strict inequality needs more than nonlinearity
alone. An arbitrary measurable $g$ has no fixed Jensen direction.
Expectation of a finite integrable hinge sum is the corresponding sum of
expectations. An infinite sum needs an interchange justification.

The canonical [hinge and angle treatment](incerto-ml-econometrics.md#pipeline-hinges)
proves the finite piecewise-linear representation and explains smoothing
and extrapolation. It is the relevant link to chapter 11's ML discussion.
It does not assert exact finite representation of every nonlinear payoff.
For sequential exposure, [absorbing resource paths](incerto-ruin.md)
retain all prior crossings. Conditioning on survival changes a distribution;
it does not generally leave the original payoff law unchanged.

Binary calibration remains useful for a specified event. Choosing a safer
action need not require distorting its probability: decision costs and
severity belong in the payoff or loss function. The
[point-forecast note](incerto-point-forecast.md) derives what changes when
we score amounts rather than event indicators.

## Sources and reproduction

The [chapter 11 guide](incerto-scoft-ch11.md) maps every section to its v4
locator and records the main qualifications to the source. The elementary
payoff, convexity, score, binomial and beta statements above are proved
here; the general iid finite-variance CLT is cited from Durrett,
*Probability: Theory and Examples*, fifth edition,
Theorem 3.4.1 [@durrett2019probability].
The figures are generated by
`uv run python demos/incerto/calibration_payoffs.py` using the shared
figure exporter. All curves and samples are model calculations.
