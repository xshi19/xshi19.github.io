---
title: Power-law goodness of fit and alternatives
options:
  concept:
    id: power-law-goodness-of-fit
    type: method
    prerequisites:
      - power-law-fitting
    tags:
      - power-law
      - model-comparison
      - bootstrap
---

A best-fitting power law can still fit badly. Two questions follow the
[exponent and threshold fit](incerto-power-law-fitting.md): how unusual is
its discrepancy under a power-tail model, and does another tail family fit
better? The first asks about absolute goodness of fit; the second compares
specified alternatives. Neither can be answered by a straight-looking
log-log plot.

We use iid positive observations, a full sample of size $N$, and
$n=n_{\mathrm{tail}}$ retained values $x_i\ge u=x_{\min}$. As in the
[shared notation](https://xshi19.github.io/math/notation/), $\alpha>0$ is the
survival exponent. The local density exponent is $a=\alpha+1$; CSN uses its
symbol $\alpha$ for this $a$. All package `alpha` values below follow the
survival convention.

:::{note} Worked example
The [tech layoffs example](incerto-layoffs-tails.md) applies these methods
to event counts and workforce log losses. The examples on this theory page
remain synthetic.
:::

(csn-gof-bootstrap)=
## A fitted-null bootstrap for the KS distance

Let $(\widehat\alpha,\widehat u)$ be the fit obtained by minimizing the
refitted KS distance over the declared candidate rule, and let $D_{\mathrm{obs}}$
be that minimum. A textbook KS p-value for a fully specified continuous CDF
is inappropriate: the exponent and cutoff were learned from these same data.
Discrete data also lack the continuous distribution-free KS reference law.

The semiparametric bootstrap of @clauset2009powerlaw, §4.1, constructs a null
population with an empirical body and a fitted power-law tail. For each of
$B$ independent replicates:

1. Generate $N$ independent observations. For each draw, choose the fitted
   tail with probability $n/N$ and the body with probability $1-n/N$.
2. A tail draw follows the fitted continuous Pareto or discrete zeta law above
   $\widehat u$. A body draw is sampled uniformly, with replacement, from the
   original observations strictly below $\widehat u$.
3. Fit the replicate from the beginning, including its own threshold search
   and exponent estimate. Use the same minimum tail count and candidate
   **rule** as before; an observed-value rule uses this replicate's values.
4. Calculate $D_b^*$ against that replicate's fitted model on its selected tail.

The tail count before refitting is binomial with parameters $N$ and $n/N$.
Fixing it at the observed count would define a different resampling experiment.
If the body is empty, every draw comes from the fitted tail. The model leaves
the body's shape unspecified; it does not pretend the whole sample is Pareto.

We estimate the tail probability by

$$
\widehat p=\frac1B\sum_{b=1}^B
\mathbf1\{D_b^*\ge D_{\mathrm{obs}}\}.
$$

The inclusive comparison records ties conservatively, particularly for integer
data. The helper uses this empirical fraction, which can equal zero; zero
means no exceedance in these $B$ replicates, not a proven zero probability.
Fitted-null bootstrap calibration is an approximation to the composite testing
problem, not an exact finite-sample uniform-p theorem.

A small p-value rejects this power-tail fitting procedure at a chosen
significance level. A large value means the observed discrepancy was not
unusual in the fitted-null experiment. It neither proves a power law nor gives
the probability that the hypothesis is true. Small tail samples can have little
power to distinguish curved tails. Choosing a new candidate rule after seeing
the p-value introduces another selection step.

### How many replicates?

Conditional on the fitted null and data, the indicators are iid Bernoulli
with some Monte Carlo target probability $p_*$. Hence

$$
\begin{aligned}
\operatorname{Var}_{\mathrm{MC}}(\widehat p)
&=\frac{p_*(1-p_*)}{B},\\
\operatorname{se}_{\mathrm{MC}}(\widehat p)&\le\frac1{2\sqrt B}.
\end{aligned}
$$

Requiring this worst-case standard error to be at most $\varepsilon$ gives
$B\ge1/(4\varepsilon^2)$. This is a one-standard-error precision rule,
not a guarantee of absolute error within $\varepsilon$. For an approximate
95% margin, the corresponding normal calculation multiplies the requirement
by $1.96^2$. Near zero or one, a binomial interval is more informative than a
plug-in standard error of zero. Increasing $B$ reduces simulation noise; it
does not repair model bias, dependence, or threshold-selection uncertainty in
the original observations.

[`powerlaw_gof_bootstrap`](incerto-api-reference.md#api-powerlaw-gof-bootstrap)
refits every replicate and fails explicitly if a fit fails. It does not discard
difficult replicates. Here the small sample and $B=39$ keep the demonstration
fast. Its worst-case Monte Carlo standard error is about $0.08$, so these
outputs are illustrations rather than precise testing decisions. We use
[`sample_powerlaw`](incerto-api-reference.md#api-sample-powerlaw) to generate
the synthetic tail.

:::{dropdown} Code

```python
import numpy as np
from xmath.incerto.powerlaw import sample_powerlaw, powerlaw_gof_bootstrap

rng = np.random.default_rng(340)
x = rng.uniform(1, 3, 240)
tail = rng.random(x.size) < 0.4
x[tail] = sample_powerlaw(1.8, 3, int(tail.sum()), random_state=rng)
result = powerlaw_gof_bootstrap(x, replicates=39, min_tail=40, random_state=341)
fit = result["fit"]
print(f"xmin={fit['xmin']:.3f}; tail count={fit['n_tail']}")
print(f"KS={fit['ks']:.3f}; bootstrap p={result['p_value']:.3f}")
print(f"Worst-case Monte Carlo SE: {result['worst_case_mc_se']:.3f}")
print(f"Distinct refitted thresholds: {len(np.unique(result['xmins']))}")
```

**Output**

```text
xmin=3.315; tail count=89
KS=0.066; bootstrap p=0.410
Worst-case Monte Carlo SE: 0.080
Distinct refitted thresholds: 33
```
:::

(csn-common-tail-likelihood)=
## Compare normalized likelihoods on the same tail

Select a common threshold $u$ and keep exactly the same observations for every
comparison. If a candidate has unconditioned density $g(x;\theta)$ and survival
$\bar G(u;\theta)$, its conditional tail density is

$$
\begin{aligned}
q(x;\theta,u)&=\frac{g(x;\theta)}{\bar G(u;\theta)},\qquad x\ge u,\\
\ell(\theta;u)&=\sum_{i=1}^n\log g(x_i;\theta)\\
&\quad-n\log\bar G(u;\theta).
\end{aligned}
$$

This display assumes a continuous law. For an integer law the denominator is
$P_\theta(X\ge u)$, and numerator and denominator must both use counting
measure. Do not compare log probability masses with log densities, or let
one model discard observations that another model must explain.

Conditioning changes the fit, not merely its reported likelihood. For example,
fitting an ordinary lognormal to the retained values and subtracting a tail
normalizer afterward is generally not maximum likelihood for the conditional
model, because that normalizer depends on the parameters.

The following continuous alternatives all have support $[u,\infty)$ after
conditioning. Each parameter is fitted on that same retained sample.

### Lognormal

Let $\mu\in\mathbb R$ and $\sigma>0$ be the location and standard deviation
of the untruncated log variable. With $\bar\Phi=1-\Phi$,

$$
\begin{aligned}
z(x)&=(\log x-\mu)/\sigma,\\
q_{\mathrm{LN}}(x)&=\frac{\exp[-z(x)^2/2]}
{x\sigma\sqrt{2\pi}\,\bar\Phi(z(u))}.
\end{aligned}
$$

The denominator accounts for the omitted lower part of the parent lognormal.
A broad lognormal can mimic a power tail over a finite interval. In limiting
parameter sequences its conditional density can approach a Pareto law, so
regular, distinguishable interior fits cannot simply be presumed.

### Exponential

For rate $\lambda>0$, conditioning an exponential above $u$ gives

$$
\begin{aligned}
q_{\mathrm E}(x)&=\lambda e^{-\lambda(x-u)},\\
\widehat\lambda&=\left[\frac1n\sum_{i=1}^n(x_i-u)\right]^{-1}.
\end{aligned}
$$

This rate is determined by excesses above the cutoff. A mean of the raw
retained values gives the wrong rate for this conditional model.

### Stretched exponential / Weibull

Write $y=x/u$, shape $k>0$, and $\beta=(u/\tau)^k>0$, where $\tau$ is the
parent Weibull scale. Then

$$
q_{\mathrm W}(x)
=\frac{k\beta}{u}y^{k-1}\exp[-\beta(y^k-1)].
$$

The name *stretched exponential* usually refers to $0<k<1$; the implementation
fits the full Weibull family, including the exponential case $k=1$. At fixed
$k$, differentiating the conditional log likelihood gives

$$
\widehat\beta(k)=
\left[\frac1n\sum_{i=1}^n(y_i^k-1)\right]^{-1}.
$$

Profiling out $\beta$ leaves a one-dimensional numerical shape fit. Comparing
Weibull against exponential is a nested problem; comparing Weibull against
Pareto requires the non-nested conditions discussed below.

### Power law with exponential cutoff

The cutoff law is sometimes called a *truncated power law*. Here “cutoff” means
an exponential damping factor, not a hard finite upper endpoint. Define the
dimensionless cutoff $b=\lambda u$, with $\lambda>0$, and

$$
\begin{aligned}
C(a,b)&=\int_1^\infty z^{-a}e^{-b(z-1)}\,dz,\\
q_{\mathrm C}(x)&=\frac{(x/u)^{-a}e^{-b(x/u-1)}}{uC(a,b)}.
\end{aligned}
$$

For $b>0$, any real $a$ gives an integrable density. At $b=0$, normalization
requires $a>1$ and $C(a,0)=1/(a-1)$, recovering the pure Pareto law. The
coefficient $a-1$ in an interior cutoff fit is not its asymptotic survival
exponent: the exponential factor eventually dominates. All positive moments
of an interior cutoff law exist.

The normalizing integral is part of the optimization. Omitting it rewards
arbitrary changes of scale in the purported density and invalidates a
likelihood-ratio comparison.

(csn-vuong)=
## Non-nested comparisons: sign, normalization, and p-value

For two fitted conditional densities evaluated on the same $n$ observations,
let

$$
\begin{aligned}
d_i&=\log\widehat q_1(x_i)-\log\widehat q_2(x_i),\\
R&=\sum_{i=1}^n d_i,\qquad \bar d=R/n,\\
s_d^2&=\frac1n\sum_{i=1}^n(d_i-\bar d)^2,\\
Z_V&=\frac{R}{\sqrt n\,s_d}.
\end{aligned}
$$

Positive $R$ favors model 1; negative $R$ favors model 2. The normalized
statistic $Z_V$ is sometimes also called normalized $R$. Under the null that
the two pseudo-true models have equal expected log likelihood, Vuong's
non-nested result gives an asymptotic standard normal reference and the
two-sided p-value

$$
p_V=2\{1-\Phi(|Z_V|)\}.
$$

We cite the fitted-model asymptotic result in @vuong1989likelihood rather than
prove it. Its conditions include iid sampling, regular interior parameter
estimation, distinguishable non-nested models, and a finite, strictly positive
variance of the limiting per-observation contrast. The heuristic central
limit argument applies to that limiting contrast; fitted scores sharing
estimated parameters are not themselves independent observations.

These qualifications matter for heavy tails. Against an exponential, a
Pareto log-likelihood contrast contains a term proportional to $X$. Under an
exact Pareto generator, its variance is infinite if $\alpha\le2$, and even its
expectation can fail for $\alpha\le1$. A finite array of scores cannot verify
the population moment assumptions. Overlap, a nearly vanishing $s_d$, or an
optimizer drifting toward a family boundary also requires different analysis.

A small $p_V$ and the sign of $R$ distinguish relative fit under those
conditions. They do not establish that the preferred model fits well in an
absolute sense. A large $p_V$ leaves the comparison unresolved; it does not
establish equivalence. The statistic below uses the unpenalized log likelihoods,
not an AIC or BIC adjustment.

If $u$ was selected on the same data, the fixed-threshold reference theorem
does not automatically cover that selection. Report such p-values as
conditional approximations, or use an independent threshold choice or a
simulation design that repeats the selection procedure. Several pairwise
tests add a multiplicity issue; a collection of unadjusted p-values is not a
single calibrated family-selection decision.

[`fit_tail_model`](incerto-api-reference.md#api-fit-tail-model) supplies fitted
per-observation log likelihoods, and
[`likelihood_ratio_test`](incerto-api-reference.md#api-likelihood-ratio-test)
computes the comparison. This example fixes $u=1$ before sampling. The Pareto
generator has $\alpha=3$, so the exponential contrast has finite variance.
Both samples have 2500 observations; each comparison uses the entire sample.

:::{dropdown} Code

```python
import numpy as np
from xmath.incerto.powerlaw import (
    sample_powerlaw, fit_tail_model, likelihood_ratio_test,
)
rng = np.random.default_rng(350)
samples = {
    "Pareto": sample_powerlaw(3, 1, 2500, random_state=rng),
    "Exponential": 1 + rng.exponential(size=2500),
}
for name, x in samples.items():
    pl = fit_tail_model(x, 1, "powerlaw")
    exp = fit_tail_model(x, 1, "exponential")
    lr = likelihood_ratio_test(pl["loglik"], exp["loglik"])
    print(f"{name}: R={lr['R']:.2f}, Z={lr['normalized_R']:.2f}, p={lr['p_value']:.3g}")
```

**Output**

```text
Pareto: R=233.67, Z=4.31, p=1.6e-05
Exponential: R=-206.54, Z=-11.82, p=2.93e-32
```
:::

A separate lognormal simulation illustrates conditional fitting of every
alternative. We use the same realized tail for all five fits. Lognormal
likelihoods can be close to other families in a finite sample; this table
reports optimization outcomes and log likelihoods, without assigning a
common normal-reference test to all pairs.

:::{dropdown} Code

```python
import numpy as np
from xmath.incerto.powerlaw import fit_tail_model

x = np.random.default_rng(360).lognormal(1, 0.6, 1000)
models = ("powerlaw", "lognormal", "exponential",
          "stretched_exponential", "truncated_powerlaw")
for model in models:
    fit = fit_tail_model(x, 1, model)
    print(f"{model}: n={fit['n_tail']}, loglik={fit['loglikelihood']:.2f}")
    print(f"  status: {fit['status']}")
```

**Output**

```text
powerlaw: n=950, loglik=-1980.75
  status: ok
lognormal: n=950, loglik=-1720.13
  status: ok
exponential: n=950, loglik=-1741.61
  status: ok
stretched_exponential: n=950, loglik=-1723.98
  status: ok
truncated_powerlaw: n=950, loglik=-1720.72
  status: ok
```
:::

The alternative fitter currently supplies continuous densities. Exact discrete
power-law fits and bootstrap checks are available, but an integer-data
comparison needs separately normalized discrete alternatives. Treating a
continuous density as a probability at an integer is not that extension.
Numerical multi-start optimization also cannot certify a global maximum;
check the returned status, parameter boundaries, and sensitivity to the model.

(csn-nested-cutoff)=
## The nested cutoff comparison needs its own calibration

The pure power law is the $\lambda=0$ boundary of the cutoff family. If both
optimizations find their maxima, the larger family cannot have lower maximized
likelihood. With model 1 the pure law and model 2 the cutoff law,

$$
\begin{aligned}
R&=\ell_{\mathrm{PL}}-\ell_{\mathrm C}\le0,\\
T_{\mathrm{LR}}&=2(\ell_{\mathrm C}-\ell_{\mathrm{PL}})=-2R.
\end{aligned}
$$

The sign therefore gives no evidence by itself for a cutoff. Under an ordinary
regular interior restriction with one extra parameter, Wilks' reference is
$\chi_1^2$. Here $\lambda\ge0$ and the null sits at its boundary. With one
regular constrained direction and interior identifiable nuisance parameters,
a common reference is the mixture

$$
\tfrac12\delta_0+\tfrac12\chi_1^2,
$$

where $\delta_0$ is a point mass at zero. This boundary result is cited from
@self1987asymptotic, §3, Case 5. For a positive observed statistic its upper-tail
p-value is half the $\chi_1^2$ upper tail; at zero the inclusive upper-tail
probability is one.

That mixture is **not a universal cutoff test**. The cutoff score at the
Pareto null involves $X-\mathbb E X$; ordinary finite-information regularity
already fails when the null tail has $\alpha\le2$. An estimated threshold,
other boundary parameters, or failure to find a likelihood maximum introduces
further issues. A null simulation calibrated for the actual fitting procedure
is preferable when the reference-law assumptions are unsupported: simulate
under the null, repeat every fitted step, and compare refitted LR statistics.
The KS goodness-of-fit bootstrap above is a different statistic and does not
supply this nested LR p-value.

The helper's `nested=True` option returns $T_{\mathrm{LR}}$ and leaves its
p-value uncalibrated by default. Selecting `calibration="chi2"` or
`calibration="boundary"` explicitly asserts the corresponding conditions;
the software cannot verify them from two score arrays.

```python
import numpy as np
from xmath.incerto.powerlaw import fit_tail_model, likelihood_ratio_test

x = 2 + np.random.default_rng(370).exponential(size=600)
pure = fit_tail_model(x, 2, "powerlaw")
cutoff = fit_tail_model(x, 2, "truncated_powerlaw")
lr = likelihood_ratio_test(pure["loglik"], cutoff["loglik"], nested=True)
print(f"Nested LR statistic: {lr['statistic']:.2f}")
print(lr["status"])
```

**Output**

```text
Nested LR statistic: 55.81
nested statistic requires calibration
```

## What to report

Report the observational unit and independence assumptions, continuous or
integer support, candidate-threshold rule, selected threshold, retained count,
exponent convention, and uncertainty procedure. For goodness of fit include
the KS discrepancy, replicate count, seed, Monte Carlo precision, and failures.
For each comparison include the common tail, conditional likelihood family,
optimization status, signed $R$, and the reason a chosen p-value calibration
is applicable. Preserve inconclusive outcomes instead of turning every
comparison into a winner.

## See also

- [Power-law fitting](incerto-power-law-fitting.md) for MLEs, discrete likelihoods,
  threshold choice, and joint bootstrap uncertainty.
- [Hill estimator](incerto-hill-estimator.md) and
  [tail threshold selection](incerto-tail-threshold-selection.md).
- [Tail-class catalog](incerto-tail-class-catalog.md) for differences among
  power, lognormal, and stretched-exponential asymptotics.

## References

- Clauset, Shalizi, and Newman, *Power-law distributions in empirical data*
  @clauset2009powerlaw: §4.1 for the fitted-null bootstrap, §5 and appendix C
  for relative fit. We qualify reference distributions where boundary or
  infinite-moment cases need separate treatment.
- Vuong, *Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses*
  @vuong1989likelihood: fitted-model non-nested asymptotics; the full technical
  theorem is cited, not proved here.
- Self and Liang, *Asymptotic Properties of Maximum Likelihood Estimators and
  Likelihood Ratio Tests under Nonstandard Conditions* @self1987asymptotic:
  the regular boundary-mixture result, not a blanket claim about heavy-tail
  cutoff fits.
