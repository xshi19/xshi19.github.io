---
options:
  concept:
    id: layoffs-tails
    type: empirical-example
    depends_on:
    - hill-estimator
    - tail-threshold-selection
    - power-law-fitting
    - power-law-goodness-of-fit
    tags:
    - empirical
    - tail-risk
    - employment
title: Tech layoffs — counts and workforce losses
---

A cut of 100 people and a cut of 50% of a workforce measure different things.
The first measures the number of jobs lost; the second measures the fraction
removed from a company. A large employer can dominate the first distribution
without making an unusually large proportional cut.

For **people laid off per event**, the empirical reading is **fat tail, tail
index about 1**. The [Hill estimator](incerto-hill-estimator.md) gives 1.04
using the largest 10% of observed counts and 1.01 using the largest 20%.
A fitted pure power law nevertheless fails the goodness-of-fit check below.
“Fat tail” here describes the observed upper range; these diagnostics do not
establish an asymptotic tail class.

## Data and the observed unit

The source is layoffs.fyi through the Kaggle mirror `swaptr/layoffs-2022`,
retrieved **2026-10-26** [@swaptr2026layoffs]. Events run from **2020-03-11 to
2026-10-22**. We pool countries, industries, and years in that snapshot.

We trim company names and group the 4,644 source rows by company and date.
For same-day rows we sum available counts and retain the largest reported
workforce fraction. An all-missing measurement stays missing. This gives
**4,621 company-date events**, of which **3,014 have counts**. Dates on different
days remain separate even for the same company. Exact repeated rows are
included in the same aggregation; this is a reproducible reporting rule, not
verification that every source row describes distinct jobs.

For percentages, 2,891 events have a reported fraction. We separate **377
reported 100% cuts**, exclude **one zero** from a positive-tail calculation,
and leave **1,730 missing fractions** unfilled. The second sample therefore
contains **2,513 events with a reported fraction strictly between 0 and 1**.
It is not restricted to events with an observed count. No inferred headcounts
or filled-in percentages enter either figure.

A company-date event is neither a unique firm nor an independently sampled
worker. Repeated employers, common economic conditions, and selective reporting
all affect inference from this pooled sample.

## Reading a Hill plot

Let $x_{(1)}\leq\cdots\leq x_{(n)}$ be the positive observations in ascending
order. We use the $k$ largest values and the next value as the threshold:

$$
u_k=x_{(n-k)}.
$$

The Hill estimate and its reciprocal are

$$
\widehat\xi_k=\frac{1}{k}\sum_{j=1}^{k}
\log\frac{x_{(n-j+1)}}{u_k},
$$

$$
\widehat\alpha_k=1/\widehat\xi_k.
$$

Here $\alpha$ is the survival-tail exponent, following the
[shared notation](https://xshi19.github.io/math/notation/); it is one less than
the density exponent used in some power-law software. The relation
$\xi=1/\alpha$ belongs to the positive regularly varying tail setting.
Computing the statistic does not verify that setting [@hill1975simple].

Increasing $k$ brings more observations into the calculation and lowers the
threshold. Very small $k$ uses little information; large $k$ can include the
body of the distribution. [Threshold selection](incerto-tail-threshold-selection.md)
explains this bias–variability tradeoff. Ties contribute zero log spacings;
we keep them rather than jittering the observations.

Both plots show $k$ from 25 to roughly one third of the available sample.
The shaded **95% pointwise bootstrap intervals** are the 2.5th and 97.5th
percentiles of 1,000 full-sample resamples, holding $k$ fixed. They describe
resampling variability under an iid approximation. They are not a simultaneous
band, do not account for employer clustering or missingness, and do not confer
guaranteed confidence coverage on an extreme-order statistic. The markers
identify the largest 10% and 20%, rounded to integer $k$.

## People laid off: an index near one

```{figure} ../assets/incerto/figures/layoffs-counts-hill.svg
:alt: Hill estimates for 3014 layoff counts, with pointwise bootstrap intervals and markers at k 301 and 603.

People laid off per company-date event. The dashed line marks $\alpha=1$;
it is a reference, not a fitted model.
```

The two marked estimates are close: **1.04 at $k=301$** and **1.01 at $k=603$**.
Their pointwise bootstrap intervals are **[0.94, 1.21]** and **[0.96, 1.13]**.
The curve lets us see the intervening variation instead of presenting a single
selected threshold as settled. Round-number counts make steps visible as the
threshold crosses tied values.

An index near one is a useful description of the severity of the observed
count tail. It also warns against treating an ordinary average as a stable
summary of exposure to large cuts. It does **not** establish an infinite
population mean. The [Pareto moment boundary](incerto-pareto-moment-existence.md)
is a statement about an exact unbounded model; actual layoffs are constrained
by workforces, and the pure model is rejected here.

## Percentages: use the loss of the retained workforce

Let $p$ denote the reported fraction of the workforce laid off. Define the
local log-loss variable

$$
L=-\log(1-p),\qquad 0\leq p<1.
$$

In this page **$L$ means workforce log loss**, not the slowly varying function
called $L$ in the notation canon. It is dimensionless, and all logarithms are
natural. Its inverse is

$$
p=1-e^{-L}.
$$

This is the lower-bound-zero, upper-bound-one case of Cirillo and Taleb's
bounded-to-unbounded transformation [@cirillo2015violent] (equation 1). The
coordinate maps $[0,1)$ onto $[0,\infty)$: $L=0$ at no loss, and $L$ diverges
as the retained fraction $1-p$ tends to zero. For example, 50%, 90%, and 99%
cuts give log losses of 0.69, 2.30, and 4.61. Thus the transformation separates
large proportional cuts that a bounded percentage scale compresses near 100%.

Removing the coordinate's endpoint does not produce a fat tail by itself.
For example, if a hypothetical fraction is uniform on $(0,1)$, then for
$\ell\geq0$,

$$
\begin{aligned}
&P(L>\ell)\\
&\quad=P(p>1-e^{-\ell})\\
&\quad=e^{-\ell}.
\end{aligned}
$$

That transformed variable is exponential. This calculation establishes only
that the transformation does not determine a tail class; the layoffs figures
remain empirical diagnostics.

**Endpoint rule:** a reported $p=1$ gives an infinite log loss. We count those
377 events separately and exclude them from the finite-$L$ fit. We do not
replace 100% by 99% or by $1-\varepsilon$. A reported full-workforce cut need
not establish a legal company closure. A zero fraction maps to zero and is
excluded because Hill uses positive observations. Missing percentages are
excluded without imputation. The figure therefore describes reported partial
cuts, conditional on $0<p<1$; it says nothing about the probability of a full
cut among all companies.

Use [`workforce_log_loss`](incerto-api-reference.md#api-workforce-log-loss).

```python
from xmath.incerto.datasets import workforce_log_loss

losses = workforce_log_loss([0, 0.5, 0.9, 0.99])
print("Log loss: [" + ", ".join(f"{v:.2f}" for v in losses) + "]")
```

**Output**

```text
Log loss: [0.00, 0.69, 2.30, 4.61]
```

```{figure} ../assets/incerto/figures/layoffs-logloss-hill.svg
:alt: Hill estimates for 2513 reported partial-cut log losses, with pointwise bootstrap intervals and markers at k 251 and 503.

Reported workforce fractions after the log-loss transform. Full-workforce
cuts, zeros, and missing fractions are excluded by the stated rule.
```

The marked estimates are **1.98 at $k=251$** and **2.28 at $k=503$**, with
pointwise intervals **[1.73, 2.45]** and **[1.92, 2.50]**. They are
larger and more threshold-sensitive than the count estimates. The reported
fractions are rounded, with many ties; a nonlinear transform preserves those
ties. The largest included fraction is 98%, so the largest observed finite
$L$ is only 3.91. These estimates describe that visible range. They do not
establish a Pareto tail for $L$, and they are not exponents for raw percentages.
The “index about 1” conclusion applies to counts.

## Does an exact power-law tail fit?

We add the [Clauset–Shalizi–Newman fit](incerto-power-law-fitting.md)
[@clauset2009powerlaw] to distinguish an exponent estimate from a model check.
For each distinct candidate threshold, fit a continuous Pareto tail to all
values at or above it; retain candidates with at least 50 observations and
choose the smallest two-sided empirical KS distance. Threshold ties are
included. Write $u$ for the selected cutoff (the code calls it `xmin`).

| Sample | $u$ | Tail events | $\widehat\alpha$ | KS |
| --- | ---: | ---: | ---: | ---: |
| Counts | 230 people | 681 | 1.04 | 0.053 |
| Reported log loss | 0.371 | 387 | 1.89 | 0.114 |

The second threshold corresponds to a 31% workforce cut. The conditional
fixed-threshold standard errors are 0.04 and 0.10 respectively. They omit
threshold selection and model error; they are not the Hill intervals.
The CSN inclusive tail count also differs from Hill's fixed $k$ when values
at the cutoff tie.

The [goodness-of-fit bootstrap](incerto-power-law-goodness-of-fit.md)
resamples the empirical body and generates the fitted continuous tail, then
refits both cutoff and exponent in each replicate. **For each sample, zero of
1,000 replicates had KS distance at least as large as the observed distance.**
The recorded estimate is $\widehat p_{\mathrm{GOF}}=0$; this is a finite Monte
Carlo count, not a claim of zero probability. Both fits are rejected by the
usual 0.1 CSN screening rule.

This test uses continuous tails for rounded observations. Such a null does
not reproduce the piles at common counts and percentages, which can increase
the observed KS distance relative to simulated samples. A discrete or explicit
rounding model would answer a different question. Dependence is another reason
to treat the nominal calibration cautiously. The present result rejects the
specified pure continuous power-law description; it does not reject every
fat-tailed alternative.

We stop at this comparison. Additional distributions can fit similar finite
ranges, and a likelihood ranking would not repair a failed absolute fit.
Company-size estimates would introduce a separate headcount and imputation
problem. Neither is needed for the two observables examined here.

## Reproduce the calculation

No layoffs rows are distributed with this page. Figures and aggregate outputs
are generated beforehand; the site build neither downloads nor reads the
local CSV. Install the Kaggle CLI separately, configure your own credentials,
and run:

```sh
UV_NO_SYNC=1 uv run python scripts/download_layoffs.py
UV_NO_SYNC=1 uv run python demos/incerto/layoffs_tails.py --data data/incerto/cache/layoffs/layoffs.csv
```

Downloads remain under the ignored cache directory. A later Kaggle download
may have changed; the numbers here use the dated snapshot above. The loader
accepts both the raw mirror columns and the research-clean `n`/`frac` columns.
[`load_layoffs`](incerto-api-reference.md#api-load-layoffs) calls
[`clean_layoffs`](incerto-api-reference.md#api-clean-layoffs) to apply the
company-date rule. Set `INCERTO_LAYOFFS_PATH` to an existing CSV for the example
below; without that variable, the loader uses the default cache.

Use [`hill_alpha_estimator`](incerto-api-reference.md#api-hill-alpha-estimator),
[`select_xmin`](incerto-api-reference.md#api-select-xmin), and
[`workforce_log_loss`](incerto-api-reference.md#api-workforce-log-loss).

:::{dropdown} Code: counts, log loss, and selected thresholds

```python
import os
from xmath.incerto.datasets import load_layoffs, workforce_log_loss
from xmath.incerto.estimators import hill_alpha_estimator
from xmath.incerto.powerlaw import select_xmin

data = load_layoffs(os.environ.get("INCERTO_LAYOFFS_PATH"))
p = data.fraction
samples = {"Counts": data.count[data.count > 0],
           "Log loss": workforce_log_loss(p[(p > 0) & (p < 1)])}
print(f"Events: {data.count.size}")
for name, x in samples.items():
    print(f"{name}: {len(x)} observations")
    for fraction in (0.1, 0.2):
        k = round(fraction * len(x))
        print(f"  Hill k={k}: {hill_alpha_estimator(x, k):.2f}")
    fit = select_xmin(x, min_tail=50)
    print(f"  CSN cutoff={fit['xmin']:.3f}, tail={fit['n_tail']}, alpha={fit['alpha']:.2f}")
```

**Output**

```text
Events: 4621
Counts: 3014 observations
  Hill k=301: 1.04
  Hill k=603: 1.01
  CSN cutoff=230.000, tail=681, alpha=1.04
Log loss: 2513 observations
  Hill k=251: 1.98
  Hill k=503: 2.28
  CSN cutoff=0.371, tail=387, alpha=1.89
```
:::

The figure script uses
[`hill_stability_bootstrap`](incerto-api-reference.md#api-hill-stability-bootstrap)
for the bands, [`powerlaw_gof_bootstrap`](incerto-api-reference.md#api-powerlaw-gof-bootstrap)
for the refitted model checks, and
[`powerlaw_standard_error`](incerto-api-reference.md#api-powerlaw-standard-error)
for the conditional errors. It draws through the shared
[`set_theme`](incerto-api-reference.md#api-set-theme),
[`vertical_figure`](incerto-api-reference.md#api-vertical-figure), and
[`savefig`](incerto-api-reference.md#api-savefig) helpers.

The seeds are 20261029 for counts and 20261030 for reported log loss, with
the generator reset for each method. Point estimates
match the reference refit. Interval endpoints can differ slightly because
that refit consumes one random stream across five analyses; this page shares
a resample across all $k$ within each plotted curve. No infinite bootstrap
estimates occurred on the plotted grid.

The recorded [aggregate output](../assets/incerto/outputs/layoffs-tails.txt)
and [summary JSON](../assets/incerto/outputs/layoffs-tails.json) retain sample
sizes, exclusions, the four reported Hill estimates and intervals, and the
CSN and GOF summaries. Full Hill curves and order statistics are not stored
in the JSON. To print the saved summary without the CSV:

```sh
UV_NO_SYNC=1 uv run python demos/incerto/layoffs_tails.py --recorded assets/incerto/outputs/layoffs-tails.json
```

Regenerating the SVGs requires the local CSV and the `--data` command above.
The demo keeps the plotting curves in memory and writes only the summaries.

## References and related examples

- @swaptr2026layoffs, layoffs.fyi via the Kaggle mirror; snapshot retrieved
  2026-10-26.
- @hill1975simple, tail-index estimation.
- @clauset2009powerlaw, fitting and testing power-law tails.
- @cirillo2015violent, the bounded-to-unbounded coordinate transformation.
- [NYC dog-name frequencies](incerto-nyc-dog-name-tail.md) show why a broad
  rank-size pattern can coexist with unstable tail estimates.
- [S&P 500 tail diagnostics](incerto-sp500-tail.md) apply loss-based diagnostics
  to another local-data example.
