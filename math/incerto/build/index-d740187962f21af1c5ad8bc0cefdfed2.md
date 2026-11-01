---
title: Incerto / fat tails
---

Tail questions begin by specifying an event and the evidence available for it.
A fraction counted in a finite sample and a probability under a proposed model
are different quantities. Keeping both visible makes later estimation arguments
easier to assess.

Start with [counting exceedances](../incerto-counting-exceedances.md), a
worked example with a short Python demo. It uses only a finite list and strict
inequalities; no moment or distributional assumption is needed for the count.

The following reading path moves from a distributional model
to estimates of tail quantities:

1. [Pareto distribution](../incerto-pareto.md) specifies an exact power tail;
   [regular variation](../incerto-regular-variation.md) allows slowly varying
   corrections at high thresholds.
2. [Karamata's theorem](../incerto-karamata.md) connects tail shape to integrals.
   [Pareto moment existence](../incerto-pareto-moment-existence.md) proves the
   exact moment boundaries, and the
   [mean excess function](../incerto-mean-excess-function.md) describes
   overshoots beyond a threshold.
3. The [Hill estimator](../incerto-hill-estimator.md) estimates a positive tail
   index from log-excesses. The
   [extreme-value index](../incerto-extreme-value-index.md) places that estimate
   within the wider classification of extreme-value limits.
4. The [generalized Pareto distribution](../incerto-generalized-pareto.md)
   models threshold excesses, and
   [plug-in tail estimation](../incerto-plug-in-tail-estimation.md) explains how
   a fitted tail model enters a mean estimate, including its moment-boundary
   singularity and the contribution of the distribution body.

[Power-law fitting](../incerto-power-law-fitting.md) develops the continuous
and discrete likelihoods and a KS threshold search. Continue with
[goodness of fit and alternatives](../incerto-power-law-goodness-of-fit.md)
for bootstrap calibration and comparisons on a common tail. The
[tech layoffs example](../incerto-layoffs-tails.md) applies those methods to
counts and workforce losses, with two Hill plots and explicit endpoint handling.

The second reading path connects extreme-value limits to diagnostics.

## Peaks over threshold & maxima

1. The [Pickands–Balkema–de Haan theorem](../incerto-pickands-balkema-de-haan.md)
   justifies the GPD excess model under a maximum-domain-of-attraction assumption.
2. The [generalized extreme-value distribution](../incerto-generalized-extreme-value.md)
   describes normalized block-maxima limits; the
   [Frechet law](../incerto-frechet.md) is its positive-shape member and connects
   those limits to regularly varying tails.
3. [Tail threshold selection](../incerto-tail-threshold-selection.md) makes the
   finite-sample tuning decision explicit through exceedance counts, shape,
   mean excess, and modified-scale stability.

## One-big-jump diagnostics

1. [Subexponentiality](../incerto-subexponentiality.md) explains why a large sum
   of a fixed number of nonnegative iid terms has the same leading tail as
   its maximum.
2. The [survival tail ratio](../incerto-survival-tail-ratio.md) compares two
   moderate extremes with one doubled extreme. The
   [max-to-sum ratio](../incerto-max-to-sum-ratio.md) measures the largest
   observation's share of a realized total. Their limits answer different
   questions and neither diagnostic alone proves an empirical tail class.
3. [Symmetric shifted double Pareto](../incerto-double-pareto.md) provides a
   two-sided power-tail model, with a clear distinction between symmetry and
   existence of the mean.
4. [Body, shoulders, and tails](../incerto-body-shoulder-tail.md) derives local
   density changes under variance mixing. Its curvature boundaries describe
   mixture geometry; choosing a Pareto threshold remains a separate task.
5. [Iso-density tail geometry](../incerto-iso-density-tail-geometry.md) compares
   equal and axial splits of a large sum under normal and Cauchy densities.
   Density at a point and integrated tail probability remain distinct.

The third reading path contrasts light-tailed concentration with stable sums
and infinite-mean averages.

## Sums, stable limits, and thin-tail contrast

1. The [Cramér exponential-moment condition](../incerto-cramer-condition.md)
   gives a Chernoff bound and separates positive exponential moments from
   ordinary moment existence.
2. The [generalized central limit theorem](../incerto-generalized-central-limit-theorem.md)
   describes stable limits of normalized iid sums, including tail balance and
   centering at the index-one boundary.
3. [LLN failure under infinite mean](../incerto-lln-failure.md) proves that
   averages of nonnegative iid variables with infinite mean diverge almost
   surely. Capped Pareto means make the truncation argument explicit.

The [chapter 7 guide](../incerto-scoft-ch7.md) adds
[Cauchy running statistics](../incerto-cauchy-running.md) and
[time versus ensemble averages](../incerto-ergodicity.md), with separate
statements for integrability, ergodicity and multiplicative growth.

## Tail catalog & indexes

The [tail class catalog](../incerto-tail-class-catalog.md) compares regular
variation and subexponentiality for Pareto, lognormal, Weibull, exponential,
and gamma examples. Browse the concepts through the
[theorem](../incerto-theorem-concepts.md), [method](../incerto-method-concepts.md),
and [distribution](../incerto-distribution-concepts.md) indexes. Mixture and
variance-gamma theory links to the existing Normix notes.

[Maximum entropy](../incerto-maximum-entropy.md) derives a Pareto model from
a fixed logarithmic moment, with explicit support and reference measure.

## Choose an entry path

For an introduction with basic calculus and probability, read
[Pareto](../incerto-pareto.md), [moment existence](../incerto-pareto-moment-existence.md),
[running means](../incerto-lln-preasymptotic.md), and
[max-to-sum ratios](../incerto-max-to-sum-ratio.md), then
[regular variation](../incerto-regular-variation.md). These pages separate
finite moments from reliable finite-sample summaries.

To analyze data:

1. Begin with the [NYC dog-name example](../incerto-nyc-dog-name-tail.md).
   Specify the observed unit and which tail is being studied.
2. Compare empirical survival with candidate models, then use
   [threshold selection](../incerto-tail-threshold-selection.md).
   [Body/shoulder geometry](../incerto-body-shoulder-tail.md) concerns mixture
   densities; its curvature boundaries do not select empirical thresholds.
3. Compare [Hill](../incerto-hill-estimator.md) and
   [mean excess](../incerto-mean-excess-function.md), reporting threshold
   sensitivity and the example's discrete-count and dependence caveats.
4. The [S&P method](../incerto-sp500-tail.md) extends the workflow to daily
   losses when an authorized local data cache is available.

## Evidence and exposure

1. [Inference asymmetry](../incerto-inference-asymmetry.md) constructs two
   laws with close finite-sample behavior and very different means.
2. [Asymmetric payoffs](../incerto-asymmetric-payoffs.md) separates gain
   frequency, expected payoff and the effect of curvature.
3. [Ruin](../incerto-ruin.md) gives finite-horizon survival formulas and
   explains why a growing expectation can coexist with terminating paths.
4. [Decentralized exposure](../incerto-decentralization.md) distinguishes
   nonlinear damage reduction from independence between risks.

## Mixtures and moment estimation

1. [Variance mixtures](../incerto-mixture-uncertainty.md) hold variance fixed
   while changing kurtosis, or change the moment regime by mixing precision.
2. [Body and shoulder geometry](../incerto-body-shoulder-tail.md) separates
   variance curvature, scale curvature, and finite-mixture crossings.
3. [Moment estimation](../incerto-moment-estimation.md) distinguishes a
   population moment from the moments and consistency of its estimator.
4. [Kappa](../incerto-kappa.md) measures finite-sum concentration and supports
   [sample-size comparisons](../incerto-moment-estimation.md#moment-estimation-sufficiency)
   under an explicit absolute-error criterion.

## Dependence in several dimensions

[Multivariate tails](../incerto-multivariate-tails.md) builds a Student vector
from a shared scale, derives its mutual information, and separates finite
sample statistics from population moments. [Density geometry](../incerto-iso-density-tail-geometry.md)
compares identical marginals under different joint laws.

## Return diagnostics and statistical pipelines

[S&P diagnostics](../incerto-sp500.md) separates aggregation, drawdowns,
moment concentration and record counts. [Tail assumptions in statistical
pipelines](../incerto-ml-econometrics.md) compares frozen risk estimators
and derives hinge weights from slopes. Empirical S&P results require an
authorized local cache; the published figures use specified synthetic models.

## Reading and reference

- [Empirical examples](../incerto-empirical-example-concepts.md): two committed
  datasets, simulations, the layoffs study with recorded figures, and the S&P
  method with explicit data availability.
- [SCoFT guide](../incerto-scoft-intro.md):
  [Chapter 1 — Prologue](../incerto-scoft-ch1.md),
  [Chapter 2 — Glossary, definitions, and notations](../incerto-scoft-ch2.md),
  [Chapter 3 — Non-technical overview](../incerto-scoft-ch3.md),
  [Chapter 4 — Finite-moment fat tails](../incerto-scoft-ch4.md),
  [Chapter 5 — Subexponentials and power laws](../incerto-scoft-ch5.md),
  [Chapter 6 — Thick tails in higher dimensions](../incerto-scoft-ch6.md),
  [Chapter 7 — Averages and limit laws](../incerto-scoft-ch7.md),
  [Chapter 8 — How much averaging helps](../incerto-scoft-ch8.md),
  [Chapter 9 — Maxima and hidden tails](../incerto-scoft-ch9.md),
  [Chapter 10 — Diagnosing returns](../incerto-scoft-ch10.md),
  [Chapter 11 — Calibration and payoffs](../incerto-scoft-ch11.md), and
  [Chapter 12 — Point forecasts and tail risk](../incerto-scoft-ch12.md).
- [External guides](../incerto-external-reading.md), including
  [Embrechts, Klüppelberg, and Mikosch](../incerto-reading-embrechts.md).
- [Concept graph](../incerto-dependency-dag.md), with a plain linked fallback.
- [Incerto glossary](../incerto-glossary.md),
  [shared notation](https://xshi19.github.io/math/notation/),
  [Python API](../incerto-api-reference.md), and
  [中文入口](../incerto-chinese-entry.md).

Figures and calculations are static. Reproduction scripts run separately from
the site build; neither notebooks nor data fetches execute during publication.

The [mixture sample](https://xshi19.github.io/math/normix-theory/normix-conditioning-a-mixture/) introduces a
complementary question: how conditioning specifies a model rather than counts
observations.

[The tail beyond the sample maximum](../incerto-hidden-tail.md) separates
missing probability, hidden moment contributions, and empirical CDF accuracy.

## Calibration and forecast losses

[Probability calibration](../incerto-calibration-payoff.md) separates event
frequencies, conditional severity, and the sampling distribution of a score.
[Point forecasts](../incerto-point-forecast.md) connect amount losses to their
moment requirements. The [chapter 11 guide](../incerto-scoft-ch11.md) maps
these calculations to the source and links the model figures.
The [chapter 12 guide](../incerto-scoft-ch12.md) extends the same note to
mean-error scaling, score comparisons, predictive ranges and action-dependent
pandemic forecasts, with synthetic bounded-loss examples.
