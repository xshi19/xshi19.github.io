---
title: Incerto / fat tails
---

Tail questions begin by specifying an event and the evidence available for it.
A fraction counted in a finite sample and a probability under a proposed model
are different quantities. Keeping both visible makes later estimation arguments
easier to assess.

Start with [counting exceedances](../incerto-counting-exceedances.md), an original
worked example with a short Python demo. It uses only a finite list and strict
inequalities; no moment or distributional assumption is needed for the count.

The first imported concept batch develops a path from a distributional model
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

## Tail catalog & indexes

The [tail class catalog](../incerto-tail-class-catalog.md) compares regular
variation and subexponentiality for Pareto, lognormal, Weibull, exponential,
and gamma examples. Browse all imported concepts through the
[theorem](../incerto-theorem-concepts.md), [method](../incerto-method-concepts.md),
and [distribution](../incerto-distribution-concepts.md) indexes. Mixture and
variance-gamma theory links to the existing Normix notes.

These twenty-six concept imports comprise twenty-three body notes and three
indexes, alongside the original counting-exceedances example and this track hub.
The notes retain mathematical arguments and use static calculations; they do
not execute notebooks during the site build. Use the [shared notation](https://xshi19.github.io/math/notation/) canon for overlapping symbols. Data examples, reading guides, and dependency-graph navigation remain planned.

The [mixture sample](https://xshi19.github.io/math/normix-theory/normix-conditioning-a-mixture/) introduces a
complementary question: how conditioning specifies a model rather than counts
observations.
