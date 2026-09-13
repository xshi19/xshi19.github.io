---
title: Incerto / fat tails
---

Tail questions begin by specifying an event and the evidence available for it.
A fraction counted in a finite sample and a probability under a proposed model
are different quantities. Keeping both visible makes later estimation arguments
easier to assess.

Start with [counting exceedances](./incerto-counting-exceedances.md), an original
worked example with a short Python demo. It uses only a finite list and strict
inequalities; no moment or distributional assumption is needed for the count.

The first imported concept batch develops a path from a distributional model
to estimates of tail quantities:

1. [Pareto distribution](./incerto-pareto.md) specifies an exact power tail;
   [regular variation](./incerto-regular-variation.md) allows slowly varying
   corrections at high thresholds.
2. [Karamata's theorem](./incerto-karamata.md) connects tail shape to integrals.
   [Pareto moment existence](./incerto-pareto-moment-existence.md) proves the
   exact moment boundaries, and the
   [mean excess function](./incerto-mean-excess-function.md) describes
   overshoots beyond a threshold.
3. The [Hill estimator](./incerto-hill-estimator.md) estimates a positive tail
   index from log-excesses. The
   [extreme-value index](./incerto-extreme-value-index.md) places that estimate
   within the wider classification of extreme-value limits.
4. The [generalized Pareto distribution](./incerto-generalized-pareto.md)
   models threshold excesses, and
   [plug-in tail estimation](./incerto-plug-in-tail-estimation.md) explains how
   a fitted tail model enters a mean estimate, including its moment-boundary
   singularity and the contribution of the distribution body.

The second reading path connects extreme-value limits to diagnostics.

## Peaks over threshold & maxima

1. The [Pickands–Balkema–de Haan theorem](./incerto-pickands-balkema-de-haan.md)
   justifies the GPD excess model under a maximum-domain-of-attraction assumption.
2. The [generalized extreme-value distribution](./incerto-generalized-extreme-value.md)
   describes normalized block-maxima limits; the
   [Frechet law](./incerto-frechet.md) is its positive-shape member and connects
   those limits to regularly varying tails.
3. [Tail threshold selection](./incerto-tail-threshold-selection.md) makes the
   finite-sample tuning decision explicit through exceedance counts, shape,
   mean excess, and modified-scale stability.

## One-big-jump diagnostics

1. [Subexponentiality](./incerto-subexponentiality.md) explains why a large sum
   of a fixed number of nonnegative iid terms has the same leading tail as
   its maximum.
2. The [survival tail ratio](./incerto-survival-tail-ratio.md) compares two
   moderate extremes with one doubled extreme. The
   [max-to-sum ratio](./incerto-max-to-sum-ratio.md) measures the largest
   observation's share of a realized total. Their limits answer different
   questions and neither diagnostic alone proves an empirical tail class.
3. [Symmetric shifted double Pareto](./incerto-double-pareto.md) provides a
   two-sided power-tail model, with a clear distinction between symmetry and
   existence of the mean.
4. [Body, shoulders, and tails](./incerto-body-shoulder-tail.md) derives local
   density changes under variance mixing. Its curvature boundaries describe
   mixture geometry; choosing a Pareto threshold remains a separate task.

These eighteen imported notes retain mathematical arguments and use static
calculations; they do not execute notebooks during the site build. Shared
notation, further tail classes, data examples, and reading guides remain planned.

The [mixture sample](./normix-conditioning-a-mixture.md) introduces a
complementary question: how conditioning specifies a model rather than counts
observations.
