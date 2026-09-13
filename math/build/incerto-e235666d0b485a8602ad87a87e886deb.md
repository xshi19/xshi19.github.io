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

These nine notes retain mathematical derivations and use static calculations;
they do not execute notebooks during the site build. Shared notation, further
threshold diagnostics, data examples, and reading guides remain planned.

The [mixture sample](./normix-conditioning-a-mixture.md) introduces a
complementary question: how conditioning specifies a model rather than counts
observations.
