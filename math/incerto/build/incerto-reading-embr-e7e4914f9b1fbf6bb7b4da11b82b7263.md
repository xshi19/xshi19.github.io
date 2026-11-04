---
title: Embrechts et al. - Modelling Extremal Events
---

## What This Work Is Doing

Embrechts, Klueppelberg, and Mikosch build a rigorous probability and
statistics toolkit for rare events: [regular variation](incerto-regular-variation.md), subexponentiality,
extreme-value limits, insurance risk, point processes, dependence, and
financial applications.  For this wiki, the book is best used as the first
external bridge from Taleb-centered fat-tail intuition into standard extreme
value theory.

We map the book to concept pages.  We do not reproduce the book's
proofs or examples.

## Notation Differences

| Source emphasis | Shared notation | Meaning |
| --- | --- | --- |
| Tail $\overline F$ and integrated tails | $\bar F$ | Survival function. |
| Extreme-value index sometimes implicit in domains | $\xi$ | Shape parameter for GEV/GPD limits. |
| Power-tail index often written through regular variation | $\alpha=1/\xi$ when $\xi>0$ | [Pareto-type](incerto-pareto.md) tail exponent. |
| Threshold $u$ and excess $X-u$ | $u$ and $F_u$ | Peaks-over-threshold notation. |

The canonical symbol table is [Notation](https://xshi19.github.io/math/notation/).

## Concept Map

| Book topic | Current wiki page | Notes |
| --- | --- | --- |
| Regular variation and slowly varying functions | [Regular Variation](incerto-regular-variation.md) | The shared language for Pareto-type tails. |
| Subexponentiality | [Subexponentiality](incerto-subexponentiality.md) | The one-big-jump class for heavy-tailed sums. |
| Karamata-style moment consequences | [Karamata's Theorem](incerto-karamata.md), [Pareto Moment Existence](incerto-pareto-moment-existence.md) | The bridge from tail exponent to moment finiteness. |
| Block maxima | [Generalized Extreme-Value Distribution](incerto-generalized-extreme-value.md) | The GEV limit family for maxima. |
| Peaks over threshold | [Pickands-Balkema-de Haan Theorem](incerto-pickands-balkema-de-haan.md) | The asymptotic justification for threshold excess models. |
| Threshold exceedances | [Generalized Pareto Distribution](incerto-generalized-pareto.md) | The GPD family for excesses. |
| Threshold diagnostics | [Mean Excess Function](incerto-mean-excess-function.md), [Tail Threshold Selection](incerto-tail-threshold-selection.md) | Used before and after GPD modeling. |
| Tail-index estimation | [Hill Estimator](incerto-hill-estimator.md), [Extreme-Value Index](incerto-extreme-value-index.md) | Estimation needs threshold sensitivity, not a single magic number. |
| Empirical finance tails | [S&P 500 Tail Diagnostics](incerto-sp500-tail.md) | A small applied bridge, with iid caveats. |

## Reading Route

Start with the regular-variation material before the statistical modeling
chapters.  In this wiki's terms, the route is:

1. [Regular Variation](incerto-regular-variation.md)
2. [Subexponentiality](incerto-subexponentiality.md)
3. [Karamata's Theorem](incerto-karamata.md)
4. [Generalized Extreme-Value Distribution](incerto-generalized-extreme-value.md)
5. [Pickands-Balkema-de Haan Theorem](incerto-pickands-balkema-de-haan.md)
6. [Generalized Pareto Distribution](incerto-generalized-pareto.md)
7. [Mean Excess Function](incerto-mean-excess-function.md)
8. [Tail Threshold Selection](incerto-tail-threshold-selection.md)
9. [Hill Estimator](incerto-hill-estimator.md)
10. [S&P 500 Tail Diagnostics](incerto-sp500-tail.md)

This order keeps notation stable: first the tail class, then the moment
consequences, then the threshold approximation, then diagnostics and empirical
estimation.

## Topics awaiting dedicated notes

- Tail empirical process and point-process view.
- Declustering for dependent exceedances.
- Return levels and expected shortfall under GPD fits.

## Scope and limitations

- We paraphrase the external source and link to concept pages rather
  than copying book text.
- The current applied bridge is intentionally narrow: univariate, right-tail,
  peaks-over-threshold examples only.
- Dependence and multivariate extremes remain planned, not silently absorbed
  into the iid examples.

## References

- @embrechts1997modelling, *Modelling Extremal Events*.
- @resnick2007heavy, *Heavy-Tail Phenomena*.
