# Embrechts et al. - Modelling Extremal Events

## What This Work Is Doing

Embrechts, Klueppelberg, and Mikosch build a rigorous probability and
statistics toolkit for rare events: [regular variation](../../concepts/theorems/regular-variation.md), subexponentiality,
extreme-value limits, insurance risk, point processes, dependence, and
financial applications.  For this wiki, the book is best used as the first
external bridge from Taleb-centered fat-tail intuition into standard extreme
value theory.

We map the book to concept pages.  We do not reproduce the book's
proofs or examples.

## Notation Differences

| Source emphasis | Wiki notation | Meaning |
| --- | --- | --- |
| Tail $\overline F$ and integrated tails | $\bar F$ | Survival function. |
| Extreme-value index sometimes implicit in domains | $\xi$ | Shape parameter for GEV/GPD limits. |
| Power-tail index often written through regular variation | $\alpha=1/\xi$ when $\xi>0$ | [Pareto-type](../../concepts/distributions/pareto.md) tail exponent. |
| Threshold $u$ and excess $X-u$ | $u$ and $F_u$ | Peaks-over-threshold notation. |

The canonical symbol table is [Notation](../../notation/index.md).

## Concept Map

| Book topic | Current wiki page | Notes |
| --- | --- | --- |
| Regular variation and slowly varying functions | [Regular Variation](../../concepts/theorems/regular-variation.md) | The shared language for Pareto-type tails. |
| Subexponentiality | [Subexponentiality](../../concepts/theorems/subexponentiality.md) | The one-big-jump class for heavy-tailed sums. |
| Karamata-style moment consequences | [Karamata's Theorem](../../concepts/theorems/karamata.md), [Pareto Moment Existence](../../concepts/theorems/pareto-moment-existence.md) | The bridge from tail exponent to moment finiteness. |
| Block maxima | [Generalized Extreme-Value Distribution](../../concepts/distributions/generalized-extreme-value.md) | The GEV limit family for maxima. |
| Peaks over threshold | [Pickands-Balkema-de Haan Theorem](../../concepts/theorems/pickands-balkema-de-haan.md) | Phase 3's main EVT theorem atom. |
| Threshold exceedances | [Generalized Pareto Distribution](../../concepts/distributions/generalized-pareto.md) | The GPD family for excesses. |
| Threshold diagnostics | [Mean Excess Function](../../concepts/theorems/mean-excess-function.md), [Tail Threshold Selection](../../concepts/methods/tail-threshold-selection.md) | Used before and after GPD modeling. |
| Tail-index estimation | [Hill Estimator](../../concepts/methods/hill-estimator.md), [Extreme-Value Index](../../concepts/methods/extreme-value-index.md) | Estimation needs threshold sensitivity, not a single magic number. |
| Empirical finance tails | [S&P 500 Tail Diagnostics](../../concepts/examples/sp500-tail.md) | A small applied bridge, with iid caveats. |

## Reading Route

Start with the regular-variation material before the statistical modeling
chapters.  In this wiki's terms, the route is:

1. [Regular Variation](../../concepts/theorems/regular-variation.md)
2. [Subexponentiality](../../concepts/theorems/subexponentiality.md)
3. [Karamata's Theorem](../../concepts/theorems/karamata.md)
4. [Generalized Extreme-Value Distribution](../../concepts/distributions/generalized-extreme-value.md)
5. [Pickands-Balkema-de Haan Theorem](../../concepts/theorems/pickands-balkema-de-haan.md)
6. [Generalized Pareto Distribution](../../concepts/distributions/generalized-pareto.md)
7. [Mean Excess Function](../../concepts/theorems/mean-excess-function.md)
8. [Tail Threshold Selection](../../concepts/methods/tail-threshold-selection.md)
9. [Hill Estimator](../../concepts/methods/hill-estimator.md)
10. [S&P 500 Tail Diagnostics](../../concepts/examples/sp500-tail.md)

This order keeps notation stable: first the tail class, then the moment
consequences, then the threshold approximation, then diagnostics and empirical
estimation.

## Planned Atoms

- Tail empirical process and point-process view.
- Declustering for dependent exceedances.
- Return levels and expected shortfall under GPD fits.

## Verification Notes

- We paraphrase the external source and link to concept pages rather
  than copying book text.
- The current Phase 3 bridge is intentionally narrow: univariate, right-tail,
  peaks-over-threshold examples only.
- Dependence and multivariate extremes remain planned, not silently absorbed
  into the iid examples.

## References

- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/reading-guides/external/embrechts-modelling-extremal-events.md`. Last verified: 2026-06-06. Checked against cited sources and current concept links.
:::
<!-- incerto-provenance:end -->
