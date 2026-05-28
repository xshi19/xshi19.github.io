# Embrechts et al. - Modelling Extremal Events

## What This Work Is Doing

Embrechts, Klueppelberg, and Mikosch build a rigorous probability and
statistics toolkit for rare events: regular variation, subexponentiality,
extreme-value limits, insurance risk, point processes, dependence, and
financial applications.  For this wiki, the book is best used as the first
external bridge from Taleb-centered fat-tail intuition into standard extreme
value theory.

This guide maps the book to concept pages.  It does not reproduce the book's
proofs or examples.

## Notation Differences

| Source emphasis | Wiki notation | Meaning |
| --- | --- | --- |
| Tail $\overline F$ and integrated tails | $\bar F$ | Survival function. |
| Extreme-value index sometimes implicit in domains | $\xi$ | Shape parameter for GEV/GPD limits. |
| Power-tail index often written through regular variation | $\alpha=1/\xi$ when $\xi>0$ | Pareto-type tail exponent. |
| Threshold $u$ and excess $X-u$ | $u$ and $F_u$ | Peaks-over-threshold notation. |

The canonical symbol table is [Notation](../../notation/index.md).

## Concept Map

| Book topic | Current wiki page | Notes |
| --- | --- | --- |
| Regular variation and slowly varying functions | [Regular Variation](../../concepts/theorems/regular-variation.md) | The shared language for Pareto-type tails. |
| Karamata-style moment consequences | [Karamata's Theorem](../../concepts/theorems/karamata.md), [Pareto Moment Existence](../../concepts/theorems/pareto-moment-existence.md) | The bridge from tail exponent to moment existence. |
| Peaks over threshold | [Pickands-Balkema-de Haan Theorem](../../concepts/theorems/pickands-balkema-de-haan.md) | Phase 3's main EVT theorem atom. |
| Threshold diagnostics | [Mean Excess Function](../../concepts/theorems/mean-excess-function.md) | Used before and after GPD modeling. |
| Tail-index estimation | [Hill Estimator](../../concepts/methods/hill-estimator.md), [Extreme-Value Index](../../concepts/methods/extreme-value-index.md) | Estimation needs threshold sensitivity, not a single magic number. |
| Empirical finance tails | [S&P 500 Tail Fit](../../concepts/examples/sp500-tail.md) | A small applied bridge, with iid caveats. |

## Reading Route

Start with the regular-variation material before the statistical modeling
chapters.  In this wiki's terms, the route is:

1. [Regular Variation](../../concepts/theorems/regular-variation.md)
2. [Karamata's Theorem](../../concepts/theorems/karamata.md)
3. [Pickands-Balkema-de Haan Theorem](../../concepts/theorems/pickands-balkema-de-haan.md)
4. [Mean Excess Function](../../concepts/theorems/mean-excess-function.md)
5. [Hill Estimator](../../concepts/methods/hill-estimator.md)
6. [S&P 500 Tail Fit](../../concepts/examples/sp500-tail.md)

This order keeps notation stable: first the tail class, then the moment
consequences, then the threshold approximation, then diagnostics and empirical
estimation.

## Planned Atoms

- Generalized extreme-value distribution.
- Generalized Pareto distribution as its own distribution page.
- Subexponential one-big-jump principle.
- Tail empirical process and point-process view.
- Declustering for dependent exceedances.
- Return levels and expected shortfall under GPD fits.

## Verification Notes

- This guide paraphrases the external source and links to concept pages rather
  than copying book text.
- The current Phase 3 bridge is intentionally narrow: univariate, right-tail,
  peaks-over-threshold examples only.
- Dependence and multivariate extremes remain planned, not silently absorbed
  into the iid examples.

## References

- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [@embrechts1997modelling].
- Resnick, *Heavy-Tail Phenomena* [@resnick2007heavy].
