# Concept Dependency DAG

This page tracks the current concept graph at a coarse level.  An arrow means
"the target page depends on the source page."

```{mermaid}
flowchart LR
  notation["Notation"]
  pareto["Pareto Distribution"]
  doublePareto["Double Pareto Distribution"]
  regularVariation["Regular Variation"]
  karamata["Karamata's Theorem"]
  paretoMoments["Pareto Moment Existence"]
  llnFailure["LLN Failure Under Infinite Mean"]
  maxToSum["Max-to-Sum Ratio"]
  hill["Hill Estimator"]
  evi["Extreme-Value Index"]
  pbdh["Pickands-Balkema-de Haan Theorem"]
  meanExcess["Mean Excess Function"]
  sp500["S&P 500 Tail Fit"]
  amazon["Amazon Book Rank"]
  llnExample["Pre-Asymptotic LLN Behavior"]
  dispersion["Dispersion Ratio"]
  normalMixture["Normal Mixture"]
  varianceGamma["Variance Gamma"]
  bodyShoulderTail["Body/Shoulder/Tail Diagnostics"]

  notation --> pareto
  notation --> regularVariation
  notation --> hill
  notation --> pbdh

  pareto --> doublePareto
  pareto --> paretoMoments
  pareto --> hill
  pareto --> meanExcess
  pareto --> amazon
  pareto --> llnExample

  regularVariation --> karamata
  regularVariation --> paretoMoments
  regularVariation --> llnFailure
  regularVariation --> hill
  regularVariation --> evi
  regularVariation --> pbdh

  karamata --> paretoMoments
  paretoMoments --> llnFailure
  llnFailure --> llnExample
  llnExample --> maxToSum

  evi --> hill
  pbdh --> meanExcess
  hill --> sp500
  meanExcess --> sp500
  pbdh --> sp500

  normalMixture --> bodyShoulderTail
  varianceGamma --> bodyShoulderTail
  bodyShoulderTail --> dispersion
```

## Reading The Graph

The graph is intentionally a wiki navigation aid, not a theorem dependency
proof.  Some pages depend on shared notation, while others depend on a result
or method.  Empirical pages depend on both the mathematical concept and the
data-cleaning code used by the example.

## Near-Term Missing Nodes

- Generalized Pareto distribution.
- Generalized extreme-value distribution.
- Subexponentiality and the one-big-jump principle.
- Declustering for dependent exceedances.
- Return levels and expected shortfall.

## Backlinks

- Used by: [External Reading Guides](../reading-guides/external/index.md) and
  the Phase 3 applied examples.

<!-- incerto-provenance:start -->
:::{div} incerto-provenance
**Provenance.** Source: `content/concepts/dependency-dag.md`. Last verified: 2026-05-24. Checked against cited sources, page proof or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
