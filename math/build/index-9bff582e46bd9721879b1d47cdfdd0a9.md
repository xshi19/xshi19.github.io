---
title: Mathematical notes
---

These notes connect mathematical arguments to small computations that can be
inspected and rerun. Each track has its own questions; assumptions and notation
should remain explicit when an idea crosses between tracks.

- [Incerto / fat tails](./incerto.md) starts with a finite-sample counting
  example, before making any claim about a population tail.
- [Information Geometry](./information-geometry.md) develops an entry path from
  [Euclidean geometry](./information-geometry-euclidean-to-manifold.md) and
  [exponential families](./information-geometry-exponential-families.md) to
  [Fisher geometry](./information-geometry-fisher-vs-l2.md), EM, and duality.
- [Normix theory](./normix-theory.md) starts with conditioning a mixture and links
  to thirteen notes on GIG/GH distributions, mixtures, exponential families,
  EM, factor analysis, shrinkage, and information quantities,
  with implementation links to the independently maintained package.

The original entry notes and worked examples are joined by
[twenty-six adapted Incerto concept pages](./incerto.md) and
[thirteen Normix theory notes](./normix-theory.md), each carrying its source notice.
Further Incerto and Normix imports remain planned.
Numerical examples illustrate stated calculations; they do not establish
asymptotic theorems. No Lean statements are claimed to be checked here.
