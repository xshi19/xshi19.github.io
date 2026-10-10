# Concept Dependency DAG

The prerequisite layout gives an introductory reading order. The knowledge
layout includes proof dependencies, mathematical generalizations, applications,
and cross-references. These relationships need not share a reading order.

<!-- Use a site-root path so MyST prefixes BASE_URL correctly. -->
[Browse all concepts as a plain linked list](/assets/graphs/concept-list.html).
The list works without JavaScript. In the interactive view, Tab to a node and
press Enter or Space to select it, then Tab to its page link.

<iframe
  class="incerto-concept-graph-frame"
  src="../assets/graphs/concept-graph.html"
  title="Interactive Incerto Wiki concept graph"
></iframe>

## Reading The Graph

The graph is a navigation aid. Prerequisite arrows point from background to
pages that need it; proof arrows point from a supporting result to the page
using it. A generalization arrow points from a special case to the broader
concept. Application arrows point toward an application. References are ordinary
links, and related links are explicitly selected onward reading.

Only prerequisite edges must be acyclic. The pilot pages use the explicit
`prerequisites`, `proof_depends_on`, `special_case_of`, and `applications`
metadata fields. Existing `depends_on` lists remain compatible prerequisite
inputs while the remaining pages are reviewed. They are not a formal proof
certificate. Shared [notation](../notation/index.md) remains outside the graph.

## Near-Term Missing Nodes

- Declustering for dependent exceedances.
- Return levels and expected shortfall.
- Tail empirical process and point-process views.

## Backlinks

- Used by: [External Reading Guides](../reading-guides/external/index.md) and
  the Phase 3 applied examples.

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/dependency-dag.md`. Last verified: 2026-06-06. Checked against cited sources, page support or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
