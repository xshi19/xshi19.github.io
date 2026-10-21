---
title: Concept Dependency DAG
---

The prerequisite layout gives an introductory reading order. The knowledge
layout includes proof dependencies, mathematical generalizations, applications,
and cross-references. These relationships need not share a reading order.

<!-- Use a site-root path so MyST prefixes BASE_URL correctly. -->
[Browse all concepts as a plain linked list](https://xshi19.github.io/math/incerto/assets/graphs/concept-list.html).
The list works without JavaScript. In the interactive view, Tab to a node and
press Enter or Space to select it, then Tab to its page link.

<iframe
  style="width:100%;height:760px;border:0"
  loading="lazy"
  src="../assets/graphs/concept-graph.html"
  title="Interactive Incerto Wiki concept graph"
></iframe>

## Reading The Graph

The graph is a navigation aid. Prerequisite arrows point from background to
pages that need it; proof arrows point from a supporting result to the page
using it. A generalization arrow points from a special case to the broader
concept. Application arrows point toward an application. References are ordinary
links, and related links are explicitly selected onward reading.

Only prerequisite edges must be acyclic. The graph distinguishes reading
prerequisites from proof dependencies, generalizations, and applications.
An arrow is a navigation link, not a certificate of a formal proof.
Shared [notation](https://xshi19.github.io/math/notation/) remains outside the graph.

The graph includes Incerto concept notes with frontmatter metadata. Mixture
and variance-gamma background links to the Normix subsite in the notes; those
pages are outside this graph. Reading guides and reference indexes are also
excluded from the prerequisite relation.

## Near-Term Missing Nodes

- Declustering for dependent exceedances.
- Return levels and expected shortfall.
- Tail empirical process and point-process views.

## Backlinks

- Used by: [External Reading Guides](incerto-external-reading.md) and
  the empirical examples.
