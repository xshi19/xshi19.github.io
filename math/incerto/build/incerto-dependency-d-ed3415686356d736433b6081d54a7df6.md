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

Only prerequisite edges must be acyclic. The notes use the explicit
`prerequisites`, `proof_depends_on`, `special_case_of`, and `applications`
metadata fields. Existing `depends_on` lists remain compatible prerequisite
inputs while the remaining pages are reviewed. They are not a formal proof
certificate. Shared [notation](https://xshi19.github.io/math/notation/) remains outside the graph.

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

## Source and license

Adapted from `xshi19/incerto-wiki`, `content/concepts/dependency-dag.md`,
at revision `9717c9cf83aee18b47f423a14bce2f0d14bb8bcb`.
The original author-controlled material is used under MIT; the cited books
and external datasets retain their own rights. Calculations are static and
are not executed by the site build.

:::{dropdown} MIT permission notice

MIT License

Copyright (c) 2023 xshi19

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
:::
