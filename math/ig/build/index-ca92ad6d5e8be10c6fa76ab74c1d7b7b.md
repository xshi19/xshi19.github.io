---
title: Information Geometry
---

Information geometry studies probability models using geometric objects that
describe changes in the distributions themselves. A parameter vector gives
coordinates for a model. The score describes a first-order change in its log
density, and Fisher information measures the size of that change.

For a regular model $p_\theta$, write
$s_\theta(x)=\nabla_\theta\log p_\theta(x)$. The basic quadratic form is

```{math}
:label: ig-hub-fisher
g_\theta(v,w)
=\mathbb E_\theta\!\left[
  (v^\mathsf{T}s_\theta(X))(w^\mathsf{T}s_\theta(X))
\right].
```

The notes below derive this expression, explain why it transforms correctly
when coordinates change, and introduce the two affine structures that make
exponential families especially tractable.

## Reading path

The prerequisites are multivariable calculus, elementary linear algebra, and
probability densities and expectation. No previous course in differential
geometry is assumed. The first four notes supply the background; the last two
develop the first geometric results.

| Note | Main question |
| --- | --- |
| [1. From Euclidean space to a manifold](../information-geometry-euclidean-to-manifold.md) | What survives a change of coordinates, and what extra structure measures lengths? |
| [2. Exponential families and sufficient statistics](../information-geometry-exponential-families.md) | Why does one log-partition function determine means, covariances, and likelihood equations? |
| [3. Latent variables and ordinary EM](../information-geometry-latent-variables-em.md) | How does posterior completion produce a likelihood-increasing update? |
| [4. Conditional expectation as projection](../information-geometry-conditional-expectation.md) | In which space is conditional expectation orthogonal projection? |
| [5. Fisher geometry and the meaning of L2](../information-geometry-fisher-vs-l2.md) | Why does Euclidean distance between parameter vectors miss the model's statistical sensitivity? |
| [6. Dual coordinates and KL projections](../information-geometry-duality.md) | How do orthogonality, Pythagoras, and two notions of straightness fit together? |

For a geometry-first route, read 1, 2, 5, and 6, then return to EM and
conditional expectation. For a likelihood-first route, begin with 2 and 3, then
read 4 before the geometry notes. KL is introduced where EM needs it; a separate
information-theory course is not required.

## Conventions

Prefer the site-wide [shared notation](https://xshi19.github.io/math/notation/) canon for symbols that appear in more than one track. The local conventions below specialize that canon for this reading path.


Densities are taken with respect to a stated common measure, which may be
counting measure. Logarithms are natural. In exponential families, $\theta$
denotes natural coordinates, $\psi$ the log-partition function, and
$\eta=\nabla\psi$ expectation coordinates. A latent model has joint density
$P_\theta(x,z)$, observable marginal $p_\theta(x)$, and posterior
$r_\theta(z\mid x)$.

Smoothness, common support, integrability, and nonsingularity are stated where
they are used. Most geometric calculations concern interior regular models;
boundary distributions and redundant parameters require separate treatment.
The worked examples are exact calculations. No numerical approximation is used
as a proof.

## Where this leads

The next questions concern what happens when a latent variable is hidden:
how marginalization changes scores and divergences, how lost information affects
EM, and when dual flatness is lost. GH/Normix specialization and curvature
calculations belong to that later sequence.

The [conditioning example](https://xshi19.github.io/math/normix-theory/normix-conditioning-a-mixture/) gives a first
normal-mixture calculation. The [Normix theory track](https://xshi19.github.io/math/normix-theory/) links
to the independently maintained package. Return to the
[mathematical notes index](https://xshi19.github.io/math/) for the other tracks.
