---
title: Normix theory
---

This track develops the mathematics of normal variance–mean mixtures, their
mixing distributions, and estimation through latent variables. Begin with
[conditioning a mixture](../normix-conditioning-a-mixture.md), an original
calculation that fixes independence and moment assumptions explicitly.

The first theory batch provides the foundations:

1. [Generalized inverse Gaussian](../normix-generalized-inverse-gaussian.md)
   defines the positive mixing density, its moments, boundary families, and
   exponential-family coordinates.
2. [Generalized hyperbolic](../normix-generalized-hyperbolic.md) integrates the
   conditional Gaussian, relates joint and marginal laws, and derives the
   scale nonidentifiability and moment maps.
3. [EM for GH](../normix-em-algorithm.md) derives the GIG posterior, the three
   mixing expectations, and the normal and mixing-law M-steps.
4. [Exponential-family core](../normix-exponential-family-core.md) connects
   classical, natural, and expectation coordinates through one log-partition
   function and its derivatives.
5. [Mixture architecture](../normix-mixture-architecture.md) explains the
   mathematical roles of joint, marginal, and posterior laws and fixes the
   sufficient-statistic order.
6. [Why not gradient descent?](../normix-why-not-gradient-descent.md) separates
   structural optimization advantages from the upstream benchmark evidence
   and discusses their limits.
7. [GH family tour](../normix-gh-family-tour.md) follows gamma, inverse gamma,
   inverse Gaussian, and GIG mixing into their named normal mixtures.
8. [Normal variance–mean mixtures](../normix-normal-mixtures.md) develops
   observable moments, an exact multivariate example, and scalar CDF identities.

For a gentler route after the conditioning example, read the family tour and
normal-mixtures note before the density and estimation derivations. The
[IG exponential-family entry](https://xshi19.github.io/math/ig/information-geometry-exponential-families/)
and [ordinary EM entry](https://xshi19.github.io/math/ig/information-geometry-latent-variables-em/) supply
the general assumptions and geometry behind the calculations.

Continue after batch EM and the optimization comparison with the second batch.

## Sequential and regularized EM

1. [Online EM](../normix-online-em.md) averages posterior sufficient statistics
   sequentially and derives a Bregman regret identity with its assumptions.
2. [Shrinkage](../normix-shrinkage.md) derives penalized EM toward a reference
   distribution and distinguishes joint-KL shrinkage from covariance-only updates.
3. [EM as sufficient-statistic updates](../normix-em-framework.md) separates
   aggregation, update rules, and parameter recovery, then composes running
   averages with Bregman shrinkage.

## Structured covariances

[Factor analysis for GH](../normix-factor-analysis.md) develops low-rank plus
diagonal dispersion, ten sufficient-statistic blocks, and a constrained
M-step for a curved exponential family. It connects to the
[IG exponential-family entry](https://xshi19.github.io/math/ig/information-geometry-exponential-families/)
without assuming that ambient moment matching fits a constrained model.

## Information quantities

[Entropy, varentropy, and Rényi entropy](../normix-varentropy.md) derives
density-power formulas for exponential-family components and joint normal
mixtures. The [Fisher geometry entry](https://xshi19.github.io/math/ig/information-geometry-fisher-vs-l2/)
explains the score covariance appearing in the varentropy formula; the
[duality entry](https://xshi19.github.io/math/ig/information-geometry-duality/) supports the EM divergences.

## Notation and implementation

Cross-track symbols follow the [shared notation](https://xshi19.github.io/math/notation/) canon. Do not introduce a second name for a concept already fixed there.

The thirteen imported notes are static mathematical explanations, each retaining
its pinned source and MIT notice. Their literature mixing variable is called
$Y$; it corresponds to $W$ in the conditioning introduction. Standard normal
noise is called $Z$ throughout the overlapping constructions.

The [normix repository](https://github.com/xshi19/normix) owns the JAX
implementation, public API, package tests, and releases. Its
[package documentation](https://xshi19.github.io/normix/) remains the entry point
for installation and supported operations. This site does not install or
vendor that package. Later computational notes should use a tested upstream
version and map their mathematical convention to its supported API.

The complete [EM implementation framework](https://xshi19.github.io/normix/design/em_framework.html),
[Bessel and solver design](https://xshi19.github.io/normix/design/solvers_and_bessel.html),
and [package API](https://xshi19.github.io/normix/api/index.html) remain upstream
references. Finance theory, remaining tutorials, and research notes are deferred;
the later IG/GH research sequence remains planned.
