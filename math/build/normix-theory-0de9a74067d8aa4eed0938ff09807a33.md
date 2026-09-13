---
title: Normix theory
---

This track develops the mathematics of normal variance–mean mixtures, their
mixing distributions, and estimation through latent variables. Begin with
[conditioning a mixture](./normix-conditioning-a-mixture.md), an original
calculation that fixes independence and moment assumptions explicitly.

The first imported theory batch provides this reading path:

1. [Generalized inverse Gaussian](./normix-generalized-inverse-gaussian.md)
   defines the positive mixing density, its moments, boundary families, and
   exponential-family coordinates.
2. [Generalized hyperbolic](./normix-generalized-hyperbolic.md) integrates the
   conditional Gaussian, relates joint and marginal laws, and derives the
   scale nonidentifiability and moment maps.
3. [EM for GH](./normix-em-algorithm.md) derives the GIG posterior, the three
   mixing expectations, and the normal and mixing-law M-steps.
4. [Exponential-family core](./normix-exponential-family-core.md) connects
   classical, natural, and expectation coordinates through one log-partition
   function and its derivatives.
5. [Mixture architecture](./normix-mixture-architecture.md) explains the
   mathematical roles of joint, marginal, and posterior laws and fixes the
   sufficient-statistic order.
6. [Why not gradient descent?](./normix-why-not-gradient-descent.md) separates
   structural optimization advantages from the upstream benchmark evidence
   and discusses their limits.
7. [GH family tour](./normix-gh-family-tour.md) follows gamma, inverse gamma,
   inverse Gaussian, and GIG mixing into their named normal mixtures.
8. [Normal variance–mean mixtures](./normix-normal-mixtures.md) develops
   observable moments, an exact multivariate example, and scalar CDF identities.

For a gentler route after the conditioning example, read the family tour and
normal-mixtures note before the density and estimation derivations. The
[IG exponential-family entry](./information-geometry-exponential-families.md)
and [ordinary EM entry](./information-geometry-latent-variables-em.md) supply
the general assumptions and geometry behind the calculations.

The eight imported notes are static mathematical explanations, each retaining
its pinned source and MIT notice. Their literature mixing variable is called
$Y$; it corresponds to $W$ in the conditioning introduction. Standard normal
noise is called $Z$ throughout the overlapping constructions.

The [normix repository](https://github.com/xshi19/normix) owns the JAX
implementation, public API, package tests, and releases. Its
[package documentation](https://xshi19.github.io/normix/) remains the entry point
for installation and supported operations. This site does not install or
vendor that package. Later computational notes should use a tested upstream
version and map their mathematical convention to its supported API.

Further imports and the later IG/GH research sequence remain planned. Finance,
online EM, factor-analysis derivations, and implementation design remain upstream.
