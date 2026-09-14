---
title: Shared notation
---

This page is the **single shared notation canon** for Incerto, Information
Geometry, and Normix theory. New notes must reuse these symbols for the same
concepts. Do not invent parallel symbols that mean the same thing in another
track.

Track hubs and body notes may still define local symbols for one-off arguments.
When a concept already appears here, prefer the shared form and link here rather
than redefining it.

## Probability and expectation

| Symbol | Meaning |
| --- | --- |
| $P$, $P_\theta$ | A probability law; the model law at parameter $\theta$ |
| $p_\theta$, $p(x\mid\theta)$ | Density of $P_\theta$ with respect to a stated base measure |
| $\mathbb E$, $\mathbb E_\theta$ | Expectation; expectation under $P_\theta$ |
| $\operatorname{Cov}$, $\operatorname{Var}$ | Covariance and variance under the stated law |
| $X$, $x$ | Observable random element and a realized value |
| $Z$, $z$ | Latent or noise variable when that role is explicit (see Normix note below) |

Logarithms are natural unless a note states otherwise. Densities are taken with
respect to a common measure named in the note (Lebesgue or counting measure).

## Information geometry

| Symbol | Meaning |
| --- | --- |
| $\theta$ | Natural / parameter coordinates of a regular model or exponential family |
| $\eta$ | Expectation coordinates; for an exponential family $\eta=\nabla\psi(\theta)$ |
| $\psi$ | Log-partition function of an exponential family |
| $s_\theta(x)=\nabla_\theta\log p_\theta(x)$ | Score |
| $g_\theta$, $I(\theta)$ | Fisher metric / Fisher information (per observation when stated) |
| $t(x)$ | Sufficient statistic in an exponential family |
| $P_\theta(x,z)$, $p_\theta(x)$, $r_\theta(z\mid x)$ | Joint density, observable marginal, and posterior in a latent model |
| $D$, $\mathrm{KL}$ | Kullback–Leibler divergence (direction stated at use) |

Smoothness, common support, integrability, and nonsingular Fisher information are
assumed only where a note states them. Boundary laws and redundant
parameterizations need separate treatment.

## Incerto / fat tails

| Symbol | Meaning |
| --- | --- |
| $F$, $\bar F=1-F$ | Distribution function and survival function |
| $\alpha$ | Pareto / regularly varying tail index (power exponent) |
| $\xi$ | Extreme-value index |
| $u$, $x_m$ | Threshold; Pareto scale (minimum) when that model is used |
| $e(u)$ | Mean excess function above $u$ |
| $L$ | Slowly varying function in a regular-variation representation |

A finite-sample exceedance count is not the same object as a model probability
$\bar F(u)$. Keep both visible when a note moves between data and model.

## Normix / mixtures

| Symbol | Meaning |
| --- | --- |
| $Y$ | Positive mixing variable in Normix theory notes (GIG/GH constructions) |
| $W$ | Mixing variable in the original conditioning example; map $W\leftrightarrow Y$ when linking |
| $\operatorname{GIG}(p,a,b)$ | Generalized inverse Gaussian with parameters $(p,a,b)$ |
| GH | Generalized hyperbolic law obtained by normal variance–mean mixing with a GIG |
| $\mu$, $\gamma$, $\delta$, $\omega$ | Location / skewness / scale conventions as defined in the GH notes |

Package API documentation at [normix](https://xshi19.github.io/normix/) may use
implementation names; mathematical notes here follow this shared canon and map
to the API when needed.

## How to extend this page

Add a row when the same concept is about to be named in more than one track.
Prefer one short shared definition over three local glossaries. Track-specific
indexes (for example Incerto concept hubs) may point here for overlapping
symbols.
