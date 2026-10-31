---
title: Why not gradient descent?
---

For a full exponential family, reducing data to sufficient statistics can make
maximum likelihood much cheaper than repeatedly evaluating a density on every
observation. For GH mixtures, [EM](./normix-em-algorithm.md) uses the tractable
joint family to organize that calculation. These structural advantages explain
the upstream fitting choices; they do not rule out direct likelihood optimization.

## Exponential-family likelihood after one reduction

For independent observations from a regular exponential family, form

```{math}
\widehat\eta=\frac1n\sum_i t(x_i),\qquad
\widehat\theta=\operatorname*{arg\,min}_{\theta\in\Theta}
\{\psi(\theta)-\langle\theta,\widehat\eta\rangle\}.
```

An attained interior optimum matches $\nabla\psi(\widehat\theta)$ to
$\widehat\eta$. For [GIG](./normix-generalized-inverse-gaussian.md), the
statistic is $(\log y,y^{-1},y)$, so one data reduction leaves a
three-parameter convex problem in natural coordinates. The Bessel normalizer
is then evaluated per candidate parameter, without a separate evaluation for
each observation. A carefully implemented direct GIG likelihood optimizer
can use this same reduction; the saving comes from recognizing sufficiency.

For the [GH marginal](./normix-generalized-hyperbolic.md), that reduction is
not available in the same fixed statistic vector of $x$ alone. EM instead
computes posterior expected complete-data statistics and applies the joint
expectation-to-parameter map. Its normal block has a closed-form update;
the GIG block still uses numerical optimization. Thus EM can itself contain
a gradient-based solver.

The EM ascent guarantee concerns the **observed-data** likelihood under exact
E-steps and improving M-steps. Dempster, Laird, and Rubin introduced the method
in [1977](https://academic.oup.com/jrsssb/article-abstract/39/1/1/7027539);
the [IG derivation](https://xshi19.github.io/math/ig/information-geometry-latent-variables-em/#ig-em-monotonicity)
states what that guarantee does and does not imply.

## Coordinates and feasible parameters

Direct optimization often replaces positive parameters by unconstrained ones,
for example

```{math}
a=\log(1+e^{\phi_a}),\qquad b=\log(1+e^{\phi_b}),\qquad
\Sigma=LL^\top
```

with a triangular $L$ having positive diagonal. The objective is the average
negative observed log likelihood,

```{math}
\operatorname{NLL}(\phi)=-\frac1n\sum_i\log f_X(x_i;\vartheta(\phi)).
```

Such maps enforce interior feasibility in exact arithmetic, but change the
optimization geometry. Their derivatives can become small near a boundary,
and floating-point evaluation can still underflow. Softplus cannot represent
$a=0$ or $b=0$ at a finite unconstrained parameter. Box constraints have their
own boundary behavior.

A well-defined gradient also does not resolve the
[GH scale redundancy](./normix-generalized-hyperbolic.md#normix-gh-scale).
Different parameter tuples can yield the same observed density, so parameter
error and likelihood error measure different things.

## What the pinned upstream comparison reports

The source design note describes a CPU benchmark comparing moment-based
exponential-family MLE or GH EM with L-BFGS and Adam on a softplus-parameterized
NLL, plus a box-constrained GIG NLL solve. It reports $n=2000$ for gamma/GIG
and $n=600,d=2$ for GH, using JAX 0.9.1. The following is a qualitative summary
of that **reported experiment**, not a benchmark run for this site:

| Case | Reported finding |
| --- | --- |
| Gamma control | L-BFGS and sufficiently iterated Adam matched the likelihood and fitted parameters of the structured method |
| Interior GIG | L-BFGS matched the structured MLE; the structured implementation was faster; Adam lagged at the reported step budget |
| Nearly degenerate GIG | Flat likelihood directions allowed large parameter differences; some direct solves drove a parameter toward a numerical bound or underflow |
| Two-dimensional GH | EM and L-BFGS reached similar likelihoods, with EM faster in the measured setup; Adam had not reached a comparable likelihood at its tested budget |

The benchmark motivates the upstream default in the tested regimes. Runtime
ratios depend on implementation, hardware, compilation, initialization, stopping
rules, and objective evaluation. No timing or finite-difference accuracy from
that experiment is presented as a locally reproduced result.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/design/why_not_gradient_descent.md)
retains the quantitative report and its reproduction command.

## What this does not show

The comparison does not establish that EM always outperforms direct optimization,
that Adam cannot fit GIG, or that L-BFGS cannot recover an interior MLE.
The source itself reports successful interior quasi-Newton fits. A weakly
identified direction can permit large parameter error even when densities
are close; the joint [Hellinger comparison](./normix-generalized-hyperbolic.md#hellinger-distance-for-the-joint-law)
is useful but depends on the chosen latent representation.

The source also distinguishes an earlier order-derivative problem in Bessel
wrappers from its current benchmark. It reports agreement between automatic
differentiation and finite differences of the same kernel at the pin. That
is evidence about the tested numerical derivative, not a proof of global
optimization quality or an independent validation of the kernel's absolute
accuracy. Those upstream tests were not executed for this adaptation.

The [exponential-family core](./normix-exponential-family-core.md) explains
why convexity in natural coordinates, Hessian conditioning, and convexity
after a nonlinear parameter transformation are separate questions. EM adds
its own limitations: posterior work scales with the observations, convergence
can be slow, and local stationary points and boundary fits remain possible.

## When direct optimization is useful

A surrounding model may need to optimize mixture parameters jointly with
parameters that do not admit an EM or exponential-family update. A minibatch
setting may also make full posterior sweeps inconvenient. In those situations,
a differentiable observed likelihood is a useful primitive, with explicit
feasibility constraints, initialization, and convergence checks.

Implementation remains in the [package API](https://xshi19.github.io/normix/api/index.html)
and [EM fitting guide](https://xshi19.github.io/normix/user_guide/em_fitting.html).
The deferred [solver/Bessel design](https://xshi19.github.io/normix/design/solvers_and_bessel.html)
and [EM framework](https://xshi19.github.io/normix/design/em_framework.html)
describe the numerical machinery. This mathematical note contains no executable
JAX loop and adds no alternative fitter to this repository.

## Source and adaptation

Adapted from `xshi19/normix`, `docs/design/why_not_gradient_descent.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/design/why_not_gradient_descent.md)
records the original version. Notation, mathematical qualifications, and links
were adapted for this site; API recipes, executable package cells, and notebook
plots are omitted. No upstream benchmark execution or formal-proof verification
is claimed for this adaptation.

:::{dropdown} MIT permission notice

MIT License

Copyright (c) 2020 xshi19

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
