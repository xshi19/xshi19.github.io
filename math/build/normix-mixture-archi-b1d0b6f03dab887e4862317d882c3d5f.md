---
title: Mixture architecture — joint and marginal laws
---

A normal variance–mean mixture has two related statistical descriptions:
the joint law of an observation and its mixing variable, and the marginal law
of the observation alone. Keeping these distinct explains the sufficient
statistics and expectation-to-parameter map used by [EM](./normix-em-algorithm.md).

## Conditional, joint, and marginal descriptions

Let $Y>0$ be independent of $Z\sim\mathcal N_d(0,I_d)$, and let
$LL^\top=\Sigma\succ0$. Define

```{math}
X=\mu+\gamma Y+\sqrt Y\,LZ,\qquad
X\mid Y=y\sim\mathcal N_d(\mu+\gamma y,y\Sigma).
```

For $d=1$, this is [conditioning a mixture](./normix-conditioning-a-mixture.md)
with $Y=W$, $\gamma=\beta$, and $\Sigma=\sigma^2$. For a mixing density $g$,

```{math}
:label: normix-mixture-layers
f(x,y)=f(x\mid y)g(y),\qquad
f_X(x)=\int_0^\infty f(x,y)\,dy,\qquad
f(y\mid x)=\frac{f(x,y)}{f_X(x)}.
```

| Law | Variables available | Statistical role |
| --- | --- | --- |
| Conditional normal | $X$ at fixed $Y=y$ | Specifies the sampling mechanism |
| Joint | $(X,Y)$ | Complete-data likelihood, sufficient statistics, joint divergences |
| Marginal | $X$ | Observed likelihood, moments, observable probabilities |
| Posterior | $Y$ given $X=x$ | Converts observations into expected complete-data statistics |

Drawing both variables and discarding $Y$ produces a marginal draw. Integrating
out $Y$ is an operation on a probability model, not an assertion that $Y$ was
observed.

## When the joint is an exponential family

For GIG mixing, expansion of $\log f(x,y)$ gives the six statistic blocks

```{math}
:label: normix-mixture-statistics
t(x,y)=(\log y,y^{-1},y,x,x/y,xx^\top/y).
```

The scalar coefficients in the first three slots are

```{math}
p-1-d/2,\qquad
-\tfrac12(b+\mu^\top\Sigma^{-1}\mu),\qquad
-\tfrac12(a+\gamma^\top\Sigma^{-1}\gamma).
```

The factors of $1/2$ follow from the GIG kernel
$y^{p-1}e^{-(ay+b/y)/2}$. The remaining coefficients and the log-partition
are derived in [GH](./normix-generalized-hyperbolic.md#gh-natural-params).
Matrix blocks use the trace inner product on symmetric matrices.

The first three expectations are always ordered as
$(\mathbb E[\log Y],\mathbb E[Y^{-1}],\mathbb E[Y])$ in this batch.
Upstream implementation storage may use another order; descriptive moments,
rather than slot numbers, identify the mathematical correspondence.

The joint GIG-normal model is an exponential family in these statistics.
An arbitrary mixing density does not automatically give this particular
finite-dimensional representation. Likewise, marginalization does not generally
preserve exponential-family structure. For the full observable GH model,
Bessel terms couple parameters to the observation inside the log density.
The mere presence of a Bessel function or integral is **not** a proof that a
model is outside the exponential family; GIG itself is a counterexample to
that reasoning. Special restricted marginal families can behave differently.

## Recovering a model from expected statistics

Let $\eta_i=\mathbb E[t_i(X,Y)]$ in the order [](#normix-mixture-statistics).
Conditional Gaussian moments imply

```{math}
\eta_4=\mu+\gamma\eta_3,\qquad
\eta_5=\mu\eta_2+\gamma.
```

If $\eta_2\eta_3\ne1$, this linear system gives

```{math}
\mu=\frac{\eta_4-\eta_3\eta_5}{1-\eta_2\eta_3},\qquad
\gamma=\frac{\eta_5-\eta_2\eta_4}{1-\eta_2\eta_3}.
```

The sixth moment gives

```{math}
\Sigma=\eta_6-\eta_5\mu^\top-\mu\eta_5^\top
+\eta_2\mu\mu^\top-\eta_3\gamma\gamma^\top.
```

The mixing law is recovered from its own moment equations. This is the joint
family's inverse expectation map, with the domain conditions stated in the
[exponential-family core](./normix-exponential-family-core.md).
An arbitrary tuple of six arrays need not be a feasible expectation vector.
In particular, the recovered covariance must be positive definite and the
mixing moments must admit the claimed distribution.

In EM, replace each $\eta_i$ by an average of conditional expectations given
the observations. This uses the same mathematical map, but the expectations
now come from the current posterior. See
[latent variables and ordinary EM](./information-geometry-latent-variables-em.md)
for why this improves the observed likelihood under the stated assumptions.

## Conditional parameters and observable moments

The normal parameters enter the marginal moments through

```{math}
\mathbb E[X]=\mu+\gamma\mathbb E[Y],\qquad
\operatorname{Cov}(X)=\mathbb E[Y]\Sigma
+\operatorname{Var}(Y)\gamma\gamma^\top.
```

Thus $\mu$ and $\Sigma$ describe the conditional mechanism, and cannot simply
be read as the mean and covariance of the observed distribution. The
[normal-mixtures note](./normix-normal-mixtures.md) works through this distinction.
Moreover, a latent rescaling can change the joint representation while
preserving the marginal law; see
[GH identifiability](./normix-generalized-hyperbolic.md#model-identifiability).

This note covers the full positive-definite covariance model. Factor-analysis
constructions introduce additional latent variables and different complete-data
statistics; the [factor-analysis note](./normix-factor-analysis.md) derives
their posterior moments and constrained M-step.

Class naming, storage, constructor recipes, and fitter contracts remain in the
[upstream mixture design](https://xshi19.github.io/normix/design/mixtures.html),
[EM framework](https://xshi19.github.io/normix/design/em_framework.html), and
[package API](https://xshi19.github.io/normix/api/index.html).

## Source and adaptation

Rewritten as a mathematical note from `xshi19/normix`, `docs/design/mixtures.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/design/mixtures.md)
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
