---
title: Conditional expectation as projection
---

Conditional expectation is an orthogonal projection when the projected
quantity is square integrable. The ambient vectors are random variables, and
the inner product is an expectation under a fixed probability law. This gives
a precise meaning to the part of a quantity that can be recovered from an
observation.

## Fix the probability law and the space

Work on a probability space with law $P$. The Hilbert space $L^2(P)$ consists
of real random variables $A$ with $\mathbb E[A^2]<\infty$, identifying variables
that agree almost surely. Its inner product and norm are

```{math}
\langle A,B\rangle_P=\mathbb E[AB],\qquad
\|A\|_P^2=\mathbb E[A^2].
```

Let $X$ be an observation and $\sigma(X)$ the information it generates. Define
the subspace

```{math}
\mathcal H_X=L^2(\sigma(X),P).
```

For ordinary real- or vector-valued observations, this is the space of all
square-integrable functions $g(X)$. The space is closed: an $L^2$-convergent
sequence of $\sigma(X)$-measurable variables has an almost surely convergent
subsequence, and its limit has a $\sigma(X)$-measurable version. Thus its
$L^2$ limit belongs to $\mathcal H_X$.

The observation $X$ itself need not have a finite second moment. It is the
projected quantity $A$, and each comparison function $g(X)$, that must be in
$L^2(P)$.

## The residual is orthogonal to every function of the observation

Let $m=\mathbb E[A\mid X]$. Conditional Jensen gives
$\mathbb E[m^2]\le\mathbb E[A^2]$, so $m\in\mathcal H_X$. By the defining
property of conditional expectation,

```{math}
\mathbb E[(A-m)g(X)]
=\mathbb E\!\left[g(X)\mathbb E[A-m\mid X]\right]=0
```

for bounded $g$. For a general $g(X)\in\mathcal H_X$, truncate $g$ and pass
to the limit using Cauchy–Schwarz. The same identity holds because $A-m$ is
square integrable. Therefore

```{math}
:label: ig-conditional-projection
\Pi_X A=\mathbb E[A\mid X],\qquad
A-\Pi_X A\perp\mathcal H_X.
```

This proves the projection claim directly: the candidate lies in the closed
subspace and its residual is orthogonal to that subspace. Uniqueness is up to
almost sure equality, as it must be in $L^2(P)$.

## Pythagoras and least-squares prediction

For any $g(X)\in\mathcal H_X$, write
$A-g(X)=(A-m)+(m-g(X))$. Expanding its squared norm and using orthogonality
eliminates the cross term:

```{math}
:label: ig-conditional-pythagoras
\mathbb E[(A-g(X))^2]
=\mathbb E[(A-m)^2]+\mathbb E[(m-g(X))^2].
```

The first term is independent of $g$. Consequently $m$ uniquely minimizes
mean squared prediction error over all square-integrable functions of $X$.
The theorem concerns a chosen joint law of $(A,X)$; it does not by itself give
a procedure for learning an unknown conditional mean from finite data.

Taking $g=0$ also proves $\|\Pi_XA\|_P\le\|A\|_P$. Applying conditional
expectation twice changes nothing, so $\Pi_X^2=\Pi_X$. These are the
contraction and idempotence properties of orthogonal projection.

## A nonlinear example

Let $X$ be uniform on $[-1,1]$ and let $\varepsilon$ be independent of $X$
with mean zero and variance $\tau^2<\infty$. Set
$A=X^2+\varepsilon$. Then

```{math}
\mathbb E[A\mid X]=X^2.
```

This is nonlinear in the observed value, but conditional expectation is still
a linear operator on random variables:
$\Pi_X(aA+bB)=a\Pi_XA+b\Pi_XB$.

If prediction is restricted to affine functions $a+bX$, symmetry gives
$\operatorname{Cov}(A,X)=0$, so the best affine predictor is the constant
$\mathbb E[A]=1/3$. Since $\mathbb E[X^4]=1/5$,

```{math}
\begin{aligned}
\mathbb E[(A-X^2)^2]&=\tau^2,\\
\mathbb E[(A-1/3)^2]&=\tau^2+\frac15-\frac19
=\tau^2+\frac4{45}.
\end{aligned}
```

Projecting onto the span of $1$ and $X$ discards a predictable nonlinear
component. Projecting onto $\mathcal H_X$ retains it.

## Vector quantities and total covariance

For a finite-dimensional random vector $A$ with
$\mathbb E[\|A\|_2^2]<\infty$, apply the scalar result to each component.
Writing $m=\mathbb E[A\mid X]$ and $\mu=\mathbb E[A]$ gives
$A-\mu=(A-m)+(m-\mu)$. Both cross-covariance matrices vanish by the same
orthogonality identity. Hence

```{math}
:label: ig-total-covariance
\operatorname{Cov}(A)
=\operatorname{Cov}(\mathbb E[A\mid X])
 +\mathbb E[\operatorname{Cov}(A\mid X)].
```

The second term is positive semidefinite. It measures the variation in $A$
remaining after $X$ is known. The scalar version is the law of total variance
used in the [normal-mixture conditioning example](./normix-conditioning-a-mixture.md).

In [ordinary EM](./information-geometry-latent-variables-em.md), posterior
expectations are computed under the current model law. If they are viewed as
projections, that law, and therefore the inner product, must be held fixed for
the step. A later missing-information calculation will apply total covariance
to a joint score after first proving how its marginal score is obtained.

Continue with [Fisher geometry](./information-geometry-fisher-vs-l2.md), which
uses $L^2$ inner products on scores, or return to the
[Information Geometry hub](./information-geometry.md).
