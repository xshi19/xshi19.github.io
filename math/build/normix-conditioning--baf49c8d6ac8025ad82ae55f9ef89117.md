---
title: Conditioning a mixture
---

(conditioning-a-mixture)=
## Fix the model first

Let $Z$ be standard normal and let $W\geq0$ be independent of $Z$. For constants
$\mu,\beta\in\mathbb R$ and $\sigma>0$, define

```{math}
:label: sample-normal-mixture
X=\mu+\beta W+\sigma\sqrt W\,Z.
```

Conditional on $W=w>0$, this is normal with mean $\mu+\beta w$ and variance
$\sigma^2w$. At $w=0$ it is a point mass at $\mu$. This conditional description
does not require choosing a particular distribution for $W$.

## One exact calculation

Assume in addition that $\mathbb E[W^2]<\infty$. Conditional expectation gives

```{math}
\mathbb E[X]=\mu+\beta\mathbb E[W].
```

The law of total variance separates variation inside each conditional normal
from variation of its mean:

```{math}
\operatorname{Var}(X)
=\mathbb E[\operatorname{Var}(X\mid W)]
 +\operatorname{Var}(\mathbb E[X\mid W])
=\sigma^2\mathbb E[W]+\beta^2\operatorname{Var}(W).
```

For the limiting case $W=1$ almost surely, this reduces to a normal variable with
mean $\mu+\beta$ and variance $\sigma^2$. The second-moment assumption is a
sufficient condition for the displayed calculation, not a definition of every
possible normal mixture.

The [normal-mixtures note](./normix-normal-mixtures.md) extends this calculation
to vectors. The [GH note](./normix-generalized-hyperbolic.md) uses the literature
symbols $Y=W$, $\gamma=\beta$, and $\Sigma=\sigma^2$ in one dimension,
while keeping $Z$ standard normal and independent. Choosing a
[GIG law](./normix-generalized-inverse-gaussian.md) for the mixing variable
leads to the posterior calculations in [EM for GH](./normix-em-algorithm.md).

This original derivation introduces a mathematical model, not an assertion
about a specific `normix` constructor or parameterization. Consult the
[upstream package](https://github.com/xshi19/normix) when writing executable
package examples. Compare [counting exceedances](./incerto-counting-exceedances.md)
for a quantity that describes observations without specifying their model.
