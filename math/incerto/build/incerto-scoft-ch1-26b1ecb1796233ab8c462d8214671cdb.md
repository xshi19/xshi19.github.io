---
title: Chapter 1 — Prologue and tail terminology
---

## Reading the Prologue

The Prologue asks what changes in statistical reasoning when extreme outcomes
matter. Its terminology is deliberately broader than a single asymptotic tail
class. We use that question to separate three properties: a large fourth
moment relative to variance, the existence of exponential moments, and the
scaling of a survival function.

The source is Taleb's *Statistical Consequences of Fat Tails*, third edition
(2025), **arXiv:2001.10488v4, chapter 1, printed pp. 3–6**
[@taleb2025scoftv4]. This discussion chapter states no numbered mathematical
theorem. The calculations below and in the
[tail class catalog](incerto-tail-class-catalog.md) are our own teaching
examples; the Prologue is their motivation, not their proof source.

| Source locator | Mathematical question to carry into the notes | Reading route |
| --- | --- | --- |
| Ch. 1, p. 3 | Which hypotheses does a proposed statistical tool need? | [Exponential moments](incerto-cramer-condition.md) and [Pareto moment existence](incerto-pareto-moment-existence.md) |
| Ch. 1, unnumbered terminology discussion, pp. 4–5 | Does a description relative to the Gaussian identify an asymptotic class? | The bounded counterexample below; [catalog definitions and examples](incerto-tail-class-catalog.md) |
| Ch. 1, p. 6 | What can a calculation establish when a model assumption is uncertain? | [Counting exceedances](incerto-counting-exceedances.md) separates a finite count from a population probability |

The source uses “thick” in a Gaussian-relative, kurtosis-based sense, and
“fat” more broadly than a strict power-law definition (ch. 1, pp. 4–5).
Here we name the mathematical property explicitly. The
[Cramér condition](incerto-cramer-condition.md) asks for a finite positive
exponential moment; [regular variation](incerto-regular-variation.md) asks
for a limiting multiplier ratio; [subexponentiality](incerto-subexponentiality.md)
asks about the tail of an iid sum. None is defined by the word “fat” alone.

## High kurtosis with bounded support

(scoft-ch1-bounded-kurtosis)=
For a variable with finite fourth moment and positive variance, define its
Pearson kurtosis, locally denoted by $K$, as

$$
K=\frac{\mathbb E[(X-\mathbb E X)^4]}{\operatorname{Var}(X)^2}.
$$

This is kurtosis, not excess kurtosis $K-3$. A standard Gaussian has $K=3$:
integration by parts with $\phi'(z)=-z\phi(z)$ gives
$\int z^4\phi(z)\,dz=3\int z^2\phi(z)\,dz=3$, where both integrals run
over $\mathbb R$ and $\phi$ is the standard normal density.

We prove that arbitrarily large kurtosis is compatible with bounded support
and finite exponential moments. Fix a local probability $0<\varepsilon\le1$
and give $X$ the three-point law

$$
P(X=0)=1-\varepsilon,\qquad
P(X=\varepsilon^{-1/2})=P(X=-\varepsilon^{-1/2})=\frac{\varepsilon}{2}.
$$

Symmetry gives $\mathbb E X=0$. Summing over the three atoms gives

$$
\mathbb E X^2=\varepsilon\varepsilon^{-1}=1,\qquad
\mathbb E X^4=\varepsilon\varepsilon^{-2}=\varepsilon^{-1},\qquad
K=\varepsilon^{-1}.
$$

For every real $\theta$, the same finite sum gives

$$
\mathbb E e^{\theta X}
=1-\varepsilon+\varepsilon\cosh(\theta/\sqrt{\varepsilon})<\infty.
$$

For each fixed $\varepsilon$ the right tail ends at
$\varepsilon^{-1/2}$. It therefore cannot have a regularly varying tail
with positive exponent, or belong to the unbounded-support subexponential
class used here. Choosing smaller $\varepsilon$ produces a *different*
bounded law with larger kurtosis. It does not turn any one of these laws
into a power-tailed distribution. For example, $\varepsilon=0.01$ gives
variance $1$, kurtosis $100$, and support $\{-10,0,10\}$.

## Ordinary moments and tail classes

The catalog supplies a second contrast: a lognormal variable has all positive
power moments finite but has no finite positive exponential moment. It is
subexponential, yet not regularly varying. An exact Pareto variable has a
positive power moment of order $q$ precisely when $q<\alpha$.
The catalog [proves the lognormal moment calculations and separates the cited
classifications](incerto-tail-class-catalog.md#tail-class-catalog-moment-gates);
the [Pareto moment note](incerto-pareto-moment-existence.md) owns the exact
Pareto integral proof.

These examples answer different questions. A finite population variance
requires a finite second moment. The kurtosis comparison requires a finite
fourth moment as well. A finite Chernoff bound requires an exponential moment at
the chosen positive parameter. A multiplier-ratio claim requires information
arbitrarily far into the tail. A finite empirical fourth moment alone
establishes none of those population assumptions.

## Reproduce the model comparisons

From the installed repository environment, run:

```sh
uv run python demos/incerto/prologue_tail_classes.py
```

The script evaluates specified probability laws in dimensionless units.
There are no observed data, random draws, fitted parameters, or confidence
intervals. It prints the three-point calculation for
$\varepsilon\in\{0.1,0.01,0.001\}$ and the multiplier ratios for the
catalog's five unbounded models. A fixed 241-point grid spans
$1\le u\le10^4$, with multiplier $t=2$; runtime is bounded by this grid
and two static figure exports, normally a few seconds.

```{figure} ../assets/figures/incerto/prologue-kurtosis.svg
:alt: Three bounded laws have variance one and kurtosis 10, 100 and 1000 as the probability of a nonzero value decreases.

Exact three-point model calculations. Each marked law has finite support;
the growing support endpoint is reported with each point. The Gaussian
reference is Pearson kurtosis 3. No samples were generated.
```

The [catalog figure](incerto-tail-class-catalog.md#tail-class-catalog-demo)
shows why a vanishing multiplier ratio cannot by itself distinguish
lognormal and Weibull subexponential tails from exponential and gamma tails.
The code uses log survival functions to avoid dividing two underflowed
probabilities; it stops on non-finite values rather than dropping them.
Finite curves illustrate the models and do not prove their limiting classes.

## References and next reading

- Taleb, *Statistical Consequences of Fat Tails*, third edition (2025),
  [arXiv:2001.10488v4](https://arxiv.org/abs/2001.10488v4), ch. 1,
  pp. 3–6; terminology on pp. 4–5. Source context only.
- [Tail class catalog](incerto-tail-class-catalog.md): proofs, exact
  classification locators, and the model-ratio comparison.
- [SCoFT reading map](incerto-scoft-intro.md): available chapter guides.

Symbols follow the [shared notation](https://xshi19.github.io/math/notation/).
The Prologue supplies no equation-level notation to translate; $K$ and
$\varepsilon$ above are local to our counterexample. The book's text and
figures are not reproduced here.
