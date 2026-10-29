---
title: Counting exceedances
---

(counting-exceedances)=
## A finite-sample quantity

For finite real observations $x_1,\ldots,x_n$ with $n\geq 1$, define the fraction
strictly above a finite threshold $t$ by

```{math}
:label: empirical-exceedance-fraction
\widehat S_n(t)=\frac{1}{n}\sum_{i=1}^n\mathbf{1}\{x_i>t\}.
```

Here $t$ is the local name for the threshold $u$ in the
[shared notation](https://xshi19.github.io/math/notation/).
$\widehat S_n$ describes the observations; $\bar F(u)=P(X>u)$ describes a
population law. The symbol $S_n$ without a hat denotes a sample sum elsewhere.

The strict inequality matters when an observation equals the threshold. For
$(1,2,2,4)$ at $t=2$, only the last observation contributes, so the fraction is
$1/4$. Counting observations greater than or equal to $2$ would give $3/4$ and
answer a different question.

## What the count establishes

Each indicator is either zero or one, so $0\leq\widehat S_n(t)\leq1$. If
$s<t$, every observation exceeding $t$ also exceeds $s$. Therefore
$\widehat S_n(t)\leq\widehat S_n(s)$. These statements follow directly from
counting; they are exact finite-sample arguments.

Use [`exceedance_fraction`](incerto-api-reference.md#api-exceedance-fraction), [`empirical_survival_function`](incerto-api-reference.md#api-empirical-survival-function).

```python
from xmath import exceedance_fraction
from xmath.incerto.stats import empirical_survival_function

sample = [1, 2, 2, 4]
fractions = [exceedance_fraction(sample, u) for u in (0, 2, 4)]
print("Fractions: [" + ", ".join(f"{v:.4g}" for v in fractions) + "]")
support, survival = empirical_survival_function(sample)
print("Support: [" + ", ".join(f"{v:.4g}" for v in support) + "]")
print("Strict survival: [" + ", ".join(f"{v:.4g}" for v in survival) + "]")
```

**Output**

```text
Fractions: [1, 0.25, 0]
Support: [1, 2, 4]
Strict survival: [0.75, 0.25, 0]
```

Ties form one jump.

The same fixed-sample calculation is available as
`uv run python demos/incerto/exceedances.py`.

A zero count above $4$ does not establish that a population cannot exceed $4$.
Inference about an unseen population requires sampling and modeling assumptions
that this example deliberately leaves unspecified. The
[Normix sample](https://xshi19.github.io/math/normix-theory/normix-conditioning-a-mixture/) instead starts
with an explicitly defined random-variable model.

The calculation uses a fixed synthetic sample; it needs no external data.

(exceedances-cdf-and-atoms)=
## CDF, survival, and ties

Use the right-continuous convention $F(u)=P(X\le u)$. Its complement is
$\bar F(u)=P(X>u)$, even when $X$ has atoms. For the empirical law that
assigns mass $1/n$ to each observation,

$$
\begin{gathered}
\widehat F_n(u)=\frac1n\sum_{i=1}^n\mathbf1_{\{x_i\le u\}},\\
\widehat S_n(u)=1-\widehat F_n(u).
\end{gathered}
$$

We prove this identity by partitioning the observations into $x_i\le u$
and $x_i>u$. The alternative count with $x_i\ge u$ adds the empirical
mass at $u$; it is not the complement of this CDF.

| Threshold $u$ for $(1,2,2,4)$ | $\widehat F_4(u)$ | $\widehat S_4(u)$ | Fraction $x_i\ge u$ |
| --- | --- | --- | --- |
| $1$ | $1/4$ | $3/4$ | $1$ |
| $2$ | $3/4$ | $1/4$ | $3/4$ |
| $4$ | $1$ | $0$ | $1/4$ |

The runnable example above uses
[`empirical_survival_function`](incerto-api-reference.md#api-empirical-survival-function).
The chapter-2 demo, `uv run python demos/incerto/glossary_worked_cases.py`,
draws the steps. Rank plotting positions with a nonzero
last ordinate are a different plotting convention.

## What iid sampling adds

Now assume $X_1,\ldots,X_n$ are iid from $F$, and fix a nonrandom threshold
$u$. Put $q=\bar F(u)$. Each indicator $I_i=\mathbf1_{\{X_i>u\}}$ has
$\mathbb E I_i=q$ and $\operatorname{Var}(I_i)=q(1-q)$ because $I_i^2=I_i$.
Independence makes the cross-covariances zero. Consequently,

$$
\begin{gathered}
\mathbb E\widehat S_n(u)=q,\\
\operatorname{Var}(\widehat S_n(u))=\frac{q(1-q)}n.
\end{gathered}
$$

These are exact finite-sample identities, proved by summing the indicators.
For any fixed $\varepsilon>0$, Chebyshev's inequality gives
$P(|\widehat S_n(u)-q|>\varepsilon)\le q(1-q)/(n\varepsilon^2)$,
which tends to zero. No moment assumption on $X$ is needed: the indicators
are bounded. This argument is pointwise in a fixed $u$; it proves neither a
uniform-in-threshold theorem nor consistency of an unbounded payoff average.

(counting-exceedances-ties)=
## A survival step and its CDF complement

```{figure} ../assets/figures/incerto/exceedances-ties.svg
:alt: Strict empirical survival is one quarter at threshold two, and its CDF complement is three quarters. Open and closed endpoints mark the two-observation jump at the tie.

The fixed list $(1,2,2,4)$. Filled dots give each function's value at a
sample value; open dots give its left limit. At $u=2$, the strict
survival counts only the observation $4$, while the CDF counts $1,2,2$.
Both functions are right-continuous and sum to one at every threshold,
including ties. These are exact counts, not fitted tail probabilities.
```

The existing [`empirical_survival_function`](incerto-api-reference.md#api-empirical-survival-function)
example supplies the same support and survival values; subtracting from one
gives the CDF. Regenerate with
`uv run python demos/incerto/visualization_p2.py`.

## Source context

The [chapter 2 guide](incerto-scoft-ch2.md) maps SCoFT v4 §2.1, pp. 7–8,
and §2.2.27, p. 19, to these conventions [@taleb2025scoftv4]. The displayed
empirical indicator in the latter section points in the survival direction;
our CDF uses $\le$ and our survival uses $>$. The counting and sampling
arguments above are proved here independently of that source presentation.
