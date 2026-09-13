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

The strict inequality matters when an observation equals the threshold. For
$(1,2,2,4)$ at $t=2$, only the last observation contributes, so the fraction is
$1/4$. Counting observations greater than or equal to $2$ would give $3/4$ and
answer a different question.

## What the count establishes

Each indicator is either zero or one, so $0\leq\widehat S_n(t)\leq1$. If
$s<t$, every observation exceeding $t$ also exceeds $s$. Therefore
$\widehat S_n(t)\leq\widehat S_n(s)$. These statements follow directly from
counting; they are exact finite-sample arguments.

The helper `xmath.exceedance_fraction` implements this convention. From a checkout,
run `uv run python demos/incerto/exceedances.py` to obtain fractions $1$, $1/4$,
and $0$ at thresholds $0$, $2$, and $4$. The observations are synthetic and fixed.

A zero count above $4$ does not establish that a population cannot exceed $4$.
Inference about an unseen population requires sampling and modeling assumptions
that this example deliberately leaves unspecified. The
[Normix sample](./normix-conditioning-a-mixture.md) instead starts
with an explicitly defined random-variable model.

This is an original worked derivation for the foundation, with no imported wiki
text or external data.
