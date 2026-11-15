---
title: Finite-sample evidence and tail non-identification
options:
  concept:
    id: inference-asymmetry
    type: theorem
    prerequisites: [pareto]
    related: [asymmetric-payoffs, ruin]
    tags: [fat-tails, inference]
---

A sample can give an accurate count of what happened while leaving the size
of an unobserved loss almost unconstrained. We make this distinction precise
with two laws whose samples usually coincide but whose means can be far apart.
The [exceedance note](incerto-counting-exceedances.md) explains why estimating
a fixed event probability does not require a moment of the underlying variable.

Use the [shared notation](https://xshi19.github.io/math/notation/): $P$ and
$Q$ are probability laws, $X_1,\ldots,X_n$ are observations, and $\alpha$
is a positive Pareto exponent. All draws below are iid under the chosen law.

(inference-contamination)=
## A finite-sample indistinguishability bound

We prove the sample-law bound and its consequence for bounded tests.

:::{prf:theorem} Finite-sample indistinguishability under contamination
:label: thm-inference-asymmetry-contamination

Let $P$ and $R$ be probability laws on the same measurable space and let
$Q=(1-\varepsilon)P+\varepsilon R$, with $0<\varepsilon<1$.
Define total variation by

$$
d_{\mathrm{TV}}(P,Q)=\sup_A|P(A)-Q(A)|,
$$

where the supremum is over measurable events. For every integer $n\ge1$, set
$d=1-(1-\varepsilon)^n$. Then

$$
d_{\mathrm{TV}}(P^{\otimes n},Q^{\otimes n})
\le 1-(1-\varepsilon)^n\le n\varepsilon.
$$

Every measurable test $\varphi$ on the $n$-fold sample space with values in $[0,1]$
satisfies

$$
\left|\mathbb E_{Q^{\otimes n}}\varphi-
\mathbb E_{P^{\otimes n}}\varphi\right|\le d.
$$

In particular, for $\delta\in[0,1]$, rejection probability at most $\delta$
under $P^{\otimes n}$ implies power at most $\delta+d$ under $Q^{\otimes n}$.
:::

:::{prf:proof}
To generate a $Q$ sample,
independently choose the $R$ component with probability $\varepsilon$ at each
draw. With probability $(1-\varepsilon)^n$, no such choice occurs and the
conditional sample law is exactly $P^{\otimes n}$. Thus
$Q^{\otimes n}=(1-d)P^{\otimes n}+dH$ for some probability law $H$, where
$d=1-(1-\varepsilon)^n$. For any event $A$, the probability difference is
$d|H(A)-P^{\otimes n}(A)|\le d$. The union bound on the $n$ component
choices gives $d\le n\varepsilon$. The same mixture identity proves the
test bound, because both expectations under $H$
and $P^{\otimes n}$ lie in $[0,1]$. A test with rejection probability at
most $\delta$ under $P^{\otimes n}$ therefore has power at most $\delta+d$
under $Q^{\otimes n}$.
:::

Here $\delta$ is a test level, not the tail exponent.

(inference-separated-tails)=
## A bounded law and a Pareto contamination

The bound is sharp for disjoint supports and permits arbitrarily large mean
gaps in the following family.

:::{prf:corollary} Close sample laws with separated tail means
:label: cor-inference-asymmetry-separated-tails

Let $P$ be uniform on $[0,1]$ and $R$ an exact
[Pareto law](incerto-pareto.md) with minimum $x_m>1$ and exponent
$\alpha>1$. For $0<\varepsilon<1$, set
$Q=(1-\varepsilon)P+\varepsilon R$. For every integer $n\ge1$,
$d_{\mathrm{TV}}(P^{\otimes n},Q^{\otimes n})=1-(1-\varepsilon)^n$.

For $x\ge x_m$ the contaminated survival and mean are

$$
\bar F_Q(x)=\varepsilon(x_m/x)^\alpha,
\qquad
\mathbb E_Q X=\frac{1-\varepsilon}{2}
  +\varepsilon\frac{\alpha x_m}{\alpha-1}.
$$

For $1<\alpha\le2$, $Q$ has infinite second moment. For each fixed
$n\ge1$, $0<\eta<1$, desired gap $D>0$, and $\alpha>1$, there are
$\varepsilon\in(0,1)$ and $x_m>1$ such that the sample total variation is
at most $\eta$ while $\mathbb E_QX-\mathbb E_PX>D$.
:::

:::{prf:proof}
The supports are disjoint. The event $\max_i X_i>1$ has probability zero
under $P^{\otimes n}$ and $d=1-(1-\varepsilon)^n$ under $Q^{\otimes n}$,
attaining the bound in [](#thm-inference-asymmetry-contamination).
Mixing the two survivals and the two finite means gives the displayed
formulas. The infinite second moment for $1<\alpha\le2$ follows from the
[Pareto moment calculation](incerto-pareto-moment-existence.md).
Choose $0<\varepsilon\le\eta/n$, then increase $x_m$ until the mean gap
exceeds $D$. The theorem keeps sample total variation at most $\eta$.
:::

Thus a small sample-law discrepancy need not control an unbounded mean
functional over this unrestricted family.

The order of the quantifiers matters. For each fixed $\varepsilon>0$, the
detection probability tends to one as $n\to\infty$. The construction
does not say that a fixed alternative can remain hidden indefinitely.

For $\varepsilon=10^{-4}$, $x_m=10^6$, $\alpha=3/2$ and $n=1000$,
the mean is $300.49995$ instead of $0.5$, yet the probability that all
observations come from $P$ is approximately $0.904833$. Conditional on that
event, the entire observed vector has the baseline law. The sample mean is
still unbiased under $Q$: linearity gives
$\mathbb E_Q\bar X_n=n^{-1}\sum_i\mathbb E_Q X_i=\mathbb E_Q X$.
Rare large values supply its expectation.

(inference-bounded-payoffs)=
## What remains identifiable under additional bounds

The one-draw mixture also bounds the effect on a bounded payoff.

:::{prf:corollary} Bounded-payoff sensitivity to contamination
:label: cor-inference-asymmetry-bounded-payoffs

Let $P,R$ be probability laws on the same measurable space,
$Q=(1-\varepsilon)P+\varepsilon R$, and $0<\varepsilon<1$.
If a measurable payoff satisfies $0\le g\le B$ with $0\le B<\infty$, then

$$
|\mathbb E_Q g(X)-\mathbb E_P g(X)|\le\varepsilon B.
$$

If instead $|g|\le B$, the bound is $2\varepsilon B$.
:::

:::{prf:proof}
The mixture identity used in [](#thm-inference-asymmetry-contamination)
gives $\mathbb E_Qg-\mathbb E_Pg=\varepsilon(\mathbb E_Rg-\mathbb E_Pg)$.
The expectations lie in $[0,B]$ in the first case and $[-B,B]$ in the second.
:::

A common payoff bound therefore changes the problem; see
[asymmetric payoffs](incerto-asymmetric-payoffs.md). A known uniform
integrability condition or a restricted parametric family can also rule out
the alternatives used here. Such information is an assumption to justify.

An observation outside $[0,1]$ rejects this particular bounded $P$ exactly.
For a nondegenerate Gaussian, every finite upper-tail threshold has positive
exceedance probability. A large value therefore supports a statistical
comparison, not a logical contradiction. No finite observation
alone rejects the class of all bounded laws with unknown bounds.

## Worked calculation

```{figure} ../assets/figures/incerto/lecture-inference-payoffs.svg
:alt: Two vertically stacked plots show rare-contamination detection above and expected quadratic payoff changes below.

Upper panel: exact probabilities of at least one contamination, for fixed sample
sizes and a varying contamination probability. These are population curves,
not simulated detection rates. The lower panel is explained in the payoff note.
```

Use [`repeated_event_probability`](incerto-api-reference.md#api-repeated-event-probability), [`finite_payoff_mean`](incerto-api-reference.md#api-finite-payoff-mean).

```python
from xmath.incerto.lecture_risk import repeated_event_probability, finite_payoff_mean

epsilon, alpha, cutoff = 1e-4, 1.5, 1e6
tail_mean = alpha * cutoff / (alpha - 1)
mean = finite_payoff_mean([1 - epsilon, epsilon], [0.5, tail_mean])
print(f"Mixture mean: {mean:.4g}")
for n in (10, 1000, 10000):
    print(f"{n:d} {repeated_event_probability(epsilon, n):.3e}")
```

**Output**

```text
Mixture mean: 300.5
10 9.996e-04
1000 9.517e-02
10000 6.321e-01
```

The component label is detectable here because the supports are disjoint.

Figure regeneration: `uv run python demos/incerto/lecture_risk.py`. The script evaluates
three sample sizes on 161 probabilities from $10^{-6}$ to $10^{-2}$ and
prints the example above. It uses no random draws or external data.

## Source context

The chapter's discussion motivates the inference question
[@taleb2025scoftv4, ch. 3, §3.5, pp. 43–49]. The mixture construction,
test bound and worked example are proved here; they specify the scope in
which the chapter's inference asymmetry becomes a mathematical statement.
