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

Let $P$ be any probability law and let
$Q=(1-\varepsilon)P+\varepsilon R$, where $R$ is another law and
$0<\varepsilon<1$. Define total variation by

$$
d_{\mathrm{TV}}(P,Q)=\sup_A|P(A)-Q(A)|,
$$

where the supremum is over measurable events. For every integer $n\ge1$,

$$
d_{\mathrm{TV}}(P^{\otimes n},Q^{\otimes n})
\le 1-(1-\varepsilon)^n\le n\varepsilon.
$$

We prove this bound and its consequence for tests. To generate a $Q$ sample,
independently choose the $R$ component with probability $\varepsilon$ at each
draw. With probability $(1-\varepsilon)^n$, no such choice occurs and the
conditional sample law is exactly $P^{\otimes n}$. Thus
$Q^{\otimes n}=(1-d)P^{\otimes n}+dH$ for some probability law $H$, where
$d=1-(1-\varepsilon)^n$. For any event $A$, the probability difference is
$d|H(A)-P^{\otimes n}(A)|\le d$. The union bound on the $n$ component
choices gives $d\le n\varepsilon$.

More generally, a measurable test $\varphi$ taking values in $[0,1]$ obeys

$$
\left|\mathbb E_{Q^{\otimes n}}\varphi-
\mathbb E_{P^{\otimes n}}\varphi\right|\le d.
$$

The same mixture identity proves this, because both expectations under $H$
and $P^{\otimes n}$ lie in $[0,1]$. A test with rejection probability at
most $\delta$ under $P$ therefore has power at most $\delta+d$ against this
$Q$. Here $\delta$ is a test level, not the tail exponent.

(inference-separated-tails)=
## A bounded law and a Pareto contamination

Take $P$ uniform on $[0,1]$ and $R$ an exact
[Pareto law](incerto-pareto.md) with minimum $x_m=b>1$ and exponent
$\alpha>1$. Their supports are disjoint. The event
$\max_i X_i>1$ has probability zero under $P^{\otimes n}$ and $d$ under
$Q^{\otimes n}$, so the total-variation bound is an equality.

For $x\ge b$ the contaminated survival and mean are

$$
\bar F_Q(x)=\varepsilon(b/x)^\alpha,
\qquad
\mathbb E_Q X=\frac{1-\varepsilon}{2}
  +\varepsilon\frac{\alpha b}{\alpha-1}.
$$

These follow by mixing the two survivals and the two finite means. For
$1<\alpha\le2$, $Q$ has infinite second moment by the
[Pareto moment calculation](incerto-pareto-moment-existence.md), although
$P$ is bounded. For any fixed $n$, any $0<\eta<1$, and any desired positive
mean gap, choose $0<\varepsilon\le\eta/n$, then increase $b$ until the
gap is exceeded. Sample total variation stays at most $\eta$. This proves
that a small sample-law discrepancy need not control an unbounded mean
functional over this unrestricted family.

The order of the quantifiers matters. For each fixed $\varepsilon>0$, the
detection probability tends to one as $n\to\infty$. The construction
does not say that a fixed alternative can remain hidden indefinitely.

For $\varepsilon=10^{-4}$, $b=10^6$, $\alpha=3/2$ and $n=1000$,
the mean is $300.49995$ instead of $0.5$, yet the probability that all
observations come from $P$ is approximately $0.904833$. Conditional on that
event, the entire observed vector has the baseline law. The sample mean is
still unbiased under $Q$: linearity gives
$\mathbb E_Q\bar X_n=n^{-1}\sum_i\mathbb E_Q X_i=\mathbb E_Q X$.
Rare large values supply its expectation.

(inference-bounded-payoffs)=
## What remains identifiable under additional bounds

If a measurable payoff satisfies $0\le g(x)\le B$ with finite $B$, the
one-draw mixture identity gives

$$
|\mathbb E_Q g(X)-\mathbb E_P g(X)|\le\varepsilon B.
$$

For $|g|\le B$, replace the right side by $2\varepsilon B$. A common
payoff bound therefore changes the problem; see
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
:alt: Detection probabilities for rare contamination and expected quadratic payoffs as spread increases.

Left: exact probabilities of at least one contamination, for fixed sample
sizes and a varying contamination probability. These are population curves,
not simulated detection rates. The right panel is explained in the payoff note.
```

Run `uv run python demos/incerto/lecture_risk.py`. The script evaluates
three sample sizes on 161 probabilities from $10^{-6}$ to $10^{-2}$ and
prints the example above. It uses no random draws or external data.

## Source context

The chapter's discussion motivates the inference question
[@taleb2025scoftv4, ch. 3, §3.5, pp. 43–49]. The mixture construction,
test bound and worked example are proved here; they specify the scope in
which the chapter's inference asymmetry becomes a mathematical statement.
