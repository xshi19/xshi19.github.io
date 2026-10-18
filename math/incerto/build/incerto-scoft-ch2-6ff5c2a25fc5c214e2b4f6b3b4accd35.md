---
title: Chapter 2 — Glossary, definitions, and notations
---

## Reading a glossary through examples

A probability, a conditional mean, and a contribution to an expectation can
all refer to observations above the same threshold. They answer different
questions. This guide uses that distinction to connect the glossary to
[counting exceedances](incerto-counting-exceedances.md), the
[Pareto law](incerto-pareto.md), the
[extreme-value index](incerto-extreme-value-index.md), and the
[tail class catalog](incerto-tail-class-catalog.md).

The source is Taleb's *Statistical Consequences of Fat Tails*, third edition
(2025), **arXiv:2001.10488v4, chapter 2, printed pp. 7–21**
[@taleb2025scoftv4]. The tables give reading locations, not transcriptions
of the glossary. Our examples and arguments are developed independently.
The [shared notation](https://xshi19.github.io/math/notation/) controls the
symbols used here; source-specific conventions need translation.

## Notation at the point of use

| Source term or symbol / locator | Our notation and meaning | Reading route |
| --- | --- | --- |
| Probability, expectation, variance; §2.1, p. 7 | $P$, $\mathbb E$, $\operatorname{Var}$; an expectation may be infinite or undefined | [Pareto moment existence](incerto-pareto-moment-existence.md) |
| Density; §2.1, p. 7 | $p_\theta(x)$ for a parameterized density; local $f$ for a fixed Lebesgue density | [Pareto definition](incerto-pareto.md#pareto-definition) |
| CDF and survival, $F$, $\bar F$, $S$; §2.1, p. 7 | $F(u)=P(X\le u)$ and $\bar F(u)=P(X>u)$ | [CDF and atoms](incerto-counting-exceedances.md#exceedances-cdf-and-atoms) |
| Sample size and sum; §2.1, pp. 7–8 | $n$, $S_n=\sum_iX_i$, $\bar X_n=S_n/n$; a hatted $\widehat S_n(u)$ is instead a local empirical survival | [Counting exceedances](incerto-counting-exceedances.md) |
| Moment order $p$; §2.1, p. 7 | $p>0$ in $\mathbb E[X^p]$ for nonnegative $X$; use a separate local $q$ for a probability level | [Moment thresholds](incerto-pareto-moment-existence.md) |
| Tail and stable indices; §2.1, p. 8 | $\alpha>0$ is the survival power exponent, $\xi$ the EVI; a stable-law index must be identified separately | [Exact Pareto EVI proof](incerto-extreme-value-index.md#evi-pareto-calibration) |
| Threshold $K$, lower endpoint $x_{\min}$; §§2.2.1, 2.2.4, pp. 9–11 | $u$ is a threshold; $x_m>0$ is a Pareto minimum; $e(u)=\mathbb E[X-u\mid X>u]$ | [Conditional Pareto quantities](incerto-pareto.md#pareto-threshold-quantities) |
| Gaussian scale and lognormal parameters; §2.1, p. 8 | Write $N(\mu,\sigma^2)$ with variance $\sigma^2$; specify $\log X\sim N(\mu,\sigma^2)$ for a lognormal | [Lognormal moments](incerto-tail-class-catalog.md#tail-class-catalog-moment-gates) |
| Mean absolute deviation and normalized $p$ expression; §2.1, pp. 7–8 | State the center and normalization explicitly; $(n^{-1}\sum_i|x_i|^p)^{1/p}$ is a scaled norm for $p\ge1$ | [Dispersion summaries](incerto-dispersion-ratio.md) |
| Characteristic function, delta, step and special functions; §2.1, pp. 8–9 | Keep these local to the calculation; our $\theta$ remains a model parameter when used that way | [Stable-limit characteristic functions](incerto-generalized-central-limit-theorem.md); special-function identities need their own references |

For a lognormal, $\mu$ is a log-location, not its ordinary mean. The catalog
derivation gives $\mathbb E X=\exp(\mu+\sigma^2/2)$, so a prescribed mean
$m>0$ requires $\mu=\log m-\sigma^2/2$. Likewise, the stable-law index is
restricted to $(0,2]$, whereas a Pareto exponent can exceed $2$; neither is
silently interchangeable with the EVI.

## Map of the glossary concepts

The core route is probability → tail size → tail class → normalized extremes.
Other glossary entries point ahead to later chapters. A link below identifies
a relevant existing note, not a claim that all of that topic is proved here.

| Term / source locator | Our page or local interpretation |
| --- | --- |
| Power-law class; §2.2.1, p. 9 | [Regular variation](incerto-regular-variation.md): $\bar F(x)=x^{-\alpha}L(x)$ has variation index $-\alpha$ |
| Weak LLN and CLT; §§2.2.2–3, p. 10 | The convergence statements below; [preasymptotic averages](incerto-lln-preasymptotic.md) distinguishes a limit from its finite-sample accuracy |
| Expected excess; §2.2.4, pp. 10–11 | [Mean excess](incerto-mean-excess-function.md) is $e(u)$; conditional tail mean is $u+e(u)$ |
| Medium numbers, preasymptotics; §§2.2.5, 2.2.17, pp. 11, 15 | [Finite-sample averages](incerto-lln-preasymptotic.md); no universal convergence rate is supplied by naming the regime |
| Kappa; §2.2.6, pp. 11–12 | Planned chapter-8 treatment of dispersion scaling; a mean-centered formula needs an existing mean and finite, nonzero dispersion terms |
| Elliptical laws and independence; §§2.2.7–8, p. 12 | [Mixtures](https://xshi19.github.io/math/normix-theory/normix-normal-mixtures/) supply examples; independence means a product joint law, and scatter need not be a finite covariance matrix |
| Stable and multivariate stable laws; §§2.2.9–10, pp. 12–13 | [Generalized CLT](incerto-generalized-central-limit-theorem.md); normalized sums and normalized maxima are different limit problems |
| Karamata point; §2.2.11, p. 13 | [Counterexample to a constant tail factor](incerto-tail-class-catalog.md#tail-class-catalog-slow-variation); no finite transition point follows from slow variation |
| Subexponentiality; §2.2.12, pp. 13–14 | [Subexponential tails](incerto-subexponentiality.md); fix the number of iid nonnegative summands while the threshold tends to infinity |
| Student-t proxy; §2.2.13, p. 14 | [Survival ratios](incerto-survival-tail-ratio.md); label degrees of freedom locally and distinguish this model from an exact Pareto |
| Stochasticizing, metaprobability; §§2.2.18, 2.2.31, pp. 15–16, 20 | [Random-scale examples](incerto-scoft-ch4.md); a distribution over a parameter defines a mixture, with dependence specified separately |
| VaR and conditional VaR; §2.2.19, p. 16 | For a nonnegative loss $X$, define a local quantile $Q(q)=\inf\{x:F(x)\ge q\}$; see the quantile and tail-mean distinction below |
| MS plot; §2.2.21, pp. 16–17 | [Max-to-sum ratio](incerto-max-to-sum-ratio.md); for moment order $p$ compare $\max_i X_i^p$ with $\sum_i X_i^p$, not with $(\sum_iX_i)^p$ |
| Maximum domain of attraction; §2.2.22, p. 17 | [EVI definition and Pareto proof](incerto-extreme-value-index.md#evi-pareto-calibration); a nondegenerate normalized maximum limit is essential |
| Tail payoff integrals; §§2.2.23–24, pp. 17–18 | [Tail contribution identity](incerto-pareto.md#pareto-threshold-quantities); $u\bar F(u)$ omits the contribution above $u$ |
| Empirical distribution; §2.2.27, p. 19 | [Strict counts](incerto-counting-exceedances.md); bounded indicator averages do not establish unbounded-moment convergence |
| Hidden tail and shadow moment; §§2.2.28–29, p. 20 | [Plug-in estimation](incerto-plug-in-tail-estimation.md); the model assigns mass beyond the sample maximum, while a sample alone does not identify that mass |
| Tail dependence; §2.2.30, p. 20 | A limiting conditional exceedance probability at marginal quantiles, when the limit exists; dedicated multivariate treatment is planned |
| Dynamic hedging; §2.2.32, p. 21 | Planned *Dynamic Hedging* sequence; no replication theorem follows from the glossary alone |
| Citation ring, rent seeking, pseudo-empiricism, skin in the game, Wittgenstein's ruler, Black Swans; §§2.2.14–16, 2.2.20, 2.2.25–26, pp. 14–19 | Interpretive source vocabulary, outside this chunk's mathematical claims; no distributional theorem or empirical finding is inferred from these labels |

The normalized expression in the first table fails the triangle inequality
when $0<p<1$: for $(1,0)$ and $(0,1)$ it gives $1$ for their sum but
$2^{1-1/p}<1$ for the sum of their values. Independence can always be
expressed by $P(X\in A,Y\in B)=P(X\in A)P(Y\in B)$ for measurable sets.
A density ratio can express it only where the denominator is positive and
the densities exist. An elliptical scatter matrix is a model parameter;
identifying it with a covariance additionally requires second moments and
the appropriate scalar normalization.

## Which limit is being taken?

For random variables $Y_n,Y$, convergence in probability means
$P(|Y_n-Y|>\varepsilon)\to0$ for every $\varepsilon>0$.
Almost-sure convergence means $P(\lim_nY_n=Y)=1$.
Convergence in distribution means convergence of the CDFs at continuity
points of the limiting CDF. These are the conventions behind the arrows in
§2.1, p. 8.

For iid $X_i$ with $\mathbb E|X_1|<\infty$, the weak LLN gives
$\bar X_n\xrightarrow{P}\mu$, where $\mu=\mathbb E X_1$.
With the stronger assumption $0<\sigma^2=\operatorname{Var}(X_1)<\infty$,
the classical CLT gives

$$
\frac{\sqrt n(\bar X_n-\mu)}{\sigma}\xrightarrow{d}N(0,1).
$$

These two general theorems are cited, not proved here (SCoFT v4
§§2.2.2–3, p. 10). Neither statement gives a universal finite-sample error
bound. The [exceedance note](incerto-counting-exceedances.md) does prove a
pointwise probability bound for its bounded indicators.

By contrast, the subexponential relation compares $P(S_n>x)$ and $P(M_n>x)$
as $x\to\infty$ with $n$ fixed. It does not say that $M_n/S_n$ tends to one
as $n$ grows. The [Pareto sums section](incerto-pareto.md#sums-and-moment-orders)
keeps the growing-sample regimes and moment orders separate.

## A probability does not specify the tail payoff

For Pareto $(\alpha=3/2,x_m=1)$, threshold $u=4$ is the quantile
$Q(7/8)$. The exceedance probability is $1/8$, but the conditional mean
above that threshold is $12$. The [Pareto derivation](incerto-pareto.md#pareto-threshold-quantities)
gives mean excess $8$ and tail contribution $3/2$.
All four numbers describe the same distribution and threshold.

For an integrable nonnegative loss, define expected shortfall at level
$0<q<1$ by $(1-q)^{-1}\int_q^1Q(v)\,dv$.
When there is no atom at the quantile, this agrees with the conditional tail
mean. At an atom, conditioning on $X>Q(q)$ generally selects a different
probability mass. For the empirical law of $(1,2,2,4)$ and $q=1/2$, the upper
half consists of equal masses at $2$ and $4$, so the quantile integral gives
$3$. Strict conditioning on $X>2$ instead gives $4$. This calculation explains
why the probability level, sign convention, and treatment of ties must accompany
the terms VaR and expected shortfall.

Even uniform CDF accuracy does not control a mean without additional tail
conditions. To prove this, let $P_j$ put mass $1-1/j$ at zero and $1/j$ at
$j^2$, for integers $j\ge2$. Its CDF differs from that of a point mass at
zero by at most $1/j$, yet its mean is $j$. This deterministic sequence of
laws is a counterexample to such an implication; it is not a simulation of
an iid empirical process. It isolates the probability-versus-moment issue
behind §§2.2.27–28, pp. 19–20, without importing a uniform convergence theorem.

(scoft-ch2-demo)=
## Reproduce the worked cases

```sh
uv run python demos/incerto/exceedances.py
uv run python demos/incerto/glossary_worked_cases.py
```

The second script uses the fixed list $(1,2,2,4)$ and exact population
formulas. It prints conditional Pareto quantities at $u=1,4,16$ for
$\alpha=0.8,1,1.5,3$, the uniform-CDF counterexample for $j=10,100,1000$,
and multiplier ratios for the catalog's logarithmic correction. It evaluates
four Pareto maximum CDFs with $\alpha=1.5$ on 301 points from $0.25$ to $4$,
and a 161-point multiplier grid from $1$ to $10^4$. Units are dimensionless.
There is no RNG, external data, model fitting, or distribution truncation;
the finite grids bound the work to a few seconds and two figure exports.

```{figure} ../assets/figures/incerto/glossary-exceedances-maxima.svg
:alt: Strict empirical survival jumps down at tied values; normalized Pareto maximum CDFs approach the Frechet CDF as sample size increases.

Left: the empirical law assigns half its mass to $2$; closed circles mark
the value after each downward jump. Right: exact iid Pareto maximum CDFs
and their proved Fréchet limit. The finite curves illustrate the formulas;
they are not estimates or evidence for an unknown population model.
```

```{figure} ../assets/figures/incerto/glossary-slow-variation.svg
:alt: The multiplier ratio of a logarithmically corrected power tail approaches one quarter while remaining above the exact Pareto ratio on the plotted range.

Both tails have exponent $2$. The corrected law has $L(u)=1+\log u$,
which grows without bound even as its multiplier ratio approaches the
Pareto value. The catalog proves the limiting claim and validates the law.
```

The numerical helper uses log probabilities for tail calculations and
reports mathematical infinity for Pareto means when $\alpha\le1$.
Unexpected non-finite arithmetic raises an error. CDF values may round to
zero or one at floating-point limits; this does not assert finite support.
Tests compare finite formulas with independent integration and distribution
calculations. The probability bounds, conditional identities, maximum limit,
and counterexamples are proved in the notes, not by the tests.

## References and next reading

- Taleb, *Statistical Consequences of Fat Tails*, third edition (2025),
  [arXiv:2001.10488v4](https://arxiv.org/abs/2001.10488v4), ch. 2,
  §§2.1–2.2.32, pp. 7–21. Section/page locators above identify source context.
- [Tail class catalog](incerto-tail-class-catalog.md) gives the precise
  classification sources and separates proved formulas from cited theorems.
- [Chapter 3 — Non-technical overview](incerto-scoft-ch3.md) connects the
  definitions to statistical consequences; its source references retain
  the edition identified on that page.
