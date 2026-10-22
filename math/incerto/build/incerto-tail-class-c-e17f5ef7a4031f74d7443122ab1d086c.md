---
title: Tail Class Catalog
options:
  concept:
    id: tail-class-catalog
    type: distribution
    depends_on:
      - subexponentiality
      - regular-variation
    tags:
      - fat-tails
      - asymptotics
      - distributions
---

## Main facts

The [SCoFT Prologue guide](incerto-scoft-ch1.md) separates Gaussian-relative
kurtosis from asymptotic tail classes. This catalog compares nonnegative
right-tail models by [subexponentiality](incerto-subexponentiality.md) and
[regular variation](incerto-regular-variation.md), then checks which moments
exist. The definitions concern a specified population law.

The positive classifications are cited, not proved here: Foss, Korshunov,
and Zachary, second edition (2013), §3.5, Theorem 3.29 and its following
paragraph, p. 57 (regular variation, including Pareto); Theorem 3.30 and
its following examples, p. 58 (lognormal and Weibull with shape below one).
See the [book](https://doi.org/10.1007/978-1-4614-7101-1).
We derive the elementary tail ratios and moment contrasts below; the
appendix proves the convolution decomposition.

Let $\bar F(x)=\mathbb P(X>x)$ be the survival function.  In the table,
"regularly varying" means $\bar F\in RV_{-\alpha}$ for some $\alpha>0$, so
$\bar F(tx)/\bar F(x)\to t^{-\alpha}$ for fixed $t>0$. Here $F$ is the CDF,
$x$ is the threshold, and $\alpha$ is a positive tail exponent. All examples
below have nonnegative support, with positive scale parameters, positive gamma
shape and rate, and nonzero lognormal log-variance. See [Shared notation](https://xshi19.github.io/math/notation/).

| Distribution | Subexponential? | Regularly varying? | Diagnostic tail behavior |
| --- | --- | --- | --- |
| [Pareto](./incerto-pareto.md), $\bar F(x)=(x_m/x)^\alpha$ | Yes | Yes | Exact power tail. |
| Lognormal | Yes | No | Heavier than any exponential, lighter than any power. |
| Weibull, $\bar F(x)=\exp(-x^\beta)$ with $0<\beta<1$ | Yes | No | Stretched exponential tail. |
| Exponential | No | No | Memoryless light tail. |
| Gamma | No | No | Exponential tail with a polynomial factor. |

For the lognormal row, "heavier than any exponential, lighter than any power"
means the survival function satisfies

$$
e^{cx}\bar F(x)\to\infty
\quad\text{for every }c>0,
\qquad
x^p\bar F(x)\to0
\quad\text{for every }p>0.
$$

The inclusion direction to remember is:

$$
\bar F\in RV_{-\alpha},\ \alpha>0
\quad\Longrightarrow\quad
F\text{ is subexponential},
$$

for distributions supported on $[0,\infty)$.  The converse fails: lognormal and
stretched-Weibull tails are subexponential without being regularly varying.

(tail-class-catalog-slow-variation)=
## Slow variation need not settle to a constant

SCoFT v4 §§2.2.1 and 2.2.11, pp. 9 and 13, discuss a practical transition
to a nearly constant tail factor [@taleb2025scoftv4]. Such a transition is an
extra modeling assumption. The definition of slow variation does not imply
a finite constant limit, or an exact Pareto tail beyond a finite threshold.

Here is a complete counterexample, in dimensionless units:

$$
\bar F(x)=
\begin{cases}
1,&x<1,\\
x^{-2}(1+\log x),&x\ge1.
\end{cases}
$$

We prove that this is a survival function with a regularly varying tail.
It is continuous at $1$, tends to zero at infinity, and its derivative on
$x>1$ is $-x^{-3}(1+2\log x)<0$. Thus it defines a probability law on
$[1,\infty)$. Its factor $L(x)=1+\log x$ tends to infinity, while for every
fixed $t>0$ and sufficiently large $x$,

$$
\frac{L(tx)}{L(x)}=1+\frac{\log t}{1+\log x}\longrightarrow1,
\qquad
\frac{\bar F(tx)}{\bar F(x)}\longrightarrow t^{-2}.
$$

The [chapter 2 worked cases](incerto-scoft-ch2.md#scoft-ch2-demo) evaluate
this ratio alongside the exact Pareto ratio $1/4$ at multiplier $2$.
A finite curve cannot establish where a tail approximation becomes adequate;
[threshold selection](incerto-tail-threshold-selection.md) requires its own
diagnostics. The example disproves a constant-limit requirement analytically.

(tail-class-catalog-moment-gates)=
## Moment gates are different questions

The [Cramér condition](incerto-cramer-condition.md) asks whether
$\mathbb E e^{\theta X}<\infty$ for at least one $\theta>0$. Ordinary moments
ask about $\mathbb E X^q$, with $q>0$. For the following nonnegative models:

| Model | Finite positive power moments | Finite positive exponential moments |
| --- | --- | --- |
| Exponential, rate $1$ | Every $q>0$ | Exactly $0<\theta<1$ |
| Lognormal, $\log X\sim N(0,1)$ | Every $q>0$ | None |
| Pareto Type I, $\alpha>0$, $x_m>0$ | Exactly $0<q<\alpha$ | None |

The [Pareto moment proof](incerto-pareto-moment-existence.md) gives the last
row's power-moment boundary, including divergence at $q=\alpha$; the
[Cramér note](incerto-cramer-condition.md) proves its exponential-moment
failure. For the exponential row, direct integration gives
$\int_0^\infty x^q e^{-x}\,dx=\Gamma(q+1)<\infty$ and
$\int_0^\infty e^{-(1-\theta)x}\,dx=1/(1-\theta)$ when $\theta<1$;
the latter integral diverges when $\theta\ge1$. Here $\Gamma$ denotes
the gamma integral.

### Proof of the lognormal moment contrast

We prove both entries in the lognormal row. Let $Z\sim N(0,1)$ and
$X=e^Z$. Completing the square gives, for every real $q$,

$$
\mathbb E X^q
=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}e^{qz-z^2/2}\,dz
=e^{q^2/2}\frac{1}{\sqrt{2\pi}}
  \int_{-\infty}^{\infty}e^{-(z-q)^2/2}\,dz
=e^{q^2/2}.
$$

In contrast, for any fixed $\theta>0$,

$$
\mathbb E e^{\theta X}
=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}
  \exp(\theta e^z-z^2/2)\,dz=\infty.
$$

Indeed, $e^z/z^2\to\infty$ as $z\to\infty$, so the integrand is at
least $1$ for all sufficiently large $z$. The integral over that half-line
diverges. This establishes failure of the Cramér condition despite all
positive power moments being finite. Subexponentiality is a separate,
cited result; it does not follow just from this failure.

For a Pareto model with $\alpha=5/2$ and $x_m=1$, the mean is $5/3$
and the variance is $20/9$, but the fourth moment diverges. The formulas
follow from the linked integral: $\mathbb E X^2=5$ and
$5-(5/3)^2=20/9$. Pearson kurtosis is therefore not a finite summary for
this model. Having a finite variance does not imply a finite fourth or
positive exponential moment.

## Reusable diagnostics

(tail-class-catalog-diagnostics)=
For a nonnegative continuous law with unbounded support, compare

$$
A_t(x)=\frac{\bar F(tx)}{\bar F(x)},\qquad
R(x)=\frac{\mathbb P(X_1+X_2>x)}{2\bar F(x)},
$$

where $X_1,X_2$ are iid and $t>1$ is fixed. The first ratio checks multiplicative
tail scaling; the second is the defining two-summand subexponential ratio.
These are population quantities. Finite evaluations of them do not certify
an asymptotic class from data.

| Distribution and parameters | Multiplier ratio $A_t(x)$ | Limit of $R(x)$ |
| --- | --- | --- |
| Pareto Type I, $x\ge x_m$ | $t^{-\alpha}$ | $1$ |
| Lognormal, $\log X\sim N(0,1)$ | $\bar\Phi(\log x+\log t)/\bar\Phi(\log x)\to0$ | $1$ |
| Weibull, $0<\beta<1$ | $\exp[-(t^\beta-1)x^\beta]\to0$ | $1$ |
| Exponential, rate $1$ | $e^{-(t-1)x}\to0$ | $\infty$ |
| Gamma, shape $2$, rate $1$ | $e^{-(t-1)x}(1+tx)/(1+x)\to0$ | $\infty$ |

Here $\bar\Phi$ is the standard normal survival function. For the lognormal
row, the normal-tail asymptotic $\bar\Phi(z)\sim\phi(z)/z$, with $\phi$ the
standard normal density, yields

$$
A_t(x)\sim
\frac{\log x}{\log x+\log t}
\exp\left[-(\log t)\log x-\frac{(\log t)^2}{2}\right]
\longrightarrow0.
$$

The asymptotic for $\bar\Phi$ follows from the complementary-error-function
expansion in [NIST DLMF, Eq. 7.12.1](https://dlmf.nist.gov/7.12#E1).
It also gives
$\log\bar F(x)=-\tfrac12(\log x)^2-\log\log x-\tfrac12\log(2\pi)+o(1)$
for this standard lognormal as $x\to\infty$. Adding $cx$ or $p\log x$
proves the two exponential-versus-power comparisons stated above. For
$\log X\sim N(\mu,\sigma^2)$ with $\sigma>0$, replace $\log x$ in
the normal-tail calculation by $(\log x-\mu)/\sigma$; the same comparisons
and zero multiplier-ratio limit follow.
Thus the multiplier diagnostic separates power tails from all four other
examples, but cannot distinguish subexponential lognormal and stretched-Weibull
tails from the two light-tailed examples. The classifications of lognormal and
stretched Weibull are cited results; a ratio table is not a proof of their
subexponentiality.

## Exact light-tail checks

We prove the two negative classifications in the table. Write
$\operatorname{Gamma}(k,\lambda)$ in shape–rate notation, with $k,\lambda>0$.
At rate one, convolution of the densities with shapes $k$ and $\ell>0$
gives

$$
\frac{e^{-x}}{\Gamma(k)\Gamma(\ell)}
\int_0^x y^{k-1}(x-y)^{\ell-1}\,dy
=\frac{e^{-x}x^{k+\ell-1}}{\Gamma(k+\ell)},\qquad x>0,
$$

using the beta integral ([NIST DLMF, Eq. 5.12.1](https://dlmf.nist.gov/5.12#E1)).
Thus independent shapes add. Scaling by
$1/\lambda$ gives the same rule at a common rate $\lambda$.
The survival integral, compared with its leading integrand by l'Hôpital's
rule, gives

$$
\bar F(x)\sim\frac{(\lambda x)^{k-1}e^{-\lambda x}}{\Gamma(k)},
\qquad x\to\infty.
$$

For completeness, at rate one the derivatives of
$\int_x^\infty y^{k-1}e^{-y}\,dy$ and $x^{k-1}e^{-x}$ have ratio
$[1-(k-1)/x]^{-1}\to1$; both functions tend to zero. Consequently,
for fixed $t>1$,

$$
A_t(x)\sim t^{k-1}e^{-\lambda(t-1)x}\to0,\qquad
R(x)\sim\frac{\Gamma(k)}{2\Gamma(2k)}(\lambda x)^k\to\infty.
$$

This proves that every positive-shape gamma law, including the exponential
case $k=1$, is neither regularly varying with a finite positive exponent
nor subexponential.

For integer $k\ge1$ and rate one, repeated integration by parts gives the
exact survival $e^{-x}\sum_{j=0}^{k-1}x^j/j!$ for $x\ge0$. Hence

$$
R_{\operatorname{Exp}(1)}(x)=\frac{1+x}{2},
\qquad
R_{\operatorname{Gamma}(2,1)}(x)
=\frac{1+x+x^2/2+x^3/6}{2(1+x)}.
$$

| Threshold $x$ | Exponential sum ratio | Gamma shape 2 sum ratio |
| --- | --- | --- |
| $2$ | $1.5$ | $19/18\approx1.056$ |
| $10$ | $5.5$ | $683/66\approx10.348$ |
| $100$ | $50.5$ | $515303/606\approx850.335$ |

Both ratios diverge, although the gamma ratio happens to be near one at
$x=2$. A single moderate-threshold observation is insufficient to infer a
limiting tail class. The [Cramér condition](./incerto-cramer-condition.md)
provides the exponential-moment contrast for these two light-tailed laws.

(tail-class-catalog-demo)=
## A finite model comparison

Use [`pareto_type1`](incerto-api-reference.md#api-pareto-type1), [`log_survival_multiplier_ratio`](incerto-api-reference.md#api-log-survival-multiplier-ratio).

```python
from scipy.stats import expon, lognorm
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.tail_classes import log_survival_multiplier_ratio

for label, law in (("Pareto", pareto_type1(2.5)),
                   ("Lognormal", lognorm(1)), ("Exponential", expon())):
    log_ratio = log_survival_multiplier_ratio(law.logsf, [2, 10, 100])
    print(f"{label}: [" + ", ".join(f"{v:.4g}" for v in log_ratio) + "]")
```

**Output**

```text
Pareto: [-1.733, -1.733, -1.733]
Lognormal: [-1.081, -2.052, -3.563]
Exponential: [-2, -10, -100]
```

A finite grid illustrates the specified laws and cannot classify unknown data.

Figure regeneration: `uv run python demos/incerto/prologue_tail_classes.py` from the
repository root. It evaluates $\log A_2(u)$ for Pareto
$(\alpha=5/2,x_m=1)$, standard lognormal, Weibull $(\beta=1/2)$,
exponential (rate $1$), and gamma (shape $2$, rate $1$).
The fixed grid has 241 logarithmically spaced thresholds from $1$ to $10^4$;
all quantities are dimensionless population calculations. There is no
sampling, data fitting, truncation of a distribution, or uncertainty interval.

```{figure} ../assets/figures/incerto/prologue-tail-ratios.svg
:alt: Negative log multiplier ratios on logarithmic axes. Pareto is constant while the lognormal, stretched Weibull, exponential and gamma curves increase.

Theoretical survival functions evaluated on a finite grid. The ordinate is
$-\log[\bar F(2u)/\bar F(u)]$: an increasing curve means a decreasing
multiplier ratio. The lognormal and stretched-Weibull curves share their
zero-ratio limit with the two light-tailed models, although their convolution
tails belong to a different class.
The exponential curve (brick, circles) and gamma curve (teal, squares) nearly
coincide at high thresholds. Their ordinates are $u$ and
$u-\log[(1+2u)/(1+u)]$, respectively: the gap tends to $\log2$ while both
grow like $u$. Logarithmic axes make that relative gap small. Distinct markers
and line widths preserve both curves without shifting their values.
```

The script also prints exact exponential and gamma two-sum checks at
$u=2,10,100$. The helper uses log survival values for the multiplier figure;
non-finite values cause an error. In particular, it does not divide zeros
when an ordinary survival function underflows. The figure is numerical
illustration of the formulas above, not evidence of an observed tail class.

(tail-catalog-log-pareto)=
## A survival that varies slowly itself

The positive-exponent convention in the table does not exhaust regularly
varying functions. To examine the index-zero boundary, let
$V$ be Pareto with minimum $v_m>0$, exponent $a>0$, and dimensionless
values, and put $X=e^V$. Here $a,v_m$ are local parameters; the survival
exponent of $X$ will be zero. This construction is motivated by SCoFT v4
§5.5, pp. 102–103 [@taleb2025scoftv4]. Inverting the increasing transform
proves

$$
P(X>x)=\left(\frac{v_m}{\log x}\right)^a,\quad x\ge e^{v_m},
\qquad f_X(x)=\frac{a v_m^a}{x(\log x)^{a+1}}.
$$

For every fixed $t>0$, its survival ratio tends to one because
$\log(tx)/\log x\to1$. Thus $\bar F_X\in RV_0$, outside this catalog's
positive-power rows. Every positive power moment diverges:
$\mathbb EX^p=\mathbb E e^{pV}$, and the exponential factor eventually
exceeds every polynomial in the Pareto density integral.

We also prove subexponentiality directly for this example. For iid
nonnegative $X_1,X_2$, with $M_2=\max(X_1,X_2)$,

$$
P(M_2>x)\le P(X_1+X_2>x)\le P(M_2>x/2).
$$

Both bounds divided by $2\bar F_X(x)$ tend to one: the maximum-tail
formula is $2\bar F_X-\bar F_X^2$ and
$\bar F_X(x/2)/\bar F_X(x)\to1$. The squeeze proves the claim.
Thus a regularly varying survival can be subexponential even when its
index is zero. There is no finite positive $1/\alpha$
calibration at this index-zero boundary.

The chapter demo evaluates this survival using $\log x=2,4,8$ with
$a=2$, $v_m=1$, obtaining $1/4,1/16,1/64$. It keeps the thresholds in
logarithmic coordinates to avoid overflowing $e^V$. Finite evaluations
do not stand in for the moment-divergence or convolution proofs.

The [maximum-entropy note](incerto-maximum-entropy.md) gives another
construction of power tails, conditional on specified information constraints.
Neither construction establishes a tail law for observed data.

(tail-catalog-regimes)=
## Regime separation is not a tail class

Appendix A of SCoFT v4, §A.1, pp. 122–124 [@taleb2025scoftv4],
considers distributions with separated regimes. We can check the effect on
kurtosis without assuming a power tail. Let $B$ take $-1$ and $1$ with
equal probabilities, let $Z\sim N(0,1)$ be independent, and set
$X=mB+\sigma Z$, with $m\ge0$ and $\sigma>0$. Symmetry gives mean zero.
Expanding and discarding the odd terms proves

$$
\mathbb EX^2=m^2+\sigma^2,\qquad
\mathbb EX^4=m^4+6m^2\sigma^2+3\sigma^4,
\qquad
K=3-\frac{2m^4}{(m^2+\sigma^2)^2}.
$$

Here $K$ is Pearson kurtosis. For $m>0$ it lies below three and tends
to one as $m/\sigma\to\infty$. Yet
$\mathbb E e^{tX}=\cosh(mt)e^{\sigma^2t^2/2}<\infty$ for every real
$t$. This location mixture is light-tailed. Its density has a local
minimum at zero when $m>\sigma$: differentiating the two normal densities
gives $f''(0)=f(0)(m^2-\sigma^2)/\sigma^4>0$.
Thus low kurtosis can accompany a valley around the expectation.
It is not evidence of a small probability of a large regime change.

For $m=2$, $\sigma=1$, the variance is $5$ and $K=43/25=1.72$.
The [variance-mixture note](incerto-mixture-uncertainty.md) holds the location
fixed and mixes variances; these are different experiments. A finite normal
mixture has positive density everywhere, so a valley is not a forbidden
interval and every individual point still has probability zero.

(tail-catalog-absorbing)=
## An absorbing state needs repeated chances to be reached

For the §A.2 context, pp. 124–125, take a homogeneous two-state Markov
chain with states intact and failed. The failed state is absorbing and
an intact system fails with probability $p\in[0,1]$ at each step,
conditional on still being intact. Its row-stochastic transition matrix is

$$
T=\begin{pmatrix}1-p&p\\0&1\end{pmatrix},\qquad
T^n=\begin{pmatrix}(1-p)^n&1-(1-p)^n\\0&1\end{pmatrix}
\quad(n\ge1).
$$

Induction proves the power: multiplying by $T$ multiplies the remaining
intact probability by $1-p$. Starting intact, eventual failure has
probability one if $p>0$ and zero if $p=0$. Irreversibility alone is not
sufficient; reachability and persistent conditional risk matter. With varying
conditional hazards $p_j$, intact survival is $\prod_{j=1}^n(1-p_j)$.
For example, $p_j=2^{-j-1}$ has sum $1/2$, and
$\prod_{j=1}^n(1-p_j)\ge1-\sum_{j=1}^n p_j\ge1/2$ by induction.
Every step has positive risk but eventual failure is not certain.
The [existing survival calculation](incerto-ruin.md) treats this product;
we do not replace it with an independence assumption on unconditional events.

```{figure} ../assets/figures/incerto/multivariate-regimes.svg
:alt: Three vertical panels show separated normal-mixture densities, kurtosis falling from three toward one, and intact survival with constant versus decreasing conditional hazards.

Exact appendix-A examples. The location mixture uses $\sigma=1$, with
$m=0$ and $2$ in the density panel. The last panel compares conditional
hazard $0.05$ each step with $p_j=2^{-j-1}$, starting intact. The first
survival tends to zero; the second stays at least $1/2$. Neither construction
asserts a power-tail law. Figure regeneration: `uv run python demos/incerto/multivariate_tails.py`.
```

Use [`normal_location_mixture_kurtosis`](incerto-api-reference.md#api-normal-location-mixture-kurtosis), [`survival_from_hazards`](incerto-api-reference.md#api-survival-from-hazards).

```python
from xmath.incerto.multivariate import normal_location_mixture_kurtosis
from xmath.incerto.lecture_risk import survival_from_hazards

print(f"Location-mixture kurtosis: {normal_location_mixture_kurtosis(2):.4g}")
print(f"Constant hazard: {survival_from_hazards([0.05] * 100)[-1]:.4g}")
hazards = [2.0**(-j - 1) for j in range(1, 101)]
print(f"Decreasing hazard: {survival_from_hazards(hazards)[-1]:.4g}")
```

**Output**

```text
Location-mixture kurtosis: 1.72
Constant hazard: 0.005921
Decreasing hazard: 0.5776
```

Neither example asserts a power-tail law.

The [Pareto mixture contrast](incerto-pareto.md#pareto-regime-mixture)
shows what changes when one regime actually has a power tail.

## Caveats

- Regular variation and subexponentiality concern limits, not the shape of
  a fitted curve on a finite range.
- Very small survival probabilities can underflow in numerical calculations.
  Logarithmic ratios help preserve information; rounded zeros do not establish
  exact zero probability.
- The subexponential definition used here is the right-tail iid version for
  nonnegative summands. Two-sided and dependent settings need additional
  assumptions.
- The finite exact examples above check formulas, not the accuracy of an
  upstream sampler, survival-function implementation, or quadrature routine.

## Appendix: convolution calculation

This appendix proves the elementary two-summand convolution decomposition used
by the subexponential diagnostic.  We work with independent continuous
variables, then specialize to an iid lower-bounded distribution.  Discrete,
dependent, and two-sided variants require separate assumptions.

For independent continuous random variables $X_1$ and $X_2$ with densities
$f_1$, $f_2$ and survival functions $\bar F_1$, $\bar F_2$,

$$
\mathbb P(X_1+X_2>x)
=
\int_{-\infty}^{\infty} f_1(y)\bar F_2(x-y)\,dy.
$$

To see the iid reduction, take a common density $f$, survival function
$\bar F$, and lower support endpoint $a$, with $x>2a$.  Split the event
$\{X_1+X_2>x\}$ into three cases, ignoring probability-zero boundary points:
first $X_2\le x/2$, second $X_1\le x/2$, and third both variables exceed
$x/2$.  In the first case, conditioning on $X_2=y$ with $a\le y\le x/2$
leaves the requirement $X_1>x-y$, so this part contributes
$\int_a^{x/2} f(y)\bar F(x-y)\,dy$.  The second case contributes the same
quantity by iid symmetry.  The remaining upper-right square has probability
$\bar F(x/2)^2$ by independence.

(tail-class-catalog-convolution-split)=
For $a=0$, the partition can be recorded as disjoint regions rather than a
contour plot. Boundaries have zero probability under the continuous-law
assumption.

| Region within $\{X_1+X_2>x\}$ | Probability contribution |
| --- | --- |
| $0\le X_2\le x/2$, $X_1>x-X_2$ | $\int_0^{x/2}f(y)\bar F(x-y)\,dy$ |
| $0\le X_1\le x/2$, $X_2>x-X_1$ | The same integral by iid symmetry |
| $X_1>x/2$, $X_2>x/2$ | $\bar F(x/2)^2$ |

For iid variables with lower support endpoint $a$, symmetry gives the more
useful half-line formula

$$
\mathbb P(X_1+X_2>x)
=
2\int_a^{x/2} f(y)\bar F(x-y)\,dy+\bar F(x/2)^2.
$$

Dividing by $2\bar F(x)$ yields a normalized integral:

$$
\begin{aligned}
R(x)
&=\int_a^{x/2}\frac{f(y)\bar F(x-y)}{\bar F(x)}\,dy
  +\frac{\bar F(x/2)^2}{2\bar F(x)}\\
&=\int_a^{x/2}
\exp\{\log f(y)+\log\bar F(x-y)-\log\bar F(x)\}\,dy\\
&\quad+\frac12\exp\{2\log\bar F(x/2)-\log\bar F(x)\}.
\end{aligned}
$$

The logarithmic form applies where the factors are positive, with zero terms
understood by limits. It avoids forming tiny unnormalized probabilities before
taking their ratio. The decomposition is an ordinary conditioning argument;
no numerical integration or executable package helper is needed for it.

## References

- @taleb2025scoftv4, appendix A, §A.1, pp. 122–124, and §A.2,
  pp. 124–125: regime-mixture and transition context. The symmetric-location
  and absorbing-chain special cases are proved above.

- @taleb2025scoftv4, §5.0.1, pp. 91–93, and §5.5, pp. 102–103:
  classification and log-Pareto context. The new index-zero calculation is
  proved here; earlier positive classifications retain the references below.
- @bingham1987regular, *Regular Variation*.
- Embrechts, Klueppelberg, and Mikosch, *Modelling Extremal Events*
  [1997](https://doi.org/10.1007/978-3-642-33483-2).
- Foss, Korshunov, and Zachary, *An Introduction to Heavy-Tailed and
  Subexponential Distributions* [2013, 2nd ed.](https://doi.org/10.1007/978-1-4614-7101-1),
  §3.5, Theorem 3.29 and following paragraph, p. 57; Theorem 3.30 and
  following examples, p. 58. These support the cited positive classifications.
- Taleb, *Statistical Consequences of Fat Tails*, third edition (2025),
  [arXiv:2001.10488v4](https://arxiv.org/abs/2001.10488v4), ch. 1,
  terminology discussion, pp. 4–5: context for the companion Prologue guide,
  not a theorem source for this table.

## Backlinks

- Depends on: [Subexponentiality](./incerto-subexponentiality.md) and
  [Regular Variation](./incerto-regular-variation.md).
- Example distribution: [Pareto Distribution](./incerto-pareto.md).
- Related diagnostic: [Max-to-Sum Ratio](./incerto-max-to-sum-ratio.md).
