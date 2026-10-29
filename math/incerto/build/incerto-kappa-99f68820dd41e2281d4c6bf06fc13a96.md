---
title: Kappa and finite-sample concentration
options:
  concept:
    id: kappa
    type: method
    prerequisites:
      - pareto-moment-existence
      - generalized-central-limit-theorem
    related:
      - moment-estimation
      - dispersion-ratio
      - mixture-uncertainty
    tags:
      - concentration
      - finite-samples
      - sums
---

How much does averaging reduce mean absolute error at a specified sample
size? Kappa records the growth of the sum's absolute deviation between two
sizes. It depends on the whole distribution and the chosen sizes. A tail
exponent alone does not determine it.

We use the metric from @taleb2025scoftv4, ch. 8, §§8.1–8.3, pp. 154–160.
The derivations below establish the stated special cases and corrections;
simulated values are labeled separately. The [chapter guide](incerto-scoft-ch8.md)
maps the remaining source sections.

(kappa-definition)=
## Definition and domain

Let $X_i$ be iid, with $\mathbb E|X_1|<\infty$, mean $\mu$, and sums
$S_n=\sum_{i=1}^nX_i$, as in the
[shared notation](https://xshi19.github.io/math/notation/). Define

$$
\begin{aligned}
M(n)&=\mathbb E|S_n-n\mu|,\\
A(n)&=\mathbb E|\bar X_n-\mu|\\
&=M(n)/n.
\end{aligned}
$$

$M$ is a mean absolute deviation about the **population mean**, without
Gaussian calibration. It is neither a median deviation nor a deviation
about an estimated center; see the [dispersion note](incerto-dispersion-ratio.md).
For integers $n>n_0\ge1$, with $M(n)>M(n_0)>0$, set

$$
\begin{aligned}
\kappa_{n_0,n}
&=2-\frac{\log(n/n_0)}{\log(M(n)/M(n_0))},\\
\frac{M(n)}{M(n_0)}
&=(n/n_0)^{1/(2-\kappa_{n_0,n})}.
\end{aligned}
$$

We always print both indices. The source's $\kappa_1$ means
$\kappa_{1,2}$, whereas $\kappa_{1,n}$ compares size $n$ with one
observation. An adjacent-size comparison $\kappa_{n,n+1}$ is different.
Here “metric” names a concentration index, not a distance satisfying the
axioms of a metric space.

We prove three elementary properties. Adding a constant to every summand
cancels under centering; multiplying by $b\ne0$ multiplies both deviations
by $|b|$. Thus kappa is invariant under location and nonzero scale changes.
Conditional Jensen, applied to an independent centered increment, gives
$M(n+1)\ge M(n)$. The triangle inequality gives $M(n)\le nM(1)$.
Consequently $\kappa_{1,n}\le1$ wherever the displayed definition applies.
There is no universal lower bound of zero.

For a chain $n_0<n_1<\cdots<n_r$, logarithmic growth adds:

$$
\begin{aligned}
&\log\frac{M(n_r)}{M(n_0)}\\
&\quad=\sum_{j=1}^r
\frac{\log(n_j/n_{j-1})}{2-\kappa_{n_{j-1},n_j}}.
\end{aligned}
$$

This is a telescoping identity, with each intermediate kappa defined.
It does not justify averaging the kappas themselves.

Use [`kappa_from_mads`](incerto-api-reference.md#api-kappa-from-mads)
and [`stable_kappa`](incerto-api-reference.md#api-stable-kappa) to expose
the boundaries explicitly:

```python
from xmath.incerto.concentration import kappa_from_mads, stable_kappa

for m0, m in ((1, 2**0.5), (2/3, 14/15), (1, 1), (0, 0), (float("inf"), float("inf"))):
    print(f"MADs {m0:.4g}, {m:.4g}: {kappa_from_mads(m0, m, 2):z.4f}")
for alpha in (1, 1.5, 2):
    print(f"Stable alpha={alpha:g}: {stable_kappa(alpha):.4g}")
```

**Output**

```text
MADs 1, 1.414: 0.0000
MADs 0.6667, 0.9333: -0.0600
MADs 1, 1: -inf
MADs 0, 0: nan
MADs inf, inf: nan
Stable alpha=1: nan
Stable alpha=1.5: 0.5
Stable alpha=2: 0
```

The equal-positive case uses an extended limit. Constant laws and laws
without a finite absolute first moment have no kappa under this definition.
In particular, a [Cauchy law](incerto-cauchy-running.md) is a limiting
benchmark as the stable index decreases to one, not an admissible endpoint.

(kappa-stable)=
## Stable benchmark and attraction

For a nondegenerate stable law of index $1<\alpha\le2$, the centered sum
has the same distribution as $n^{1/\alpha}(X_1-\mu)$. Taking absolute
expectations proves

$$
\begin{aligned}
M(n)&=n^{1/\alpha}M(1),\\
\kappa_{n_0,n}&=2-\alpha,\\
A(n)/M(1)&=n^{1/\alpha-1}.
\end{aligned}
$$

The stable sum identity is the cited structural property; the algebraic
consequence for kappa is proved here. Skewness does not change the exponent,
so it cancels from kappa even though it can change $M(1)$. At $\alpha=2$
the law is Gaussian and kappa is zero. The growth exponent is $1/\alpha$;
it must not be substituted for $\alpha$ in $2-\alpha$.
The [stable-law note](incerto-generalized-central-limit-theorem.md#stable-exact-sums)
fixes the characteristic-function scale and proves the symmetric MAD formula.

A domain-of-attraction statement is weaker than the exact stable identity.
For example, suppose a centered sum divided by $a_n$ converges in law to
an integrable stable variable $Z$, the normalized absolute sums are uniformly
integrable, and $\log a_n/\log n\to1/\alpha$. Then

$$
M(n)/a_n\longrightarrow\mathbb E|Z|
$$

and substitution gives $\kappa_{1,n}\to2-\alpha$. Uniform integrability
is what permits convergence of expectations; weak convergence alone does
not supply it. This conditional argument does not assert a finite-sample
rate for every regularly varying law. A one-sided Pareto and a symmetric
Student law with the same tail exponent can have different finite kappas:
both their centers and their distribution bodies differ. The numerical
comparison below concerns exponent three only.

(kappa-student)=
## Student sums and a logarithmic limit

Use the ordinary Student law with degrees of freedom $\nu>1$, location
zero and scale one. Locally $\nu$ is also its power-tail exponent. A direct
beta integral gives

$$
M(1)=\sqrt{\frac\nu\pi}\,
\frac{\Gamma((\nu-1)/2)}{\Gamma(\nu/2)}.
$$

For completeness, substitute $v=1+x^2/\nu$ in
$2\int_0^\infty x f_\nu(x)\,dx$; the remaining integral is
$\int_1^\infty v^{-(\nu+1)/2}\,dv=2/(\nu-1)$.
The gamma recurrence yields the displayed expression.

There is also a two-summand closed form. Write $a=\nu/2$ and
$X_i=\sqrt{\nu/(2G_i)}Z_i$, where the $G_i$ are independent
Gamma$(a,1)$ and the $Z_i$ are independent standard normals, independent
of the $G_i$. Set $T=G_1+G_2$, $B=G_1/T$. The change of variables has
Jacobian $T$ and factorizes the joint density into independent
Gamma$(2a,1)$ and Beta$(a,a)$ densities. Conditional normal absolute
moments therefore give

$$
\begin{aligned}
M(2)&=\sqrt{\nu/\pi}\,
\mathbb E[T^{-1/2}]\\
&\quad\times\mathbb E[(B(1-B))^{-1/2}],\\
\frac{M(2)}{M(1)}
&=\sqrt\pi\,2^{2-\nu}
\frac{\Gamma(\nu-1/2)}{\Gamma(\nu/2)^2}.
\end{aligned}
$$

The second line follows by the gamma and beta integrals, followed by the
gamma duplication identity. These are finite precisely when $\nu>1$.

### Cubic Student: every integer size

At $\nu=3$, the characteristic function is
$\phi(t)=(1+\sqrt3|t|)e^{-\sqrt3|t|}$.
One way to verify it is to differentiate the Cauchy Fourier integral
$\int_{\mathbb R}e^{itx}(x^2+a^2)^{-1}dx=\pi e^{-a|t|}/a$
with respect to $a>0$ and use the normalized squared-denominator density
at $a=\sqrt3$. The Cauchy integral follows by closing the contour in the
half-plane where the exponential decays and taking the residue at
$ia$ or $-ia$; the arc integral tends to zero. Differentiation is justified
by an integrable bound in a neighborhood of any positive $a$.

We now derive the sum MAD, using the
[absolute-moment Fourier identity](incerto-generalized-central-limit-theorem.md#stable-mean-deviation).
Integration by parts, then $u=\sqrt3t$, gives

$$
\begin{aligned}
I_n&=\int_0^\infty(1+u)^{n-1}e^{-nu}\,du,\\
M(n)&=\frac{2\sqrt3}{\pi}\,nI_n,\\
\frac{M(n)}{M(1)}
&=e^n n^{1-n}\Gamma(n,n),\\
\Gamma(n,n)&=\int_n^\infty t^{n-1}e^{-t}\,dt.
\end{aligned}
$$

The boundary term vanishes at zero because $1-\phi(t)^n=O(t^2)$,
and at infinity because the numerator is bounded. The substitution
$t=n(1+u)$ gives the incomplete-gamma expression. This also holds at $n=1$.
Repeated integration by parts yields the independent finite formula

$$
\frac{M(n)}{M(1)}
=\frac{n!}{n^n}\sum_{j=0}^{n-1}\frac{n^j}{j!}.
$$

For the limit, $e^{-n}\sum_{j<n}n^j/j!=P(N_n<n)\to1/2$,
where $N_n$ is Poisson with mean $n$. Apply the classical CLT to a sum
of $n$ independent Poisson$(1)$ variables. Stirling's formula then gives

$$
\begin{aligned}
\frac{M(n)}{M(1)}&\sim\sqrt{\pi n/2},\\
\log n\;\kappa_{1,n}&\longrightarrow2\log(\pi/2).
\end{aligned}
$$

This constant differs from the one printed in §8.6.1, p. 165.
The calculation concerns the kappa anchored at one observation. It is
not a bound on the CDF error in a normal approximation, nor on tail-quantile
error. It does not prove that a million-term standardized sum is poorly
approximated by a Gaussian at every threshold.

(kappa-finite-variance)=
### What the logarithmic rate measures

More generally, take any iid law with variance $0<\sigma^2<\infty$.
The classical CLT is cited in the
[limit-law note](incerto-generalized-central-limit-theorem.md).
The normalized centered sums have second moment $\sigma^2$; hence
$\mathbb E[|Y_n|\mathbf1_{|Y_n|>K}]\le\sigma^2/K$ for
$Y_n=(S_n-n\mu)/\sqrt n$. This proves uniform integrability, so

$$
\begin{aligned}
M(n)/\sqrt n&\longrightarrow c,\\
c&=\sigma\sqrt{2/\pi},\\
\log n\;\kappa_{1,n}
&\longrightarrow4\log\frac{c}{M(1)}.
\end{aligned}
$$

To obtain the last line, write
$\log(M(n)/M(1))=\tfrac12\log n+\log(c/M(1))+o(1)$
and substitute in the definition. The logarithmic behavior can reflect
the mismatch between the one-observation MAD and its Gaussian calibration.
If $c=M(1)$, the constant vanishes and this argument gives no nonzero
leading rate. Kappa approaching zero does not identify the far tail.

(kappa-gamma)=
## Exponential and gamma sums

Let each summand have Gamma shape $r>0$ and scale $\theta>0$.
Here $\theta$ is a local scale, not a natural parameter. The sum has
shape $a=nr$, mean $a\theta$, and

$$
M(n)=\frac{2\theta\,a^a e^{-a}}{\Gamma(a)}.
$$

We derive this using
$\mathbb E|V-\mathbb EV|=2\mathbb E(V-\mathbb EV)_+$.
After scaling by $\theta$, the integral is

$$
\frac{2\theta}{\Gamma(a)}
\int_a^\infty(t-a)t^{a-1}e^{-t}\,dt.
$$

Integration by parts gives
$\Gamma(a+1,a)=a\Gamma(a,a)+a^ae^{-a}$,
which evaluates the integral. At $r=1$, the parent is exponential
with rate $1/\theta$. Stirling's formula or the preceding finite-variance
argument gives

$$
\log n\;\kappa_{1,n}
\longrightarrow4-2\log(2\pi).
$$

Thus even the exponential law has a positive finite kappa and this
logarithmic anchored rate. It still satisfies the
[Cramér condition](incerto-cramer-condition.md) and is not subexponential.

Use [`gaussian_sum_mad`](incerto-api-reference.md#api-gaussian-sum-mad),
[`student_sum_mad`](incerto-api-reference.md#api-student-sum-mad),
[`gamma_sum_mad`](incerto-api-reference.md#api-gamma-sum-mad), and
[`kappa_from_mads`](incerto-api-reference.md#api-kappa-from-mads).

```python
from xmath.incerto.concentration import (
    gaussian_sum_mad, student_sum_mad, gamma_sum_mad, kappa_from_mads,
)

for name, mad in (("Gaussian", gaussian_sum_mad), ("Student 3", student_sum_mad), ("Exponential", gamma_sum_mad)):
    for n in (2, 32, 1000):
        print(f"{name}, n={n}: kappa={kappa_from_mads(mad(1), mad(n), n):z.3f}")
```

**Output**

```text
Gaussian, n=2: kappa=0.000
Gaussian, n=32: kappa=0.000
Gaussian, n=1000: kappa=0.000
Student 3, n=2: kappa=0.290
Student 3, n=32: kappa=0.188
Student 3, n=1000: kappa=0.118
Exponential, n=2: kappa=0.206
Exponential, n=32: kappa=0.087
Exponential, n=1000: kappa=0.046
```

```{figure} ../assets/figures/incerto/kappa-aggregation.svg
:name: kappa-aggregation-figure
:alt: Two stacked panels compare relative mean absolute errors and kappas for Gaussian, cubic Student, exponential and stable index 1.5 laws.

Population curves with each one-observation MAD scaled to one. The stable
curve uses index 1.5. The Student and exponential curves use exact formulas;
their kappas compare each size with one observation. The upper panel also
describes an equal-weight portfolio of independent, identically distributed
centered exposures. It makes no claim about empirical returns or optimal weights.
```

(kappa-lognormal)=
## Lognormal sums

Let $X=e^{\mu_\ell+\sigma_\ell Z}$ with standard-normal $Z$ and
$\sigma_\ell>0$. These are log-location and log-scale, not the mean and
standard deviation of $X$. Write $m=\mathbb EX=e^{\mu_\ell+\sigma_\ell^2/2}$.
Completing the square in the truncated normal integral gives

$$
M(1)=2m\left[2\Phi(\sigma_\ell/2)-1\right].
$$

For example, use $M(1)=2\mathbb E(X-m)_+$ and the threshold
$Z=\sigma_\ell/2$. Exponential tilting shifts the normal mean by
$\sigma_\ell$, giving the formula. Sums of independent lognormals are
not generally lognormal. Matching a sum's first two moments to another
lognormal gives an approximation, not an identity or a proved bound on kappa.

The mismatch can be checked with third moments. For a mean-one lognormal,
put $v=e^{\sigma_\ell^2}-1>0$. Its moments
$\mathbb EV^j=e^{j(j-1)\sigma_\ell^2/2}$ give variance $v$ and third
centered moment $v^2(v+3)$. Independence makes the variance of the average
$v/n$ and its third centered moment $v^2(v+3)/n^2$. The skewnesses of
the actual average and a moment-matched lognormal are therefore

$$
\begin{aligned}
g_{\rm sum}&=(v+3)\sqrt{v/n},\\
g_{\rm fit}&=(v/n+3)\sqrt{v/n}.
\end{aligned}
$$

They differ for every $n>1$. This proves that the fitted lognormal cannot
be the exact sum law; it does not quantify its approximation error.

We prove the three limits at the scope used in §8.6.2, pp. 166–168.
For fixed $\sigma_\ell$, finite variance and the previous argument give
$\kappa_{1,n}\to0$ as $n\to\infty$.
For fixed integer $n>1$, scale invariance lets us use the mean-one variable
$V_\sigma=e^{\sigma Z-\sigma^2/2}$. Define its Gaussian approximation
error $D_\sigma$. As $\sigma\downarrow0$,

$$
\begin{aligned}
D_\sigma&=\frac{V_\sigma-1}{\sigma}-Z,\\
\mathbb E D_\sigma^2&=\frac{e^{\sigma^2}-1}{\sigma^2}-1\\
&\longrightarrow0.
\end{aligned}
$$

This identity follows from $\mathbb E V_\sigma^2=e^{\sigma^2}$ and
$\mathbb E[ZV_\sigma]=\sigma$. A finite sum preserves $L^2$ convergence,
which implies convergence of absolute expectations. Thus the MAD ratio
tends to $\sqrt n$, and kappa tends to zero.

As $\sigma\to\infty$, couple a fixed finite set of normal variables.
Each $V_{\sigma,i}\to0$ almost surely, although its mean remains one.
For $T_\sigma=\sum_{i=1}^nV_{\sigma,i}$,

$$
\begin{aligned}
&\mathbb E|T_\sigma-n|\\
&\quad=2n-2\mathbb E\min(T_\sigma,n),\\
&\mathbb E\min(T_\sigma,n)\longrightarrow0.
\end{aligned}
$$

The last line is bounded convergence. Therefore $M(n)/M(1)\to n$
and $\kappa_{1,n}\to1$. The order of limits matters. Every fixed lognormal
law still has all power moments and belongs to the Gaussian domain of attraction.

### A bounded numerical calculation

For nonnegative integrable sums, the exact lower-deviation identity is

$$
M(n)=2\mathbb E(nm-S_n)_+.
$$

For the mean-one lognormal, condition on the first $n-1$ summands and put
$a=n-\sum_{i<n}V_{\sigma,i}$. If $a\le0$, the conditional payoff is zero.
For $a>0$, completing the square gives

$$
\begin{aligned}
h(a)&=\mathbb E(a-V_\sigma)_+,\\
d&=\log(a)/\sigma+\sigma/2,\\
h(a)&=a\Phi(d)-\Phi(d-\sigma).
\end{aligned}
$$

The simulated payoff lies between zero and $n$. Multiplying its sample
average by $2m$ estimates $M(n)$ without estimating an unbounded upper-tail
payoff. This uses a known model mean. It is not available merely because a
dataset contains only small observations.

Use [`lognormal_sum_mad`](incerto-api-reference.md#api-lognormal-sum-mad).
The standard error below describes Monte Carlo integration, not uncertainty
about the assumed lognormal law.

```python
from xmath.incerto.concentration import lognormal_sum_mad, kappa_from_mads

for sigma in (0.5, 1, 2):
    single = lognormal_sum_mad(1, sigma)["mad"]
    result = lognormal_sum_mad(8, sigma, random_state=20261088)
    kappa = kappa_from_mads(single, result["mad"], 8)
    print(f"sigma={sigma:g}: MAD={result['mad']:.3f}, MC SE={result['mc_se']:.3f}, kappa={kappa:.3f}")
```

**Output**

```text
sigma=0.5: MAD=1.343, MC SE=0.011, kappa=0.108
sigma=1: MAD=4.413, MC SE=0.032, kappa=0.338
sigma=2: MAD=50.405, MC SE=0.254, kappa=0.707
```

```{figure} ../assets/figures/incerto/kappa-lognormal.svg
:name: kappa-lognormal-figure
:alt: Lognormal kappa estimates versus log-scale at aggregation sizes two and sixteen, with vertical Monte Carlo error bars.

Conditional Monte Carlo, 30,000 independent replications per point, PCG64
seeds 20261100–20261121. Error bars transform MAD plus or minus two
estimated Monte Carlo standard errors through the kappa formula, holding
the exact single-observation MAD fixed. They are numerical diagnostics,
not guaranteed coverage intervals. No sample is truncated or discarded.
```

(kappa-negative)=
## Negative kappa needs no bimodality

Consider the symmetric triangular density $f(x)=(2-|x|)/4$ on $[-2,2]$.
It has a single mode at zero and can be written as the sum of two
independent Uniform$[-1,1]$ variables. Direct integration gives $M(1)=2/3$.
The sum of two copies is a sum of four such uniforms and has $M(2)=14/15$.
Here is an explicit calculation of the second value. If $T$ is a sum of
four Uniform$[0,1]$ variables, symmetry and integration of its CDF give

$$
\begin{aligned}
\mathbb E|2T-4|
&=4\int_0^2P(T<t)\,dt,\\
P(T<t)&=\frac{t^4-4(t-1)_+^4}{24},\\
&\hspace{2em}0\le t\le2,\\
\mathbb E|2T-4|&=\frac{4(32-4)}{120}=\frac{14}{15}.
\end{aligned}
$$

The CDF formula is inclusion–exclusion for the volume below
$u_1+\cdots+u_4<t$ in the unit cube. Thus

$$
\kappa_{1,2}=2-\frac{\log2}{\log(7/5)}<0.
$$

This counterexample corrects the bimodality suggestion in §8.4.4, p. 162.
Negative kappa says only that this MAD grows more slowly than the Gaussian
benchmark across the chosen sizes.

The source's separated Gaussian mixture is another example. For
$X=aR+\sigma Z$, with independent equiprobable $R\in\{-1,1\}$ and
standard-normal $Z$, condition on the number $K$ of positive signs:

$$
\begin{aligned}
K&\sim\operatorname{Binomial}(n,1/2),\\
S_n\mid K&\sim N(a(2K-n),n\sigma^2).
\end{aligned}
$$

If $H(b,s)=\mathbb E|b+sZ|$, splitting the normal integral at $-b/s$ gives

$$
\begin{aligned}
H(b,s)&=s\sqrt{2/\pi}\,e^{-b^2/(2s^2)}\\
&\quad+|b|[2\Phi(|b|/s)-1],\\
M(n)&=\sum_{k=0}^n\binom nk2^{-n}\\
&\quad\times H(a(2k-n),\sigma\sqrt n).
\end{aligned}
$$

This finite mixture calculation underlies the figure. In particular,
$M(1)=H(a,\sigma)$ and
$M(2)=\{H(2a,\sqrt2\sigma)+H(0,\sqrt2\sigma)\}/2$.
For fixed $\sigma>0$ and $a\to\infty$, their ratio tends to one from
above, so $\kappa_{1,2}\to-\infty$.

Use [`normal_location_mixture_sum_mad`](incerto-api-reference.md#api-normal-location-mixture-sum-mad).

```python
from xmath.incerto.concentration import normal_location_mixture_sum_mad, kappa_from_mads

for a in (0, 1, 3):
    m1 = normal_location_mixture_sum_mad(1, a)
    m2 = normal_location_mixture_sum_mad(2, a)
    print(f"separation={a:g}: kappa={kappa_from_mads(m1, m2, 2):.3f}")
```

**Output**

```text
separation=0: kappa=0.000
separation=1: kappa=-0.134
separation=3: kappa=-2.028
```

```{figure} ../assets/figures/incerto/kappa-shape.svg
:name: kappa-shape-figure
:alt: Two panels compare Pareto and Student finite-sum kappas at exponent three and negative kappas of a separated normal mixture and a unimodal triangular law.

Top: exact cubic Student curve and estimates for Pareto exponent 3, minimum
1, centered at its true mean $3/2$. Pareto points use 20,000 independent
replications and seeds 20261090–20261093, with MAD-based error bars as
above; the single-observation Pareto MAD is exact. This comparison changes
the distribution body as well as its asymmetry. Bottom: exact Gaussian
mixture calculations with $\sigma=1$, plus the triangular counterexample.
Neither panel establishes a universal ordering by skewness or kurtosis.
```

## Reproduce and continue

`uv run python demos/incerto/concentration.py` regenerates these figures
and the [sample-sufficiency figure](incerto-moment-estimation.md#moment-estimation-sufficiency).
All curves are population calculations or explicitly labeled simulations.
The [sample-size calculation](incerto-moment-estimation.md#moment-estimation-sufficiency)
uses absolute error, not a universal threshold for confidence-interval validity.

Source locators: @taleb2025scoftv4, ch. 8, §§8.2–8.4, pp. 156–162,
and §§8.6.1–8.6.4, pp. 164–169. The attached paper @taleb2018kappa
contains the earlier definition and derivations in §§II–IV and appendices
D–G. The proofs, counterexample, computations and figures here are original
treatments of that scope.
