---
title: Generalized Central Limit Theorem
options:
  concept:
    id: generalized-central-limit-theorem
    type: theorem
    depends_on:
      - regular-variation
      - pareto-moment-existence
    proof_depends_on: [pareto]
    tags:
      - stable-laws
      - sums
      - infinite-variance
---

(generalized-clt-statement)=
## Statement

The ordinary central limit theorem uses finite variance and normalizes sums by
$\sqrt{n}$.  The substantive generalized central limit theorem is a
classification theorem: a nondegenerate distribution can occur as a weak limit
of centered and normalized iid sums if and only if it is stable.  In other
words, if there are constants $a_n>0$ and $b_n$ such that

$$
\frac{X_1+\cdots+X_n-b_n}{a_n}
\Rightarrow Z,
$$

for a nondegenerate limit $Z$, then $Z$ must be stable; conversely, stable laws
arise as such limits for suitable iid summands.  The Gaussian law is the stable
case $\alpha=2$.  The heavy-tailed stable cases have index $0<\alpha<2$.

For a two-sided regularly varying sufficient condition, assume

$$
\begin{gathered}
\mathbb P(|X|>x)=x^{-\alpha}L(x),\\
0<\alpha<2,
\end{gathered}
$$

where $L$ is slowly varying at infinity, with tail balance

$$
\begin{gathered}
\frac{\mathbb P(X>x)}{\mathbb P(|X|>x)}\to p,\\
\frac{\mathbb P(X<-x)}{\mathbb P(|X|>x)}\to q,\\
p+q=1.
\end{gathered}
$$

Choose $a_n$ so that $n\mathbb P(|X|>a_n)\to1$.  Then, with suitable
centering constants $b_n$,

$$
\frac{X_1+\cdots+X_n-b_n}{a_n}
\Rightarrow Z_\alpha,
$$

where $Z_\alpha$ is an $\alpha$-stable law whose skewness is determined by
$p-q$.  For nonnegative Pareto-type examples this reduces to a strongly
right-skewed stable limit, and the scaling is of order $n^{1/\alpha}$ up to a
slowly varying factor.  When $1<\alpha<2$, the mean exists but the variance is
infinite, so a common centering choice is $b_n=n\mathbb E[X]$.

A common centering summary is

$$
\begin{gathered}
b_n=\\
\begin{cases}
0, & 0<\alpha<1,\\
n\mathbb E[X\mathbf 1_{\{|X|\le a_n\}}], & \alpha=1,\\
n\mathbb E[X], & 1<\alpha<2.
\end{cases}
\end{gathered}
$$

The first row applies in the usual nonnegative case; the second is a
standard choice at $\alpha=1$.

Here $S_n=\sum_{i=1}^n X_i$ is the sum, $M_n=\max_{1\le i\le n}X_i$
is the maximum, $\mathbb E$ is expectation, and $\Rightarrow$ denotes
convergence in distribution. The tail exponent $\alpha$, norming constants
$a_n$, centering constants $b_n$, tail-balance weights $p,q$, and stable limit
$Z_\alpha$ are defined above. See [Shared notation](https://xshi19.github.io/math/notation/).

We record the stable-limit contrast needed by the site's heavy-tail
pages.  We do not classify every possible domain of attraction or every
parameterization of stable laws.

The classification and tail condition are cited from @durrett2019probability,
Theorems 3.8.8 (p. 191) and 3.8.2 (p. 186), with characteristic functions
in §3.8, p. 189. That theorem uses truncated centering; the conventions
above can change the limit's location. For exact Pareto with $1<\alpha<2$
and $a_n=x_m n^{1/\alpha}$, the difference between full and truncated centering divided by $a_n$ is
the constant $\alpha/(\alpha-1)$, since
$n\mathbb E[X\mathbf1_{\{X>a_n\}}]/a_n=\alpha/(\alpha-1)$.
For a scale satisfying only $nP(X>a_n)\to1$, this ratio converges to
the same constant. Thus full-mean centering preserves convergence, with
the limit shifted downward by $\alpha/(\alpha-1)$.
These theorems are not proved by the maximum calculation below.

## Why $\sqrt{n}$ fails

The normal law is not the only possible attractor for sums.  It is the
finite-variance attractor. For the regularly varying tails with $0<\alpha<2$
considered here, the largest observations remain visible at the scale of the
centered sum, and $\sqrt{n}$ is no longer the right normalization.

For a [Pareto-type tail](./incerto-pareto.md) with exponent
$\alpha<2$, the natural scale of the maximum is about $n^{1/\alpha}$.  Stable
normalization puts the centered sum on that same order. This is why infinite-variance
sums can keep producing large jumps instead of smoothing into Gaussian-looking
noise.

## Scaling argument

The full generalized central limit theorem is a stable-law classification
result, so we cite it rather than reproduce the proof.  What we prove
directly is the exact Pareto maximum scaling calculation, which shows why the
usual $\sqrt n$ scale is too small when $\alpha<2$.

For a [Pareto Type I](./incerto-pareto.md) variable with survival
$\bar F(x)=(x_m/x)^\alpha$, choose $a_n=x_m n^{1/\alpha}$.  Then for $y>0$ and sufficiently large $n$ such that $a_n y\ge x_m$,

$$
\begin{aligned}
&\mathbb P(M_n/a_n\le y)\\
&\quad=\left(1-\frac{1}{ny^\alpha}\right)^n\\
&\quad\to e^{-y^{-\alpha}}.
\end{aligned}
$$

The maximum remains of order $a_n$.  Since $a_n$ grows faster than
$\sqrt{n}$ when $\alpha<2$, the Gaussian finite-variance scaling is not the
right asymptotic scale for such tails.  This maximum calculation identifies the
right order of extreme observations; it is a scaling heuristic, not a proof of
stable convergence for sums.

## Pareto scaling comparison

(generalized-clt-scaling-simulation)=
Take exact Pareto Type I with $x_m=1$ and $\alpha=1.5$. Its mean is $3$
and its variance is infinite. With $S_n=\sum_{i=1}^n X_i$, compare

$$
\begin{gathered}
T_n=\frac{S_n-3n}{n^{2/3}},\\
G_n=\frac{S_n-3n}{\sqrt n}=n^{1/6}T_n.
\end{gathered}
$$

The identity holds for every sample. Consequently, whenever its interquartile
range is nonzero, the spread of $G_n$ is exactly $n^{1/6}$ times that of
$T_n$. Here are the normalization factors, rounded to three decimals:

| Sample size $n$ | Stable scale $n^{2/3}$ | Gaussian scale $\sqrt n$ | Ratio $n^{1/6}$ |
| --- | --- | --- | --- |
| $100$ | $21.544$ | $10$ | $2.154$ |
| $1000$ | $100$ | $31.623$ | $3.162$ |
| $10000$ | $464.159$ | $100$ | $4.642$ |

The cited stable-limit theorem implies convergence of the central quantiles
of $T_n$ to those of a nondegenerate continuous stable law. Thus the central
spread of $G_n$ grows on the $n^{1/6}$ scale. The table evaluates the
normalizations; it is neither simulated evidence nor a proof of stable
convergence.

```{figure} ../assets/figures/incerto/generalized-clt.svg
:alt: The 10th to 90th sampling spread of centered Pareto sums is steadier under n to the two-thirds normalization than under square-root normalization.

Pareto $\alpha=1.5$, $x_m=1$: 800 independent samples at each of
$n=30,100,300,1000,3000$, PCG64 seed 20261074. Both panels use the same
centered sums $S_n-3n$. Lines are medians and bands pointwise 10th/90th
sampling percentiles; they are not confidence intervals. Every repetition
contributes, without clipping. The lower panel's interquartile range is
exactly $n^{1/6}$ times the upper panel's for each simulated size, up to
roundoff. Finite differences between the correctly scaled distributions
are expected; the plot does not prove stable convergence.
```

Use [`repeated_centered_sums`](incerto-api-reference.md#api-repeated-centered-sums)
with [`pareto_type1`](incerto-api-reference.md#api-pareto-type1).

```python
import numpy as np
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.limits import repeated_centered_sums

n = np.array([100, 1000])
sums = repeated_centered_sums(pareto_type1(1.5), n, center=3,
                              repetitions=200, random_state=74)
for size, row in zip(n, sums):
    iqr = np.diff(np.quantile(row, [.25, .75]))[0]
    print(f"n={size}: stable IQR={iqr/size**(2/3):.4g}, sqrt IQR={iqr/size**.5:.4g}")
```

**Output**

```text
n=100: stable IQR=3.509, sqrt IQR=7.56
n=1000: stable IQR=3.381, sqrt IQR=10.69
```

Regenerate with `uv run python demos/incerto/limit_distributions.py`.

The next calculation concerns an exactly stable parent.

(stable-exact-sums)=
## Exact stable sums and their averages

We use a symmetric stable law with characteristic function

$$
\begin{gathered}
\varphi_X(t)=\exp(i\mu t-|\sigma t|^\alpha),\\
0<\alpha\le2,\quad\sigma>0.
\end{gathered}
$$

Here $\alpha$ is the stability index; for $\alpha<2$ it is also the
power-tail exponent. At $\alpha=2$ this is $N(\mu,2\sigma^2)$,
so $\sigma$ is not its standard deviation. The existence of these stable
laws is a cited input from @durrett2019probability, Example 3.3.23,
pp. 138–139 (with the Gaussian endpoint immediate).
Multiplying characteristic functions proves, for iid copies,

$$
\begin{gathered}
S_n-n\mu\overset d=n^{1/\alpha}(X_1-\mu),\\
\bar X_n-\mu\overset d=n^{1/\alpha-1}(X_1-\mu).
\end{gathered}
$$

For $1<\alpha\le2$ the scale shrinks. The finite absolute moment proved
below also makes the strong law applicable, so $\bar X_n\to\mu$ almost
surely. At $\alpha=1$ the average keeps its Cauchy law; at $\alpha<1$
the scale grows and $\mu$ is only a location. These exact identities must
not be transferred to a Pareto parent merely because its normalized sums
have a stable limit.

(stable-mean-deviation)=
## Mean absolute deviation of a symmetric stable law

For the preceding parameterization and $1<\alpha\le2$, we prove

$$
\begin{gathered}
\mathbb E|X-\mu|=\frac{2\sigma}{\pi}\Gamma(1-1/\alpha),\\
\begin{aligned}
&\mathbb E|\bar X_n-\mu|\\
&\quad=\frac{2\sigma}{\pi}\Gamma(1-1/\alpha)n^{1/\alpha-1}.
\end{aligned}
\end{gathered}
$$

Start with $|x|=(2/\pi)\int_0^\infty(1-\cos(tx))/t^2\,dt$.
For completeness, the integral at $x=1$ equals $\pi/2$: write
$u^{-2}=\int_0^\infty s e^{-su}\,ds$ and use Tonelli to get
$\int_0^\infty (1+s^2)^{-1}\,ds=\pi/2$. Scaling gives the identity
for all $x$. Tonelli again, followed by $v=(\sigma t)^\alpha$, gives

$$
\begin{aligned}
&\mathbb E|X-\mu|\\
&\quad=\frac{2\sigma}{\pi\alpha}
\int_0^\infty(1-e^{-v})v^{-1-1/\alpha}\,dv.
\end{aligned}
$$

Integration by parts makes the integral
$\alpha\Gamma(1-1/\alpha)$; both boundary terms vanish when $\alpha>1$.
For $\alpha\le1$, the integrand near zero is asymptotic to
$v^{-1/\alpha}$, so the absolute moment is infinite. Symmetry and
integrability give $\mathbb EX=\mu$ in the finite case. The average's
formula now follows from the exact scale identity.

Use [`symmetric_stable_mean_abs`](incerto-api-reference.md#api-symmetric-stable-mean-abs).

```python
from xmath.incerto.limits import symmetric_stable_mean_abs

for alpha in (1, 1.5, 2):
    print(f"alpha={alpha:g}: " + ", ".join(
        f"{symmetric_stable_mean_abs(alpha, n=n):.4g}" for n in (1, 100)))
```

**Output**

```text
alpha=1: inf, inf
alpha=1.5: 1.705, 0.3674
alpha=2: 1.128, 0.1128
```

(clt-cumulants)=
## Cumulants measure particular aspects of convergence

Let the iid summands have mean $\mu$, variance $0<\sigma^2<\infty$,
and finite absolute moment of integer order $r\ge3$. Define the local
$r$th cumulant by $\kappa_r(X)=i^{-r}(\log\varphi_X)^{(r)}(0)$,
using the continuous logarithm near $\varphi_X(0)=1$. For
$T_n=(S_n-n\mu)/(\sigma\sqrt n)$, multiplication of characteristic
functions and the chain rule prove

$$
\kappa_r(T_n)=n^{1-r/2}\frac{\kappa_r(X)}{\sigma^r}.
$$

Thus standardized skewness decays as $n^{-1/2}$ and excess kurtosis as
$n^{-1}$. Direct integration gives zero skewness and excess $-6/5$ for
$U(0,1)$; differentiating $-\log(1-t)$ gives skewness $2$ and excess
$6$ for $\mathrm{Exp}(1)$. For iid sums, divide these excesses by $n$.
These calculations describe moments, not a distance between CDFs.
Zero excess kurtosis does not identify a Gaussian law: a variable taking
$0,\sqrt3,-\sqrt3$ with probabilities $2/3,1/6,1/6$ has variance one
and fourth moment three, but is discrete.

For a CDF error bound, the cited Berry–Esseen theorem
(@durrett2019probability, Theorem 3.4.17, p. 157) requires
$\rho=\mathbb E|X-\mu|^3<\infty$ and gives
$\sup_x|P(T_n\le x)-\Phi(x)|\le3\rho/(\sigma^3\sqrt n)$.
The constant is sufficient, not optimal. For $U(0,1)$,
$\rho/\sigma^3=3\sqrt3/4$; for $\mathrm{Exp}(1)$ it is $12/e-2$,
obtained by splitting the centered absolute-moment integral at one.
These upper bounds do not prove a universal ranking of convergence speeds.

For exact Pareto, the [moment gate](incerto-pareto-moment-existence.md)
requires $\alpha>2$ for finite variance, $\alpha>3$ for Berry–Esseen's
third absolute moment, and $\alpha>4$ for excess kurtosis. The source's
$\alpha=2$ example has infinite variance. Its Gaussian-domain scaling
requires an additional logarithmic factor; it is not a finite-variance
$\sqrt n$ example. For $1<\alpha<2$, the non-Gaussian limit above applies.
The Lyapunov and Lindeberg extensions in SCoFT §7.5 need their own
array/moment conditions; the iid examples here do not prove those theorems.

## Caveats

- Stable-law parameterizations vary across books and software. The exact
  symmetric calculations specify their characteristic function; the general
  attraction statement does not fix a software skew/location convention.
- The displayed regular-variation and tail-balance condition covers the
  non-Gaussian stable range $0<\alpha<2$; the Gaussian boundary needs a
  different criterion.
- Right-tail regular variation alone is not enough for a general two-sided
  variable; the left tail can change the scale or skewness of the limit.
- Some infinite-variance distributions can still be in the Gaussian domain of
  attraction when their truncated second moment is slowly varying.
- For $0<\alpha<1$, the absolute first moment is infinite. At $\alpha=1$,
  its finiteness depends on the slowly varying factor: the tail-integral
  criterion is $\int^\infty L(x)/x\,dx<\infty$. For example, a nonnegative
  variable with survival $1/[x(\log x)^2]$ for $x\ge e$ has a finite mean,
  since that tail integral equals $1$. Exact Pareto tails have infinite mean
  at $\alpha=1$. See the boundary discussion in [Karamata's theorem](./incerto-karamata.md).
  The truncated centering above remains a standard choice at this boundary;
  finite mean alone does not justify replacing it by $n\mathbb E[X]$ without
  checking the resulting location shift on the $a_n$ scale.
- Finite samples can look calmer than the asymptotic theory suggests until a
  large observation arrives.

## References

- Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*
  [1971, 2nd ed., Wiley](https://www.wiley-vch.de/de/fachgebiete/mathematik-und-statistik/an-introduction-to-probability-theory-and-its-applications-volume-2-978-0-471-25709-7).
- Resnick, *Heavy-Tail Phenomena* @resnick2007heavy.
- @taleb2025scoftv4, §§7.2–7.5, pp. 139–148, and §7.7, pp. 151–152: chapter context.
- @durrett2019probability, with precise theorem locators at use.

## Backlinks

- Depends on: [Regular Variation](./incerto-regular-variation.md) and
  [Pareto Moment Existence](./incerto-pareto-moment-existence.md).
- Related to: [LLN Failure Under Infinite Mean](./incerto-lln-failure.md).
- Used by: [Pre-Asymptotic LLN Behavior](./incerto-lln-preasymptotic.md).
