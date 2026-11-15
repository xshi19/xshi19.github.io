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
\frac{\mathbb P(X<-x)}{\mathbb P(|X|>x)}\to 1-p,\\
0\le p\le1.
\end{gathered}
$$

Choose $a_n$ so that $n\mathbb P(|X|>a_n)\to1$.  Then, with suitable
centering constants $b_n$,

$$
\frac{X_1+\cdots+X_n-b_n}{a_n}
\Rightarrow Z_\alpha,
$$

where $Z_\alpha$ is an $\alpha$-stable law whose skewness is determined by
$2p-1$.  For nonnegative Pareto-type examples this reduces to a strongly
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
$a_n$, centering constants $b_n$, right-tail weight $p$ (left-tail weight
$1-p$), and stable limit $Z_\alpha$ are defined above.
See [Shared notation](https://xshi19.github.io/math/notation/).

We record the stable-limit contrast needed by the site's heavy-tail
pages.  We do not classify every possible domain of attraction or every
parameterization of stable laws.

The classification and tail condition are cited from @durrett2019probability,
Theorems 3.8.8 (p. 191) and 3.8.2 (p. 186), with characteristic functions
in §3.8, p. 189. That theorem uses truncated centering; the conventions
above can change the limit's location. These theorems are cited, not proved
by the maximum calculation below. We calculate the centering difference in
the exact Pareto case.

:::{prf:proposition} Full and truncated Pareto centering
:label: prop-generalized-central-limit-theorem-pareto-centering

Let $X$ be Pareto Type I with cutoff $x_m>0$ and $1<\alpha<2$.
For $a_n=x_m n^{1/\alpha}$,

$$
\frac{n\mathbb E[X]-n\mathbb E[X\mathbf1_{\{X\le a_n\}}]}{a_n}
=\frac{\alpha}{\alpha-1}.
$$

If instead $a_n>0$ satisfies $nP(X>a_n)\to1$, the ratio converges to
$\alpha/(\alpha-1)$. Consequently, if the iid sum centered by the truncated
mean and divided by $a_n$ converges in distribution to $Z$, full-mean
centering gives the limit $Z-\alpha/(\alpha-1)$.
:::

:::{prf:proof}
For $a\ge x_m$, integrating the Pareto density gives
$\mathbb E[X\mathbf1_{\{X>a\}}]
=\alpha x_m^\alpha a^{1-\alpha}/(\alpha-1)$.
Thus the ratio is $\alpha nP(X>a_n)/(\alpha-1)$ for all sufficiently
large $n$, proving both claims about centering. Subtracting this
deterministic ratio from the truncated-centered sum gives the stated shift.
:::

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

:::{prf:proposition} Pareto maximum scaling
:label: prop-generalized-central-limit-theorem-pareto-maximum

Let $X_1,X_2,\ldots$ be iid [Pareto Type I](./incerto-pareto.md) with
cutoff $x_m>0$, exponent $\alpha>0$, and survival
$\bar F(x)=(x_m/x)^\alpha$ for $x\ge x_m$. Set
$M_n=\max_{1\le i\le n}X_i$ and $a_n=x_m n^{1/\alpha}$.
For each fixed $y>0$ and all sufficiently large $n$ such that $a_ny\ge x_m$,

$$
\begin{aligned}
&\mathbb P(M_n/a_n\le y)\\
&\quad=\left(1-\frac{1}{ny^\alpha}\right)^n\\
&\quad\to e^{-y^{-\alpha}}.
\end{aligned}
$$
:::

:::{prf:proof}
Independence gives $P(M_n\le a_ny)=F(a_ny)^n$; substituting the
Pareto survival gives the equality. Since
$n\log(1-y^{-\alpha}/n)\to-y^{-\alpha}$, the CDF has the claimed limit.
:::

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
\mathbb E e^{itX}=\exp(i\mu t-|\sigma t|^\alpha),\\
0<\alpha\le2,\quad\sigma>0.
\end{gathered}
$$

Here $\alpha$ is the stability index; for $\alpha<2$ it is also the
power-tail exponent. At $\alpha=2$ this is $N(\mu,2\sigma^2)$,
so $\sigma$ is not its standard deviation. The existence of these stable
laws is a cited input from @durrett2019probability, Example 3.3.23,
pp. 138–139 (with the Gaussian endpoint immediate).
The exact scaling identity follows by multiplying characteristic functions.

:::{prf:lemma} Exact scaling of symmetric stable sums
:label: lem-generalized-central-limit-theorem-stable-scaling

Let $X_1,\ldots,X_n$ be iid with characteristic function
$\mathbb E e^{itX_1}=\exp(i\mu t-|\sigma t|^\alpha)$,
where $\mu\in\mathbb R$, $\sigma>0$, $0<\alpha\le2$, and $n\ge1$.
For $S_n=\sum_{i=1}^n X_i$ and $\bar X_n=S_n/n$,

$$
\begin{gathered}
S_n-n\mu\overset d=n^{1/\alpha}(X_1-\mu),\\
\bar X_n-\mu\overset d=n^{1/\alpha-1}(X_1-\mu).
\end{gathered}
$$
:::

:::{prf:proof}
The characteristic function of $S_n-n\mu$ is $\exp(-n|\sigma t|^\alpha)$,
which is that of $n^{1/\alpha}(X_1-\mu)$. Equality of characteristic
functions proves equality in distribution. Division by $n$ gives the
average's identity.
:::

For $1<\alpha\le2$ the scale shrinks. The finite absolute moment proved
below also makes the strong law applicable, so $\bar X_n\to\mu$ almost
surely. At $\alpha=1$ the average keeps its Cauchy law; at $\alpha<1$
the scale grows and $\mu$ is only a location. These exact identities must
not be transferred to a Pareto parent merely because its normalized sums
have a stable limit.

(stable-mean-deviation)=
## Mean absolute deviation of a symmetric stable law

We integrate the characteristic function to find the absolute-moment boundary.

:::{prf:proposition} Absolute deviation of a symmetric stable law and its average
:label: prop-generalized-central-limit-theorem-stable-absolute-deviation

Let $X_1,\ldots,X_n$ be iid with the law of $X$, where
$\mathbb E e^{itX}=\exp(i\mu t-|\sigma t|^\alpha)$ for
$\mu\in\mathbb R$, $\sigma>0$, and $0<\alpha\le2$.
Set $\bar X_n=n^{-1}\sum_{i=1}^n X_i$, $n\ge1$.
For $1<\alpha\le2$, $\mathbb EX=\mu$ and

$$
\begin{gathered}
\mathbb E|X-\mu|=\frac{2\sigma}{\pi}\Gamma(1-1/\alpha),\\
\begin{aligned}
&\mathbb E|\bar X_n-\mu|\\
&\quad=\frac{2\sigma}{\pi}\Gamma(1-1/\alpha)n^{1/\alpha-1}.
\end{aligned}
\end{gathered}
$$

For $0<\alpha\le1$, both $\mathbb E|X-\mu|$ and
$\mathbb E|\bar X_n-\mu|$ are infinite; $\mu$ is a location parameter.
:::

:::{prf:proof}
Start with $|x|=(2/\pi)\int_0^\infty(1-\cos(tx))/t^2\,dt$.
For completeness, the integral at $x=1$ equals $\pi/2$: write
$u^{-2}=\int_0^\infty s e^{-su}\,ds$ and use Tonelli to get
$\int_0^\infty (1+s^2)^{-1}\,ds=\pi/2$. Scaling gives the identity
for all $x$. Apply Tonelli again and change the integration coordinate via
$t\leftarrow(\sigma t)^\alpha$. In the remaining integral and its analysis,
$t$ denotes this transformed coordinate:

$$
\begin{aligned}
&\mathbb E|X-\mu|\\
&\quad=\frac{2\sigma}{\pi\alpha}
\int_0^\infty(1-e^{-t})t^{-1-1/\alpha}\,dt.
\end{aligned}
$$

Integration by parts makes the integral
$\alpha\Gamma(1-1/\alpha)$; both boundary terms vanish when $\alpha>1$.
For $\alpha\le1$, the integrand near zero is asymptotic to
$t^{-1/\alpha}$, so the absolute moment is infinite. Symmetry and
integrability give $\mathbb EX=\mu$ in the finite case. The average's
formula, including its infinite cases, now follows from
[](#lem-generalized-central-limit-theorem-stable-scaling).
:::

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

Here we prove the cumulant identity; the CDF error theorem below is cited.

:::{prf:proposition} Cumulant scaling of standardized iid sums
:label: prop-generalized-central-limit-theorem-cumulant-scaling

Let $X_1,\ldots,X_n$ be iid copies of $X$ with mean $\mu$, variance
$0<\sigma^2<\infty$, and finite absolute moment of integer order $r\ge3$.
With characteristic
function $\varphi_X(t)=\mathbb E e^{itX}$, define the local $r$th cumulant by
$\kappa_r(X)=i^{-r}(\log\varphi_X)^{(r)}(0)$, using the continuous logarithm
near $\varphi_X(0)=1$. Set $S_n=\sum_{i=1}^nX_i$, then standardize
$S_n\leftarrow(S_n-n\mu)/(\sigma\sqrt n)$;
throughout this section, $S_n$ denotes the standardized sum. Then, for
each integer $n\ge1$,

$$
\kappa_r(S_n)=n^{1-r/2}\frac{\kappa_r(X)}{\sigma^r}.
$$

In particular, standardized skewness is multiplied by $n^{-1/2}$ when
the third absolute moment is finite, and excess kurtosis by $n^{-1}$ when
the fourth moment is finite.
:::

:::{prf:proof}
Near zero the continuous logarithm of the characteristic function of the
standardized sum is
$n\log\varphi_X(t/(\sigma\sqrt n))-itn\mu/(\sigma\sqrt n)$.
Take $r\ge3$ derivatives at zero and multiply by $i^{-r}$.
The linear centering term vanishes, and the chain rule gives the stated
factor. For a variance-one variable, the third cumulant is skewness and
the fourth is excess kurtosis.
:::

Thus standardized skewness decays as $n^{-1/2}$ when the third absolute
moment is finite, and excess kurtosis as $n^{-1}$ when the fourth moment
is finite. Direct integration gives zero skewness and excess $-6/5$ for
$U(0,1)$; differentiating $-\log(1-t)$ gives skewness $2$ and excess
$6$ for $\mathrm{Exp}(1)$. For iid sums, divide these excesses by $n$.
These calculations describe moments, not a distance between CDFs.
Zero excess kurtosis does not identify a Gaussian law: a variable taking
$0,\sqrt3,-\sqrt3$ with probabilities $2/3,1/6,1/6$ has variance one
and fourth moment three, but is discrete.

For a CDF error bound, the cited Berry–Esseen theorem
(@durrett2019probability, Theorem 3.4.17, p. 157) requires
$\rho=\mathbb E|X-\mu|^3<\infty$ and gives
$\sup_x|P(S_n\le x)-\Phi(x)|\le3\rho/(\sigma^3\sqrt n)$.
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
