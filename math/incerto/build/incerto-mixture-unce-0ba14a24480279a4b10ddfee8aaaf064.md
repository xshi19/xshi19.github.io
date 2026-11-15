---
title: Variance mixtures and uncertain scale
options:
  concept:
    id: mixture-uncertainty
    type: distribution
    prerequisites:
      - pareto-moment-existence
    related:
      - body-shoulder-tail
      - moment-estimation
    tags:
      - mixtures
      - moments
      - tail-risk
---

A Gaussian conditional model need not give a Gaussian marginal law. Let
$Z\sim N(0,1)$ and let $Y>0$ be independent of $Z$. We study

$$
\begin{gathered}
X=\sqrt Y Z,\\
X\mid Y=y\sim N(0,y).
\end{gathered}
$$

Here $Y$ is **variance**, so $\sqrt Y$ is conditional standard deviation.
This is the centered scalar case of the
[Normix normal-mixture construction](https://xshi19.github.io/math/normix-theory/normix-normal-mixtures/).
We use its shared mixing symbol $Y$ and the
[notation canon](https://xshi19.github.io/math/notation/).
A one-observation mixture specifies no temporal dependence: volatility
clustering requires a model for a sequence of mixing variables.

(mixture-moment-transfer)=
## Which moments does mixing preserve?

We first transfer moment questions to the mixing variable.

:::{prf:proposition} Absolute moments of a centered normal variance mixture
:label: prop-mixture-uncertainty-moment-transfer

Let $Y>0$ be finite almost surely, independent of $Z\sim N(0,1)$,
and put $X=\sqrt YZ$. For every real $q>0$, the following absolute-moment
identity holds, allowing infinite expectations:

$$
\begin{gathered}
\mathbb E|X|^q=\mathbb E|Z|^q\,\mathbb E[Y^{q/2}],\\
\mathbb E|Z|^q=\frac{2^{q/2}\Gamma((q+1)/2)}{\sqrt\pi}.
\end{gathered}
$$

If $0<\mathbb EY<\infty$, then $\mathbb E|X|<\infty$, $\mathbb EX=0$ and
$\operatorname{Var}(X)=\mathbb EY$. If also $\mathbb EY^2<\infty$, its
Pearson kurtosis is

$$
\begin{aligned}
\frac{\mathbb EX^4}{(\mathbb EX^2)^2}
&=3\frac{\mathbb EY^2}{(\mathbb EY)^2}\\
&=3\left(1+\frac{\operatorname{Var}(Y)}{(\mathbb EY)^2}\right).
\end{aligned}
$$

The positive absolute moment of $X$ exists exactly when the mixing
moment $\mathbb EY^{q/2}$ exists.
:::

:::{prf:proof}
The factorization follows from independence and Tonelli's theorem applied to the
nonnegative product $Y^{q/2}|Z|^q$. In the normal integral for
$\mathbb E|Z|^q$, substituting $z^2/2$ as the integration variable gives
the displayed gamma integral. The Gaussian factor is positive and finite, so the moment exists
exactly when the mixing moment exists.
For $\mathbb EY<\infty$, Cauchy–Schwarz makes $X$ integrable,
and symmetry gives $\mathbb EX=0$. The cases $q=2,4$ give the variance
and fourth moment. Expanding $\mathbb EY^2=\operatorname{Var}(Y)+(\mathbb EY)^2$
gives the last expression for kurtosis.
:::

Symmetry alone does not make an odd signed moment exist; absolute
integrability is still required. The Gaussian reference is 3. For fixed mean variance, nonconstant $Y$
raises kurtosis. This comparison is specific to centered normal variance
mixtures; it is not an ordering of arbitrary distributions by tail risk.

(mixture-finite-states)=
## Two states, finite moments

:::{prf:proposition} Two-state Gaussian variance mixtures
:label: prop-mixture-uncertainty-two-state

Let $0\le a<1$, let $Y$ take values $1-a$ and $1+a$ with equal
probabilities, and let $Z\sim N(0,1)$ be independent. For $X=\sqrt YZ$,
$\operatorname{Var}(X)=1$ and $\mathbb EX^4=3(1+a^2)$.
Its moment generating function $M_X(t)=\mathbb E e^{tX}$ and
characteristic function $\varphi_X(t)=\mathbb E e^{itX}$ satisfy,
for $t\in\mathbb R$,

$$
M_X(t)=\tfrac12e^{(1-a)t^2/2}+\tfrac12e^{(1+a)t^2/2},\qquad
\varphi_X(t)=\tfrac12e^{-(1-a)t^2/2}+\tfrac12e^{-(1+a)t^2/2}.
$$

All positive power moments and the MGF are finite. Writing $f_a$ for its density
and $\phi$ for the standard-normal density, if $a>0$,

$$
\frac{f_a(x)}{\phi(x)}\sim
\frac1{2\sqrt{1+a}}\exp\left(\frac{ax^2}{2(1+a)}\right)
\quad\text{as }|x|\to\infty.
$$
:::

:::{prf:proof}
The cases $q=2,4$ of [](#prop-mixture-uncertainty-moment-transfer)
give the moments. Conditioning on the two values of $Y$ gives the density
and both transforms. Dividing the narrower Gaussian density by $\phi$
gives a term tending to zero; the wider component gives the displayed
exponential term, which dominates. The transform is finite for every real
$t$, and each component has all positive power moments.
:::

At $a=0.6$, the variance and fourth moment are 1 and 4.08. The MGF
and characteristic function are different functions even though the mean
is zero. This mixture has more far-tail density than the unit normal and
still satisfies the [Cramér condition](incerto-cramer-condition.md).
It is not a power law. At $a=1$ the lower state is an atom at zero,
and the full law no longer has a Lebesgue density.

Unequal probabilities remove the equal-weight example's kurtosis ceiling.

:::{prf:proposition} Unbounded kurtosis across finite mixtures
:label: prop-mixture-uncertainty-rare-state

Fix $0<c<1$. For $0<p<1$, let $Y$ equal $c$ with probability
$1-p$ and $[1-(1-p)c]/p$ with probability $p$. With independent
$Z\sim N(0,1)$, put $X=\sqrt YZ$. Then $\operatorname{Var}(X)=1$,
its kurtosis tends to infinity as $p\downarrow0$, and every fixed $p$
gives finite positive power moments and an MGF finite on all of $\mathbb R$.
:::

:::{prf:proof}
Direct summation gives $\mathbb EY=1$ and
$\mathbb EY^2=(1-p)c^2+[1-(1-p)c]^2/p\to\infty$.
By [](#prop-mixture-uncertainty-moment-transfer), kurtosis equals
$3\mathbb EY^2$. Each fixed $p$ gives only two finite positive
conditional variances, so all moments and the MGF are finite.
:::

For a different worked case, take

$$
\begin{gathered}
P(Y=0.5)=0.99,\\
P(Y=50.5)=0.01.
\end{gathered}
$$

Then $\mathbb EY=1$, $\mathbb EY^2=25.75$ and kurtosis is $77.25$.
All moments and the MGF remain finite.

A rare state can make a finite moment hard to estimate without making it
infinite.

```{figure} ../assets/figures/incerto/finite-moment-mixtures.svg
:alt: Two vertically stacked plots compare unit-variance Gaussian mixtures: central densities above and two-sided log survival below.

Exact mixture densities and two-sided probabilities. All three laws have
variance 1. The rare wide component is visible in the survival plot even
where its contribution to central density is small. No samples or data fits
are used.
```

(mixture-variance-precision)=
## Variance and precision give different laws

Three explicit constructions show why the distribution of the mixing
variable matters. All parameters below are local to this section.

:::{prf:proposition} Gamma mixing of variance
:label: prop-mixture-uncertainty-gamma-variance

Let $Y$ have gamma shape $k>0$ and scale $1/k$, and let
$Z\sim N(0,1)$ be independent. For $X=\sqrt YZ$,

$$
\begin{gathered}
\mathbb EY^r=k^{-r}\frac{\Gamma(k+r)}{\Gamma(k)},\\
r>-k,\\
M_X(t)=\left(1-\frac{t^2}{2k}\right)^{-k},\\
|t|<\sqrt{2k}.
\end{gathered}
$$

The MGF is infinite for $|t|\ge\sqrt{2k}$. All positive power moments
of $X$ exist, its variance is 1 and its kurtosis is $3(1+1/k)$.
At $k=1$, $X$ has a centered Laplace law with scale $1/\sqrt2$.
:::

:::{prf:proof}
Integrating $y^r$ against $k^ky^{k-1}e^{-ky}/\Gamma(k)$ gives
the gamma moment formula for $r>-k$. Conditioning on $Y$ gives
$M_X(t)=\mathbb E e^{t^2Y/2}$; the gamma integral converges exactly
when $k-t^2/2>0$ and evaluates to the displayed MGF.
[](#prop-mixture-uncertainty-moment-transfer) gives the moment conclusions.
At $k=1$, the MGF $(1-t^2/2)^{-1}$ identifies the stated Laplace law.
:::

The Laplace law has exponential tails. The density calculation below
gives the general gamma-mixture decay.

:::{prf:proposition} Lognormal mixing of variance
:label: prop-mixture-uncertainty-lognormal-variance

Let $U,Z$ be independent standard normals, $\tau>0$,
$Y=e^{\tau U-\tau^2/2}$ and $X=\sqrt YZ$. For $r\in\mathbb R$
and $q>0$,

$$
\mathbb EY^r=e^{(r^2-r)\tau^2/2},\qquad
\mathbb E|X|^q=\mathbb E|Z|^q e^{q(q-2)\tau^2/8}.
$$

Every positive power moment of $X$ is finite, but $M_X(t)=\infty$
for every $t\ne0$. Its right survival cannot be regularly varying with
a finite positive power exponent.
:::

:::{prf:proof}
Completing the square in the normal integral gives $\mathbb EY^r$.
Apply [](#prop-mixture-uncertainty-moment-transfer) with $r=q/2$.
For $t\ne0$, $M_X(t)=\mathbb E e^{t^2Y/2}$. In the integral over
$u$, the exponent is a positive multiple of $e^{\tau u}$ minus
$u^2/2$, eventually positive on an infinite half-line, so the integral
diverges. A regularly varying right survival with finite positive exponent
would force sufficiently high positive moments to diverge, by the
[regular-variation moment test](incerto-karamata.md).
:::

Finite moments of every order therefore do not imply a finite MGF.

:::{prf:proposition} Gamma mixing of precision gives Student tails
:label: prop-mixture-uncertainty-gamma-precision

Let $1/Y$ have gamma shape and rate both $\nu/2$, with $\nu>0$,
and let $Z\sim N(0,1)$ be independent of $Y$. Then $X=\sqrt YZ$
has density

$$
\begin{aligned}
f_X(x)&=\frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\Gamma(\nu/2)}\\
&\quad\times\left(1+\frac{x^2}{\nu}\right)^{-(\nu+1)/2}.
\end{aligned}
$$

This is Student's $t$ law at unit scale. Its right survival has exponent
$\alpha=\nu$, and for $q>0$, $\mathbb E|X|^q<\infty$ exactly when
$q<\nu$. For $\nu>2$, $\operatorname{Var}(X)=\nu/(\nu-2)$.
:::

:::{prf:proof}
Integrating the conditional normal density over precision $g=1/y$
gives an integrand proportional to
$g^{(\nu+1)/2-1}e^{-(\nu+x^2)g/2}$, a gamma integral, and hence the
displayed density. It is asymptotic to a positive constant times
$|x|^{-\nu-1}$; integrating gives the survival exponent.
By [](#prop-mixture-uncertainty-moment-transfer), the $q$th moment
reduces to an integral of $g^{(\nu-q)/2-1}$ near zero, finite exactly
when $q<\nu$. At $q=2<\nu$, the gamma recurrence gives
$\mathbb EY=\nu/(\nu-2)$, which is the variance of $X$.
:::

Unit scale is not unit variance.

These are proved model calculations. SCoFT v4 §4.2, pp. 74–75 supplies the
variance-versus-precision question; it is not a claim that every stochastic
volatility model produces a power law. Recursive uncertainty belongs to a
later treatment and requires a specified limiting construction.

## Reproduce the worked cases

Use [`normal_mixture_absolute_moment`](incerto-api-reference.md#api-normal-mixture-absolute-moment), [`normal_mixture_log_abs_survival`](incerto-api-reference.md#api-normal-mixture-log-abs-survival).

```python
import numpy as np
from xmath.incerto.finite_moments import (
    normal_mixture_absolute_moment, normal_mixture_log_abs_survival,
)

for variances, weights in (([1], [1]), ([0.5, 50.5], [0.99, 0.01])):
    m2 = normal_mixture_absolute_moment(2, variances, weights)
    m4 = normal_mixture_absolute_moment(4, variances, weights)
    tail = np.exp(normal_mixture_log_abs_survival(4, variances, weights))
    print(f"Variance / kurtosis / P(|X|>4): {m2:.4g} {m4 / m2**2:.4g} {tail:.4g}")
```

**Output**

```text
Variance / kurtosis / P(|X|>4): 1 3 6.334e-05
Variance / kurtosis / P(|X|>4): 1 77.25 0.005735
```

All moments of these two finite Gaussian mixtures exist.

Figure regeneration: `uv run python demos/incerto/finite_moments.py`. The mixture curves use
801 central-density and 241 tail-threshold points, with the parameters above.
The script also checks moment formulas and reports a bounded, seeded
estimation experiment described in [moment estimation](incerto-moment-estimation.md).
[Body and shoulder geometry](incerto-body-shoulder-tail.md) explains which
parts of a density gain or lose under a small variance perturbation.

(mixture-variance-gamma)=
## Variance–mean gamma calculation

The following extension permits a moving conditional mean.

:::{prf:proposition} Variance–mean gamma transforms and density
:label: prop-mixture-uncertainty-variance-gamma

Let $A>|\gamma|$, $\lambda>0$, and $\vartheta=2/(A^2-\gamma^2)$.
Let $Y\sim\operatorname{Gamma}(\lambda,\vartheta)$ in shape–scale
notation, independent of $Z\sim N(0,1)$, and set
$X=\gamma Y+\sqrt YZ$. Its MGF, mean and variance are

$$
\begin{gathered}
M_X(t)\\
=\left(1-\vartheta\gamma t-\tfrac12\vartheta t^2\right)^{-\lambda},\\
-A-\gamma<t<A-\gamma,
\end{gathered}
$$

$$
\begin{gathered}
\mathbb E[X]=\gamma\lambda\vartheta,\\
\operatorname{Var}(X)=\lambda\vartheta(1+\gamma^2\vartheta).
\end{gathered}
$$

The MGF is infinite outside the displayed open interval. For $x\ne0$,
its density is

$$
\begin{aligned}
f(x)&=\frac{(A^2-\gamma^2)^\lambda |x|^{\lambda-1/2}}
{\sqrt\pi\,\Gamma(\lambda)(2A)^{\lambda-1/2}}\\
&\quad\times K_{\lambda-1/2}(A|x|)e^{\gamma x}.
\end{aligned}
$$

Here $K_\nu$ is the modified Bessel function of the second kind.
As $x\to+\infty$ and $x\to-\infty$, respectively, the density is
asymptotic to positive constants times
$|x|^{\lambda-1}e^{-(A-\gamma)x}$ and
$|x|^{\lambda-1}e^{-(A+\gamma)|x|}$.
For $\lambda>1/2$, its continuous value at zero is

$$
f(0)=\frac{(A^2-\gamma^2)^\lambda\Gamma(\lambda-1/2)}
{2\sqrt\pi\,\Gamma(\lambda)A^{2\lambda-1}}.
$$

At $\lambda=1/2$ the density has a logarithmic singularity at zero;
for $0<\lambda<1/2$ it has an integrable power singularity.
For $\gamma=0$, $\lambda=1$, it reduces to $(A/2)e^{-A|x|}$.
:::

:::{prf:proof}
Conditioning and the gamma MGF give the stated transform.
The gamma integral converges exactly when
$1-\vartheta\gamma t-\vartheta t^2/2>0$, yielding the open interval.
Conditional expectation and variance give the two moments.
For $x\ne0$, integrate the normal density against the gamma density
and use the Bessel integral with $\nu=\lambda-1/2$, $a=x^2/2$,
$b=A^2/2$:

$$
\int_0^\infty y^{\nu-1}e^{-a/y-by}\,dy
=2(a/b)^{\nu/2}K_\nu(2\sqrt{ab}),\qquad a,b>0.
$$

The cited asymptotic $K_\nu(z)\sim\sqrt{\pi/(2z)}e^{-z}$ gives
the two density tails. These identities are used without proof:
[DLMF 10.32.10](https://dlmf.nist.gov/10.32#E10) and
[DLMF 10.40.2](https://dlmf.nist.gov/10.40#E2).
At zero, the density integral is a gamma integral finite exactly when
$\lambda>1/2$ and evaluates to the stated value. The singularities for
smaller shapes follow from the cited small-argument limits
[DLMF 10.30.2–3](https://dlmf.nist.gov/10.30). For $\gamma=0$ and
$\lambda=1$, the MGF is $(1-t^2/A^2)^{-1}$, which identifies the
Laplace density $(A/2)e^{-A|x|}$ as in the gamma-variance calculation.
:::

These are exponential tails, not regularly varying ones.
The helper [`univariate_variance_gamma(alpha, beta, lam)`](incerto-api-reference.md#api-univariate-variance-gamma)
uses `alpha` for $A$, `beta` for $\gamma$, and `lam` for $\lambda$;
its `alpha` is not the Pareto exponent. General family relationships
remain in the [Normix family tour](https://xshi19.github.io/math/normix-theory/normix-gh-family-tour/).

(mixture-growth-rate)=
## Random rates before exponentiation

Appendix C of @taleb2025scoftv4, §C.1, pp. 183–185, concerns transforming
a growth rate. The [fixed-horizon calculation](incerto-ergodicity.md#ergodicity-growth-transform)
shows that $X=x_m e^{TR}$ with $R\sim\mathrm{Exp}(\lambda)$ has Pareto
exponent $\lambda/T$. A distribution over rates must be specified before
claims about the outcome tail follow.

:::{prf:proposition} A finite mixture of exponential growth rates
:label: prop-mixture-uncertainty-random-rates

Let $J$ take finitely many values with $P(J=j)=w_j>0$ and
$\sum_jw_j=1$. Conditional on $J=j$, let $R\sim\mathrm{Exp}(\lambda_j)$,
where $\lambda_j>0$. For fixed $T,x_m>0$, put $X=x_m e^{TR}$.
Then for $x\ge x_m$,

$$
P(X>x)=\sum_jw_j(x_m/x)^{\lambda_j/T}.
$$

Its tail exponent is $\min_j\lambda_j/T$: relative to
$(x_m/x)^{\min_j\lambda_j/T}$, its survival tends to the combined
weight of the minimizing states. For $p>0$,

$$
\mathbb EX^p=x_m^p\sum_jw_j\frac{\lambda_j}{\lambda_j-pT}
\quad\text{if }pT<\min_j\lambda_j,
$$

and otherwise the moment is infinite.
:::

:::{prf:proof}
Conditioning on $J$ and applying the
[fixed-horizon transformation](incerto-ergodicity.md#ergodicity-growth-transform)
gives the survival. Factor its smallest power; all other terms vanish
relative to it as $x\to\infty$. Tonelli and the conditional Pareto
moment formula give the finite sum when every conditional moment exists.
If one fails, its positive weight makes the mixture moment infinite.
:::

For rates $1$ and $3$ with equal weights
and $T=1$, the outcome mean is infinite, although the rate has finite mean
$2/3$. This is a finite mixture, not an unspecified infinite hierarchy of
uncertainty. A continuous rate mixture near zero needs a separate analysis.

Use [`pareto_type1`](incerto-api-reference.md#api-pareto-type1) to evaluate
the conditional outcome laws.

```python
from xmath.incerto.distributions import pareto_type1

for horizon in (.5, 1):
    means = [pareto_type1(rate/horizon).mean() for rate in (1, 3)]
    print(f"T={horizon:g}: mixture mean={sum(means)/2:.4g}")
```

**Output**

```text
T=0.5: mixture mean=1.6
T=1: mixture mean=inf
```

## References

- @taleb2025scoftv4, *Statistical Consequences of Fat Tails*, third edition,
  2025, ch. 4, §§4.1–4.2, printed pp. 69–75: mixture mechanisms. The
  calculations above state their own hypotheses and proofs.
- The linked DLMF identities supply the Bessel integral and asymptotics used
  only in the variance–mean gamma density calculation.
