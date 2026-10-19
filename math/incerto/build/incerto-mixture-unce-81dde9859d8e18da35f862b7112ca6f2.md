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
X=\sqrt Y Z,\qquad X\mid Y=y\sim N(0,y).
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

For every real $q>0$, the absolute moment factors, including infinite values:

$$
\mathbb E|X|^q=c_q\mathbb E[Y^{q/2}],\qquad
c_q=\frac{2^{q/2}\Gamma((q+1)/2)}{\sqrt\pi}.
$$

We prove this identity by independence and Tonelli's theorem applied to the
nonnegative product $Y^{q/2}|Z|^q$. In the normal integral for $c_q$, the
substitution $t=z^2/2$ gives the displayed gamma integral. Thus the moment
exists exactly when the mixing moment exists. Symmetry alone does not make
an odd signed moment exist; absolute integrability is still required.

If $0<\mathbb EY<\infty$, then $\mathbb E|X|<\infty$, $\mathbb EX=0$ and
$\operatorname{Var}(X)=\mathbb EY$. If also $\mathbb EY^2<\infty$, its
Pearson kurtosis is

$$
\frac{\mathbb EX^4}{(\mathbb EX^2)^2}
=3\frac{\mathbb EY^2}{(\mathbb EY)^2}
=3\left(1+\frac{\operatorname{Var}(Y)}{(\mathbb EY)^2}\right).
$$

The Gaussian reference is 3. For fixed mean variance, nonconstant $Y$
raises kurtosis. This comparison is specific to centered normal variance
mixtures; it is not an ordering of arbitrary distributions by tail risk.

(mixture-finite-states)=
## Two states, finite moments

Choose equal probabilities for $Y=1-a$ and $Y=1+a$, with $0\le a<1$.
Conditioning gives variance 1 and fourth moment $3(1+a^2)$. At $a=0.6$
these are 1 and 4.08. The density is the average of two Gaussian densities.
Its moment generating function (MGF) and characteristic function are

$$
M_X(t)=\tfrac12e^{(1-a)t^2/2}+\tfrac12e^{(1+a)t^2/2},\qquad
\varphi_X(t)=\tfrac12e^{-(1-a)t^2/2}+\tfrac12e^{-(1+a)t^2/2}.
$$

They follow from the two conditional Gaussian transforms. They are different
functions, even though the mean is zero. For $a>0$, division by the unit-normal
density $\phi(x)$ gives

$$
\frac{f_a(x)}{\phi(x)}
\sim \frac{1}{2\sqrt{1+a}}
\exp\left(\frac{ax^2}{2(1+a)}\right),\qquad |x|\to\infty.
$$

The narrower component's ratio tends to zero; the wider component's ratio
increases without bound. Nevertheless $M_X(t)$ is finite for every real $t$.
The mixture has more far-tail density than the unit normal and still satisfies
the [Cramér condition](incerto-cramer-condition.md). It is not a power law.
At $a=1$ the lower state is an atom at zero and the full law no longer has
a Lebesgue density.

Unequal probabilities remove the equal-weight example's kurtosis ceiling.
For a different worked case, take

$$
P(Y=0.5)=0.99,\qquad P(Y=50.5)=0.01.
$$

Then $\mathbb EY=1$, $\mathbb EY^2=25.75$ and kurtosis is $77.25$.
All moments and the MGF remain finite. More generally, fixing $0<c<1$ and
putting probability $0<p<1$ on $b=[1-(1-p)c]/p$ and $1-p$ on $c$ preserves
$\mathbb EY=1$, while $\mathbb EY^2$ diverges as $p\downarrow0$.
Each fixed $p>0$ still gives a finite Gaussian mixture. A rare state can
make a finite moment hard to estimate without making it infinite.

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

**Gamma variance.** Let $Y$ have gamma shape $k>0$ and scale $1/k$, so
$\mathbb EY=1$. Integrating its density gives

$$
\mathbb EY^r=k^{-r}\frac{\Gamma(k+r)}{\Gamma(k)},\quad r>-k,
\qquad
M_X(t)=\left(1-\frac{t^2}{2k}\right)^{-k},\quad |t|<\sqrt{2k}.
$$

For the MGF, condition on $Y$ and integrate $e^{t^2Y/2}$ against the gamma
density. The integral diverges at and beyond the endpoints. All positive
power moments exist, and kurtosis is $3(1+1/k)$. At $k=1$, the MGF is that
of a centered Laplace law with scale $1/\sqrt2$; this law has exponential
tails. The density calculation below gives the general gamma-mixture decay.

**Lognormal variance.** Let $Y=\exp(\tau U-\tau^2/2)$, where $U$ is standard
normal, independent of $Z$, and $\tau>0$. Completing the square gives

$$
\mathbb EY^r=e^{(r^2-r)\tau^2/2},\qquad
\mathbb E|X|^q=c_q e^{q(q-2)\tau^2/8}.
$$

Every positive power moment exists, yet $M_X(t)=\mathbb E e^{t^2Y/2}=\infty$
for every $t\ne0$. To see divergence, write the expectation as an integral
over $u$; its exponent contains a positive multiple of $e^{\tau u}$ minus
$u^2/2$, and is eventually positive on an infinite half-line. Finite moments
of every order therefore do not imply a finite MGF. Nor does this construction
have a finite power-tail exponent: a regularly varying survival with finite
positive exponent would force sufficiently high moments to diverge.

**Gamma precision.** Let $Q=1/Y$ have gamma shape $\nu/2$ and rate $\nu/2$,
with $\nu>0$. Integrating the conditional density gives

$$
f_X(x)=\frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\Gamma(\nu/2)}
\left(1+\frac{x^2}{\nu}\right)^{-(\nu+1)/2}.
$$

Indeed the integrand is proportional to
$q^{(\nu+1)/2-1}e^{-(\nu+x^2)q/2}$, a gamma integral. This is Student's
$t$ law at unit **scale**. Its density is asymptotic to a positive constant
times $|x|^{-\nu-1}$; integrating that comparison gives a survival exponent
$\alpha=\nu$. The moment identity above reduces to an integral of
$q^{(\nu-q_0)/2-1}$ near zero for moment order $q_0>0$, so
$\mathbb E|X|^{q_0}<\infty$ exactly when $q_0<\nu$.
For $\nu>2$ its variance is $\nu/(\nu-2)$, not 1.

These are proved model calculations. SCoFT v4 §4.2, pp. 74–75 supplies the
variance-versus-precision question; it is not a claim that every stochastic
volatility model produces a power law. Recursive uncertainty belongs to a
later treatment and requires a specified limiting construction.

## Reproduce the worked cases

Run `uv run python demos/incerto/finite_moments.py`. The mixture curves use
801 central-density and 241 tail-threshold points, with the parameters above.
The script also checks moment formulas and reports a bounded, seeded
estimation experiment described in [moment estimation](incerto-moment-estimation.md).
[Body and shoulder geometry](incerto-body-shoulder-tail.md) explains which
parts of a density gain or lose under a small variance perturbation.

(mixture-variance-gamma)=
## Variance–mean gamma calculation

The following extension permits a moving conditional mean. It retains the
scalar calculation previously given in the chapter guide.

For a gamma mixture, let $Y\sim\operatorname{Gamma}(\lambda,\vartheta)$
(shape and scale) and $X=\gamma Y+\sqrt{Y}Z$. The local parameters satisfy
$A>|\gamma|$, $\lambda>0$, and $\vartheta=2/(A^2-\gamma^2)$.
The legacy helper `univariate_variance_gamma(alpha, beta, lam)` uses
`alpha` for $A$, `beta` for $\gamma$, and `lam` for $\lambda$; its `alpha`
is not the Pareto exponent. Conditioning and the gamma MGF give

$$
M_X(t)=\left(1-\vartheta\gamma t-\tfrac12\vartheta t^2\right)^{-\lambda},
\qquad -A-\gamma<t<A-\gamma,
$$

$$
\mathbb E[X]=\gamma\lambda\vartheta,\qquad
\operatorname{Var}(X)=\lambda\vartheta(1+\gamma^2\vartheta).
$$

The interval follows by solving the positive-quadratic-base inequality.
The variance follows from conditional variance plus the variance of the
conditional mean. For $x\ne0$, integrating the conditional Gaussian density
against the gamma density gives

$$
f(x)=\frac{(A^2-\gamma^2)^\lambda |x|^{\lambda-1/2}
K_{\lambda-1/2}(A|x|)}{\sqrt\pi\,\Gamma(\lambda)(2A)^{\lambda-1/2}}
e^{\gamma x}.
$$

To obtain this expression, substitute $\nu=\lambda-1/2$,
$a=x^2/2$, $b=A^2/2$ in the Bessel integral
$\int_0^\infty y^{\nu-1}e^{-a/y-by}\,dy
=2(a/b)^{\nu/2}K_\nu(2\sqrt{ab})$ for $a,b>0$.
The standard Bessel asymptotic $K_\nu(z)\sim\sqrt{\pi/(2z)}e^{-z}$
then gives right and left density decay rates $A-\gamma$ and $A+\gamma$,
each with a power factor $|x|^{\lambda-1}$. These are exponential tails,
not regularly varying ones. The integral representation and special-function
asymptotics are used as standard identities, not proved here
[DLMF 10.32.10](https://dlmf.nist.gov/10.32#E10) and
[DLMF 10.40.2](https://dlmf.nist.gov/10.40#E2).

At the origin, the same density integral is a gamma integral when
$\lambda>1/2$, giving

$$
f(0)=\frac{(A^2-\gamma^2)^\lambda\Gamma(\lambda-1/2)}
{2\sqrt\pi\,\Gamma(\lambda)A^{2\lambda-1}}.
$$

For smaller shapes it diverges: the Bessel limits give a logarithmic
singularity at $\lambda=1/2$ and an integrable power singularity below it.
Here the small-argument Bessel limits are cited from
[DLMF 10.30.2–3](https://dlmf.nist.gov/10.30). The symmetric case $\gamma=0$, $\lambda=1$ reduces to the Laplace density
$(A/2)e^{-A|x|}$. The general family relationships remain in the
[Normix family tour](https://xshi19.github.io/math/normix-theory/normix-gh-family-tour/).

## References

- @taleb2025scoftv4, *Statistical Consequences of Fat Tails*, third edition,
  2025, ch. 4, §§4.1–4.2, printed pp. 69–75: mixture mechanisms. The
  calculations above state their own hypotheses and proofs.
- The linked DLMF identities supply the Bessel integral and asymptotics used
  only in the variance–mean gamma density calculation.

## Source and license

Original mixture and moment explanations with the variance–mean gamma
calculation retained from this repository's chapter-4 guide. That calculation
was adapted from `xshi19/incerto-wiki`,
`content/concepts/distributions/normal-mixture.md` and
`content/concepts/distributions/variance-gamma.md`, revision
`9717c9cf83aee18b47f423a14bce2f0d14bb8bcb`, under MIT. The source books
retain their own rights. No book prose or figures are reproduced.

:::{dropdown} MIT permission notice

MIT License

Copyright (c) 2023 xshi19

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
:::
