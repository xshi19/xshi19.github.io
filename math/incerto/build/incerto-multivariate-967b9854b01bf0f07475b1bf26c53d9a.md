---
title: Multivariate tails and dependence
options:
  concept:
    id: multivariate-tails
    type: distribution
    depends_on:
      - iso-density-tail-geometry
      - mixture-uncertainty
    related:
      - moment-estimation
      - pareto
    tags:
      - fat-tails
      - dependence
---

Two coordinates can have the same marginal distributions and quite different
joint extremes. [Density geometry](incerto-iso-density-tail-geometry.md)
compares their contours; this note constructs a joint law and asks which
dependence summaries exist. The context is SCoFT v4 §§6.1–6.7,
pp. 108–119 [@taleb2025scoftv4]. The calculations below are derived here;
the semicircle limit is a separately cited theorem.

We use dimension $d$, location vector $\mu$, positive definite scatter matrix
$\Sigma$, and Student degrees of freedom $\nu>0$. Scatter specifies shape
and scale; it is a covariance matrix only when we explicitly say so.
Here $I_d$ is the identity matrix; all densities use Lebesgue measure.
Symbols follow the [shared notation](https://xshi19.github.io/math/notation/).

(multivariate-elliptical)=
## A shared random scale

An elliptical law has characteristic function

$$
\varphi_X(t)=e^{it^T\mu}\Psi(t^T\Sigma t),\qquad t\in\mathbb R^d,
$$

where $\Psi$ is a scalar function for which this is a characteristic function.
The definition can allow positive semidefinite $\Sigma$ and singular laws;
our density calculations use positive definite $\Sigma$.
It does not require moments or independence of coordinates.

Let $Z\sim N_d(0,I_d)$, let $Y>0$ be independent of $Z$, and choose
$AA^T=\Sigma$. The [normal variance-mixture construction](incerto-mixture-uncertainty.md)
extends to

$$
X=\mu+\sqrt Y\,AZ,
\qquad
\varphi_X(t)=e^{it^T\mu}\mathbb E e^{-Yt^T\Sigma t/2}.
$$

Conditioning on $Y$ proves ellipticality. It also proves
$\operatorname{Cov}(X)=\mathbb EY\,\Sigma$ when $\mathbb EY<\infty$.
For a deterministic matrix $B$ and vector $b$, substitution of $B^Tt$
in the characteristic function gives location $B\mu+b$ and scatter
$B\Sigma B^T$ for $BX+b$. This proves closure under linear maps, including
singular images. It does not assert that every stable vector is elliptical.

For example, let $Y$ equal $1-a$ or $1+a$ with equal probabilities,
$0<a<1$. Covariance remains $\Sigma$, whereas the density at $\mu$ is
multiplied, relative to $N_d(\mu,\Sigma)$, by

$$
\frac{(1-a)^{-d/2}+(1+a)^{-d/2}}2>1.
$$

Strict convexity of $y\mapsto y^{-d/2}$ proves the inequality.
This finite mixture has finite exponential moments in every direction:
its MGF is an average of two Gaussian MGFs. Increased central density
and kurtosis do not imply an asymptotic power tail.

(multivariate-student)=
## The normalized Student law

Take a precision $G=1/Y$ with gamma shape and rate both $\nu/2$.
Conditioning on $G=g$ gives the normal density with covariance $\Sigma/g$.
Writing $q=(x-\mu)^T\Sigma^{-1}(x-\mu)$, integration over $g>0$ gives

$$
\begin{aligned}
f_d(x)
&=\frac{(\nu/2)^{\nu/2}}{(2\pi)^{d/2}|\Sigma|^{1/2}\Gamma(\nu/2)}
  \int_0^\infty g^{(\nu+d)/2-1}e^{-(\nu+q)g/2}\,dg\\
&=\frac{\Gamma((\nu+d)/2)}
 {\Gamma(\nu/2)(\nu\pi)^{d/2}|\Sigma|^{1/2}}
 \left(1+\frac q\nu\right)^{-(\nu+d)/2}.
\end{aligned}
$$

The gamma integral proves both the constant and normalization, since the
integrated conditional densities each have mass one. Every nonzero linear
projection is a univariate Student with the same $\nu$, location $b^T\mu$,
and scale $\sqrt{b^T\Sigma b}$. This follows directly from $b^TAZ$ being
normal, independent of the same $G$.

The [univariate moment gate](incerto-mixture-uncertainty.md#mixture-variance-precision)
therefore applies to each coordinate: a mean exists for $\nu>1$, and
finite covariance requires $\nu>2$. Gamma integration gives

$$
\mathbb E G^{-1}=\frac{\nu}{\nu-2},\qquad
\operatorname{Cov}(X)=\frac{\nu}{\nu-2}\Sigma\quad(\nu>2).
$$

For $\nu\le2$, a finite sample scatter matrix can still be calculated;
it does not estimate a finite population covariance matrix under this model.

(multivariate-zero-correlation)=
## Diagonal scatter does not give independence

Set $d=2$, $\mu=0$, and $\Sigma=I_2$. The two marginal Student laws
share their random precision. Bayes' rule gives

$$
G\mid X_1=x\sim\operatorname{Gamma}\left(\frac{\nu+1}{2},
                                      \frac{\nu+x^2}{2}\right).
$$

Mixing the conditional normal for $X_2$ thus gives a Student law with
$\nu+1$ degrees of freedom and scale $\sqrt{(\nu+x^2)/(\nu+1)}$.
This scale varies with $x$ on sets of positive probability, which proves
dependence for every finite $\nu>0$. For $\nu>2$ the Pearson correlation
is nevertheless zero. For $\nu\le2$, zero off-diagonal scatter is **not**
a zero Pearson correlation: that population coefficient is undefined.

There is also a direct joint-tail test without moment assumptions. For
$u>0$, put $p_u(y)=2\bar\Phi(u/\sqrt y)$, with $\bar\Phi$ the standard
normal survival. Conditional independence gives

$$
P(|X_1|>u,|X_2|>u)-P(|X_1|>u)^2
=\operatorname{Var}(p_u(Y))>0.
$$

The variance is positive because $p_u$ is strictly increasing and the
mixing law is nondegenerate. Independent Student coordinates have zero
excess instead. These bounded indicator probabilities exist even for Cauchy
marginals. They do not by themselves specify an asymptotic tail-dependence
coefficient.

(multivariate-information)=
## Mutual information without second moments

For a bivariate density $f$ with marginals $f_1,f_2$, mutual information is

$$
I(X_1;X_2)=\int_{\mathbb R^2}f(x_1,x_2)
 \log\frac{f(x_1,x_2)}{f_1(x_1)f_2(x_2)}\,dx_1dx_2.
$$

It is KL divergence from the joint law to the product of its marginals.
The [KL inequality](incerto-maximum-entropy.md#maximum-entropy-shannon)
gives nonnegativity and equality exactly at independence. No variance
appears in the definition, but integrability must still be checked.

We derive the Student value. Write $h_d=-\mathbb E\log f_d(X)$ and let
$\psi_0(z)=d\log\Gamma(z)/dz$ denote the digamma function, distinct from
the shared canon's log-partition symbol. In whitened polar coordinates,
$Q=(X-\mu)^T\Sigma^{-1}(X-\mu)$ satisfies

$$
U=\frac{Q}{\nu+Q}\sim\operatorname{Beta}(d/2,\nu/2).
$$

Indeed the radial density is proportional to
$q^{d/2-1}(1+q/\nu)^{-(\nu+d)/2}$; the displayed substitution gives
the beta density. Differentiating its normalizing integral in the second
parameter gives
$\mathbb E\log(1+Q/\nu)=\psi_0((\nu+d)/2)-\psi_0(\nu/2)$.
The differentiation is justified on compact subsets of positive parameters
by integrable endpoint powers times $|\log(1-u)|$.
The beta integral is the cited input
([DLMF 5.12.1](https://dlmf.nist.gov/5.12#E1)). Substitution into the
normalized log density now proves

$$
\begin{aligned}
h_d(\nu,\Sigma)
={}&\log\frac{\Gamma(\nu/2)}{\Gamma((\nu+d)/2)}
+\frac d2\log(\nu\pi)+\frac12\log|\Sigma|\\
&+\frac{\nu+d}{2}
  \left[\psi_0((\nu+d)/2)-\psi_0(\nu/2)\right].
\end{aligned}
$$

All these logarithmic expectations are finite for $\nu>0$. Thus subtracting
entropies is legitimate even when no second moment exists. For unit diagonal
scatter and off-diagonal entry $\rho$, $|\rho|<1$,

$$
I_\nu(\rho)=-\frac12\log(1-\rho^2)+\lambda_\nu,
\qquad
\lambda_\nu=2h_1(\nu,1)-h_2(\nu,I_2)>0.
$$

The determinant gives the first term; the earlier dependence proof gives
strict positivity of the second. Independent rescaling of either coordinate
cancels from the entropy difference. At $\nu=1$,
$h_1=\log(4\pi)$ and $h_2=\log(2\pi)+3$, so
$\lambda_1=\log(8\pi)-3\approx0.224171$ nats. The Gaussian analogue
has only $-\tfrac12\log(1-\rho^2)$. A single observed sample correlation
near zero cannot establish independence in either model.

```{figure} ../assets/figures/incerto/multivariate-information.svg
:alt: Two vertical panels show positive Student mutual information at zero scatter correlation, then joint absolute-exceedance probabilities for a shared scale versus independent marginals.

Population calculations, with no fitted data. The upper panel evaluates the
entropy formula at $\rho=0$ and $0.6$. The lower panel integrates over gamma
precision at $\nu=3$ and $\rho=0$; the shared scale increases joint absolute
exceedances above the independent-marginal probability. All displayed quantities are
dimensionless; mutual information uses natural logarithms.
```

(multivariate-sample-correlation)=
## A bounded statistic need not have a population target

For two nonzero finite data vectors $x,y\in\mathbb R^n$, the uncentered
cosine is $r_0=x^Ty/(\|x\|\|y\|)$. Cauchy–Schwarz proves $|r_0|\le1$.
Pearson's sample correlation applies this formula to $x-\bar x\mathbf1$
and $y-\bar y\mathbf1$ and requires both vectors to be nonconstant.
The two statistics are different. A constant column makes the centered
coefficient undefined, rather than zero.

Hold $n-1$ observations fixed and replace the last pair by $(t,ct)$,
where $c\ne0$ and $n\ge2$. For the uncentered cosine the leading
numerator is $ct^2$, and the norm product is $|c|t^2$.
For the centered statistic, write $H=I_n-\mathbf1\mathbf1^T/n$.
Here $e_n$ is the last coordinate unit vector and $\mathbf1$ is the
all-ones vector. The leading vectors are $tHe_n$ and $ctHe_n$ with
$\|He_n\|^2=1-1/n>0$. Both coefficients therefore tend to
$\operatorname{sign}(c)$ as $t\to\infty$. This is a deterministic
contamination limit at fixed sample size, not an iid sampling theorem.

Our repeated Student samples below compare $\nu=1.5$ and $\nu=5$ with
scatter parameter $\rho=0.6$. Only the latter has population Pearson
correlation $0.6$. Sampling quantiles quantify this finite experiment;
they are neither confidence intervals nor a universal convergence rate.

(multivariate-spectra)=
## What the matrix ensemble must specify

Let $W_n$ be the upper-left $n\times n$ block of one infinite real
symmetric matrix, with independent identically distributed
entries on and above the diagonal, mean zero and variance one, mirrored
below it. The semicircle theorem says that the empirical eigenvalue
measure of $W_n/\sqrt n$ converges weakly almost surely to density
$\sqrt{4-x^2}/(2\pi)$ on $[-2,2]$. This is a cited result, not proved
here: [Tao, 254A Notes 4, Theorem 1 and §1 truncation reduction](https://terrytao.wordpress.com/2010/02/02/254a-notes-4-the-semi-circular-law/).
Finite variance suffices for this empirical-measure limit; an infinite
fourth moment does not automatically invalidate it. Convergence of the
largest eigenvalue is a different question.

Two finite identities explain the sensitivity to a large entry. For any
real symmetric matrix $M$ with eigenvalues $\ell_1,\ldots,\ell_n$,

$$
\sum_j\ell_j^2=\operatorname{tr}(M^2)=\sum_{i,j}M_{ij}^2,
\qquad \max_j|\ell_j|=\|M\|_2\ge\max_{i,j}|M_{ij}|.
$$

Orthogonal diagonalization proves the first identity, and
$|e_i^TMe_j|\le\|M\|_2$ proves the second. A single large entry
therefore forces a large spectral radius. Neither identity proves a limiting
heavy-tail spectral law. Nor does a Wigner theorem apply directly to a
sample covariance matrix $D^TD/n$, with $D$ a centered data matrix;
that matrix is positive semidefinite and has a different construction.

```{figure} ../assets/figures/incerto/multivariate-spectra.svg
:alt: Three vertical panels compare full empirical eigenvalue CDFs for normal, variance-standardized Student 3, and Student 1.5 symmetric matrices. Each panel has its own linear horizontal scale.

Twelve independent $80\times80$ symmetric matrices per law, all divided by
$\sqrt{80}$; no eigenvalue is discarded. Each thin curve is a matrix's
empirical CDF. The normal and Student-3 entries have variance one; Student
1.5 has unit Student scale and infinite variance. The common divisor in the
last panel is a finite comparison convention, not a proposed limit scaling.
The dashed semicircle CDF appears only in the two finite-variance panels.
```

(multivariate-regression)=
## Regression with a stated population target

Consider $X\sim N(0,1)$ and an independent error $\varepsilon$ with
$\mathbb E\varepsilon=0$ and $0<\tau^2=\mathbb E\varepsilon^2<\infty$.
Define the response $V=b+aX+\varepsilon$; the letter $Y$ remains reserved
for mixing variance. Independence gives

$$
\operatorname{Cov}(X,V)=a,\qquad
\operatorname{Var}(V)=a^2+\tau^2,\qquad
R^2_{\mathrm{pop}}=\operatorname{Corr}(X,V)^2
=\frac{a^2}{a^2+\tau^2}.
$$

For iid pairs, the strong law applied to $X,V,X^2,V^2,XV$ proves that
sample correlation squared converges almost surely to this value. The
product is integrable by Cauchy–Schwarz. In a fitted simple ordinary
least-squares regression with an intercept and nonconstant columns,
write $S_{AB}=\sum_i(A_i-\bar A)(B_i-\bar B)$ for centered sums.
The normal equations give $\hat a=S_{XV}/S_{XX}$ and fitted centered
response $\hat a(X_i-\bar X)$. Its sum of squares is
$S_{XV}^2/S_{XX}$, so fitted $R_n^2=S_{XV}^2/(S_{XX}S_{VV})=r_n^2$.
This proves the connection to the population target. For the expectation
statement, extend $R_n^2$ by zero on samples with a constant column.
The strong law makes both denominators positive eventually almost surely;
this bounded extension therefore satisfies
$\mathbb E R_n^2\to R^2_{\mathrm{pop}}$ by bounded convergence.
The coefficient on a degenerate sample itself remains undefined; the
numerical helper returns NaN there.
It supplies no $O(n^{-2})$ error or universal upward-bias claim.

For Student error with scale $s>0$ and $\nu>2$,
$\tau^2=s^2\nu/(\nu-2)$. At $a=s=1$, $\nu=3$, the target is
$1/4$. The fourth error moment is infinite, so the ordinary variance-based
CLT for empirical second moments cannot be used; the strong-law argument
still works. When $\nu\le2$, this population Pearson target is undefined.
Letting a finite-variance family's $\nu$ decrease to two sends its target
to zero, but does not define a correlation at the boundary.

```{figure} ../assets/figures/incerto/multivariate-sampling.svg
:alt: Two vertical panels plot medians and 10th to 90th percentile bands of centered sample correlation and fitted regression R squared across four sample sizes.

PCG64 seed 20261018, 400 replications at each size 30, 100, 300, 1000.
Upper panel: bivariate Students with common scale and scatter parameter 0.6;
the dotted population line applies only to $\nu=5$. Lower panel: response
$V=X+\varepsilon$, independent normal $X$, and unit-scale Student-3 error;
the population target is $1/4$. Bands describe sampling variation, not
Monte Carlo confidence intervals. No draw is clipped or removed.
```

## Reproduce and read further

Run `uv run python demos/incerto/multivariate_tails.py`. It prints the
finite-model calculations, quadrature errors and sampling quantiles, and
creates these three figures plus the linked geometry and regime figures.
The [API reference](incerto-api-reference.md) lists the tested helper module.
The [chapter 6 guide](incerto-scoft-ch6.md) maps all sections and appendix A;
[moment estimation](incerto-moment-estimation.md) explains why existence,
estimator variance and convergence rate must be considered separately.

SCoFT v4 [@taleb2025scoftv4], §§6.1–6.3, pp. 108–114, motivates the
mixture and independence questions; §6.4, pp. 114–115, motivates mutual
information; §6.5, pp. 115–116, spectra; §6.6, pp. 116–117, sample
correlation; and §6.7, pp. 117–119, regression. These are context locators.
The source's historical regressions and general portfolio prescriptions
are not evidence for the original models proved here.
