---
title: Generalized inverse Gaussian distribution
---

The generalized inverse Gaussian (GIG) distribution supplies the positive
mixing variable for the [GH family](./normix-generalized-hyperbolic.md).
Its normalizing integral also gives the posterior moments used by
[EM](./normix-em-algorithm.md). A standard reference is Bent Jørgensen,
*Statistical Properties of the Generalized Inverse Gaussian Distribution*
([1982; electronic edition 2012](https://link.springer.com/book/10.1007/978-1-4612-5698-4)).

## Definition and normalizing integral

Write $Y\sim\operatorname{GIG}(p,a,b)$, where $p\in\mathbb R$ and $a,b>0$.
The density on $y>0$ is

```{math}
:label: gig-pdf
f(y\mid p,a,b)=\frac{(a/b)^{p/2}}{2K_p(\sqrt{ab})}
y^{p-1}\exp\!\left[-\frac12(ay+b/y)\right].
```

Here $K_p$ is the modified Bessel function of the second kind. Its
[integral representations, NIST DLMF §10.32](https://dlmf.nist.gov/10.32)
give the identity

```{math}
:label: normix-gig-integral
\int_0^\infty y^{p-1}e^{-(ay+b/y)/2}\,dy
=2(b/a)^{p/2}K_p(\sqrt{ab}).
```

We use the open interior unless a boundary is named explicitly. A normalized
boundary law also exists for $a=0,b>0,p<0$ (inverse gamma), or
$b=0,a>0,p>0$ (gamma). Formula [](#gig-pdf) must then be interpreted by its
limit, not by substituting zero into its Bessel ratio.

The symbol $Y$ denotes a mixing variable here. In
[conditioning a mixture](./normix-conditioning-a-mixture.md), it is called
$W$; it is independent of the standard normal noise $Z$ in that construction.

## Scale and concentration parameters

Set $\delta=\sqrt{b/a}>0$ and $\omega=\sqrt{ab}>0$, so
$a=\omega/\delta$ and $b=\omega\delta$. The source calls the latter
parameter $\eta$; here $\eta$ is reserved for expectation coordinates.
Then

```{math}
:label: gig-pdf-alt
f(y\mid p,\delta,\omega)=\frac{\delta^{-p}}{2K_p(\omega)}
y^{p-1}\exp\!\left[-\frac{\omega}{2}
\left(\frac{y}{\delta}+\frac{\delta}{y}\right)\right].
```

Indeed $Y/\delta\sim\operatorname{GIG}(p,\omega,\omega)$, which verifies the
power $\delta^{-p}$. More generally, for $c>0$,

```{math}
cY\sim\operatorname{GIG}(p,a/c,bc).
```

This scaling identity is central to GH identifiability.

## Moment generating function and moments

Multiplication by $e^{uy}$ changes $a$ to $a-2u$ in the normalizing integral.
For $u<a/2$ this gives

```{math}
\begin{aligned}
M_Y(u)&=\left(\frac{a}{a-2u}\right)^{p/2}
\frac{K_p(\sqrt{b(a-2u)})}{K_p(\sqrt{ab})}\\
&=\left(\frac{\omega}{\omega-2\delta u}\right)^{p/2}
\frac{K_p(\sqrt{\omega^2-2\omega\delta u})}{K_p(\omega)}.
\end{aligned}
```

At $u=a/2$ the integral is finite only if $p<0$; for $u>a/2$ it diverges.
For every real $r$, multiplication by $y^r$ instead changes the Bessel order:

```{math}
:label: gig-moments
\mathbb E[Y^r]=(b/a)^{r/2}
\frac{K_{p+r}(\sqrt{ab})}{K_p(\sqrt{ab})}
=\delta^r\frac{K_{p+r}(\omega)}{K_p(\omega)}.
```

Thus $\mathbb E[Y]=m_1$ and $\operatorname{Var}(Y)=m_2-m_1^2$, where
$m_r=\mathbb E[Y^r]$. Differentiating at $r=0$ also yields an exact
special-function expression for the logarithmic moment:

```{math}
\mathbb E[\log Y]=\frac12\log(b/a)
+\left.\partial_\nu\log K_\nu(\sqrt{ab})\right|_{\nu=p}.
```

This is a derivative with respect to the **order**, not the argument of $K$.
Numerical evaluation may require quadrature or order derivatives; the absence
of an elementary formula does not make the identity approximate.

## Tails and moment boundaries

With $C=(a/b)^{p/2}/[2K_p(\sqrt{ab})]$, the two endpoint asymptotics are

```{math}
:label: gig-tails
\begin{aligned}
f(y)&\sim C y^{p-1}e^{-ay/2},&&y\to\infty,\\
f(y)&\sim C y^{p-1}e^{-b/(2y)},&&y\downarrow0.
\end{aligned}
```

The exponential cutoffs make every real power moment finite for $a,b>0$.
The right tail is exponential rather than a power law; the origin is suppressed
faster than any power. At the boundaries the moment domains change:

| Boundary law | Density kernel | Finite power moments |
| --- | --- | --- |
| $\operatorname{Gamma}(p,a/2)$, $p>0$ | $y^{p-1}e^{-ay/2}$ | $\mathbb E[Y^r]<\infty$ iff $r>-p$ |
| $\operatorname{InvGamma}(-p,b/2)$, $p<0$ | $y^{p-1}e^{-b/(2y)}$ | $\mathbb E[Y^r]<\infty$ iff $r<-p$ |

Gamma uses a shape and a **rate**; inverse gamma uses a shape and the
coefficient of $1/y$ in its exponent. See the
[family tour](./normix-gh-family-tour.md) for their normal mixtures.

## Skewness and kurtosis

Writing $v=m_2-m_1^2$, the central moments are

```{math}
\begin{aligned}
\mu_3&=m_3-3m_1m_2+2m_1^3,\\
\mu_4&=m_4-4m_1m_3+6m_1^2m_2-3m_1^4.
\end{aligned}
```

The standardized skewness and excess kurtosis are

```{math}
:label: gig-kurtosis
\gamma_1=\mu_3/v^{3/2},\qquad \gamma_2=\mu_4/v^2-3.
```

They are finite in the interior. At the inverse-gamma boundary with shape
$\alpha=-p$, the excess kurtosis is

```{math}
\gamma_2=\frac{6(5\alpha-11)}{(\alpha-3)(\alpha-4)},\qquad \alpha>4.
```

It diverges as $\alpha\downarrow4$ and is undefined for $\alpha\leq4$.
It does **not** diverge for every inverse-gamma boundary: shapes above four
retain a finite fourth moment.

## Exponential-family coordinates

Use the same statistic order throughout this batch:

```{math}
t(y)=(\log y,\ y^{-1},\ y)^\top,\qquad
f_\theta(y)=\mathbf1_{y>0}\exp\{\theta^\top t(y)-\psi(\theta)\}.
```

The natural parameters and their inverse map are

```{math}
:label: gig-natural-params
\theta=(p-1,-b/2,-a/2)^\top,\qquad
p=\theta_1+1,\quad b=-2\theta_2,\quad a=-2\theta_3.
```

The interior is $\mathbb R\times(-\infty,0)^2$, and the log-partition is

```{math}
:label: gig-log-partition
\psi(\theta)=\log2+\log K_p(\sqrt{ab})+\frac p2\log(b/a).
```

Differentiation yields

```{math}
:label: gig-expectation-params
\begin{aligned}
\eta_1&=\mathbb E[\log Y]=\tfrac12\log(b/a)
 +\left.\partial_\nu\log K_\nu(\sqrt{ab})\right|_{\nu=p},\\
\eta_2&=\mathbb E[Y^{-1}]=(a/b)^{1/2}
 \frac{K_{p-1}(\sqrt{ab})}{K_p(\sqrt{ab})},\\
\eta_3&=\mathbb E[Y]=(b/a)^{1/2}
 \frac{K_{p+1}(\sqrt{ab})}{K_p(\sqrt{ab})}.
\end{aligned}
```

The [exponential-family core](./normix-exponential-family-core.md) relates
$\eta=\nabla\psi$ and $\nabla^2\psi=\operatorname{Cov}(t(Y))$ to Fisher
information. The [IG entry](https://xshi19.github.io/math/ig/information-geometry-exponential-families/)
states the regularity and minimality assumptions behind these identities.

## Maximum likelihood and numerical limits

For independent positive observations $y_1,\ldots,y_n$, set
$\widehat\eta=n^{-1}\sum_i(\log y_i,y_i^{-1},y_i)^\top$. Up to a term
independent of the candidate parameters, the average log likelihood is

```{math}
:label: gig-loglik
L_{\mathrm{GIG}}(p,a,b\mid\widehat\eta)
=(p-1)\widehat\eta_1-\tfrac b2\widehat\eta_2
-\tfrac a2\widehat\eta_3-\psi(p-1,-b/2,-a/2).
```

The MLE, when attained in the interior, is

```{math}
:label: gig-mle
(\widehat p,\widehat a,\widehat b)
=\operatorname*{arg\,max}_{p\in\mathbb R,\ a,b>0}
L_{\mathrm{GIG}}(p,a,b\mid\widehat\eta).
```

It matches the three moments in [](#gig-expectation-params). Strict convexity
in natural coordinates gives uniqueness of an interior optimum, but does not
guarantee that one exists for every empirical moment vector. At fixed $p$,
only the inverse and first moments are matched. Near boundaries, Bessel
ratios and nearly dependent statistics can make inversion ill-conditioned.
A numerical failure or a large parameter error alone does not prove that an
MLE is nonexistent; see [why not gradient descent](./normix-why-not-gradient-descent.md).

## Hellinger distance

For densities $f_1,f_2$, use the convention
$H^2(f_1,f_2)=1-\int\sqrt{f_1f_2}$. Let bars denote arithmetic means of the
two GIG parameter triples. Applying [](#normix-gig-integral) to the geometric
mean of the densities gives

```{math}
H_{\mathrm{GIG}}^2=1-
\frac{(a_1/b_1)^{p_1/4}(a_2/b_2)^{p_2/4}}
{\sqrt{K_{p_1}(\sqrt{a_1b_1})K_{p_2}(\sqrt{a_2b_2})}}
\left(\frac{\bar b}{\bar a}\right)^{\bar p/2}
K_{\bar p}(\sqrt{\bar a\bar b}).
```

Equivalently, the affinity $1-H^2$ is
$\exp\{\psi((\theta^{(1)}+\theta^{(2)})/2)
-[\psi(\theta^{(1)})+\psi(\theta^{(2)})]/2\}$.
This compares distributions even when individual parameter coordinates are
poorly conditioned. The upstream note attributes this application to
Shi (2016), *Generalized Hyperbolic Distributions and Related Topics*.

## Special cases and implementation

The inverse Gaussian with mean $m>0$ and shape $\lambda>0$ is exactly
$\operatorname{GIG}(-1/2,\lambda/m^2,\lambda)$. Gamma and inverse gamma
are the boundary cases above. Their induced marginals are described in the
[GH family tour](./normix-gh-family-tour.md).

Executable examples remain in the
[upstream GIG tutorial](https://xshi19.github.io/normix/tutorials/distributions/02_gig.html);
implementation details remain in the
[package API](https://xshi19.github.io/normix/api/index.html) and
[Bessel and solver design](https://xshi19.github.io/normix/design/solvers_and_bessel.html).

## Source and adaptation

Adapted from `xshi19/normix`, `docs/theory/gig.md`, at revision
`763bb3608920661a012cf089888d349fbf680aad` (2026-09-13 import).
Copyright (c) 2020 xshi19. Licensed under MIT.
The [pinned source](https://github.com/xshi19/normix/blob/763bb3608920661a012cf089888d349fbf680aad/docs/theory/gig.md)
records the original version. Notation, mathematical qualifications, and links
were adapted for this site; API recipes, executable package cells, and notebook
plots are omitted. No upstream benchmark execution or formal-proof verification
is claimed for this adaptation.

:::{dropdown} MIT permission notice

MIT License

Copyright (c) 2020 xshi19

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
