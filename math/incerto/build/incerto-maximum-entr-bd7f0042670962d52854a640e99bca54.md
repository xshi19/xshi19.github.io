---
title: Maximum entropy and power tails
options:
  concept:
    id: maximum-entropy
    type: method
    related: [pareto, tail-class-catalog]
    tags: [entropy, constraints, fat-tails]
---

An entropy maximizer answers a constrained optimization question. Specifying
only “maximum entropy” leaves that question incomplete: the support, measure,
and constraints determine the answer. Here all densities are with respect
to Lebesgue measure in the stated coordinate, with natural logarithms and
fixed measurement units.

Appendix B of @taleb2025scoftv4, printed pp. 127–133, motivates the examples.
We prove the Shannon optima below by a relative-entropy bound. We then derive
the normalization and moments of a q-exponential family and distinguish them
from a global optimization theorem under escort constraints.

(maximum-entropy-shannon)=
## A certificate for a Shannon maximum

For a density $f$ on a fixed support $D$, its differential entropy is
$h(f)=-\int_D f(x)\log f(x)\,dx$, when defined. Consider a candidate

$$

g(x)=\exp\{\theta\mathbin{\cdot}t(x)-\psi(\theta)\},\qquad x\in D,
$$

with finite normalizer and prescribed expectations of the sufficient
statistics $t$. This is the [exponential-family form](https://xshi19.github.io/math/ig/information-geometry-exponential-families/);
$\theta$, $t$, and $\psi$ follow the [shared notation](https://xshi19.github.io/math/notation/).
Assume the candidate and competitors have the same finite constraints,
and $\int f|\log g|<\infty$.

We prove the entropy bound and its equality condition.

:::{prf:theorem} A Shannon entropy certificate under fixed constraints
:label: thm-maximum-entropy-shannon-certificate

Fix a measurable support $D\subseteq\mathbb R$ with Lebesgue measure.
Let $t:D\to\mathbb R^k$ be measurable and finite almost everywhere,
and let $\theta\in\mathbb R^k$ have finite normalizer
$\psi(\theta)=\log\int_D e^{\theta\cdot t(x)}\,dx$.
Suppose $g(x)=e^{\theta\cdot t(x)-\psi(\theta)}$ is feasible for
the constraints $\int_D g t_j=m_j$, with each $t_j$ integrable under $g$.
For every density $f$ on $D$ with the same finite expectations,
each $t_j$ integrable under $f$, and $\int_D f|\log g|<\infty$,

$$

h(g)-h(f)=\mathrm{KL}(f\Vert g)\ge0.
$$

Here $h(f)=-\int_D f\log f$ may equal $-\infty$, while $h(g)$ is
finite. Equality holds exactly when $f=g$ almost everywhere. Thus $g$
is the unique entropy maximizer in this class.
:::

:::{prf:proof}
With $r=f/g$,
$r\log r-r+1\ge0$ for $r\ge0$, with equality only at $r=1$.
Integrating against $g$ gives

$$
\mathrm{KL}(f\Vert g)=\int_D f\log(f/g)\,dx\ge0.
$$

Because $\log g$ is affine in the constrained statistics,
$\int f\log g=\int g\log g$. Consequently

$$

h(g)-h(f)=\mathrm{KL}(f\Vert g)\ge0.
$$

This identity also permits $h(f)=-\infty$ and infinite KL when the cross
entropy is finite. The negative part of $r\log r$ is integrable against
$g$, so the right side is well-defined. Equality requires $f=g$ almost
everywhere.
:::

This proves a global maximum once a feasible candidate exists. It does not
assert existence for arbitrary moment constraints.

### Uniform and Gaussian constraints

:::{prf:corollary} Uniform and Gaussian Shannon maxima
:label: cor-maximum-entropy-uniform-gaussian

On a fixed interval $[a,b]$, with finite $a<b$, the uniform density
$g=1/(b-a)$ uniquely maximizes differential entropy, with
$h(g)=\log(b-a)$. On $\mathbb R$, among densities with mean
$\mu\in\mathbb R$ and variance $\sigma^2\in(0,\infty)$,
the $N(\mu,\sigma^2)$ density uniquely maximizes entropy, with
$h(g)=\tfrac12\log(2\pi e\sigma^2)$. Uniqueness is up to equality
almost everywhere.
:::

:::{prf:proof}
Apply [](#thm-maximum-entropy-shannon-certificate). For the uniform
candidate the log density is constant, so normalization alone fixes its
cross entropy. For the Gaussian candidate,

$$
\log g(x)=-\tfrac12\log(2\pi\sigma^2)
           -\frac{(x-\mu)^2}{2\sigma^2}.
$$

The Gaussian integral gives normalization; the prescribed second moment
gives the stated entropy and finite cross entropy for every competitor.
The certificate supplies both maxima and equality conditions.
:::

There is no uniform probability density on the whole real line.
A half-line support would be a different optimization problem. These are
the scoped versions of the examples in §B.1.1 and §B.1.3, pp. 128–129.

(maximum-entropy-pareto)=
## A logarithmic constraint selects Pareto

The logarithmic constraint makes the [Pareto density](incerto-pareto.md)
a feasible exponential-family candidate.

:::{prf:corollary} Pareto maximizes entropy at a fixed log-moment
:label: cor-maximum-entropy-pareto

Fix $x_m>0$ and $m\in(0,\infty)$. Among Lebesgue densities on
$[x_m,\infty)$ with $\mathbb E\log(X/x_m)=m$, the unique Shannon
entropy maximizer, up to equality almost everywhere, is
$g(x)=\alpha x_m^\alpha x^{-\alpha-1}$, where $\alpha=1/m$.
Its entropy is $h(g)=\log(x_m m)+1+m$.
:::

:::{prf:proof}
Set $z=\log(x/x_m)$. The transformed candidate density is
$\alpha e^{-\alpha z}$ on $z\ge0$, so it is normalized and
$\mathbb E_g z=1/\alpha=m$. Also

$$
\log g(x)=\log(\alpha/x_m)-(\alpha+1)\log(x/x_m).
$$

Taking expectations gives the stated entropy. The finite log-moment
constraint ensures finite cross entropy for every competitor, so
[](#thm-maximum-entropy-shannon-certificate) gives the unique maximum.
:::

Here $m$ is a local log-moment, not $\mathbb EX$. No ordinary mean is
required: it is finite only when $m<1$, by the
[moment theorem](incerto-pareto-moment-existence.md). This supplies a complete
proof for the model considered in §B.1.2, p. 129.

:::{prf:proposition} Entropy under a change of coordinate
:label: prop-maximum-entropy-coordinate-change

Let $X$ have density $f$ on an interval $I$, and let $T:I\to T(I)$
be a differentiable increasing bijection with $T'>0$.
If $\mathbb E|\log f(X)|<\infty$ and
$\mathbb E|\log T'(X)|<\infty$, then
$h(T(X))=h(X)+\mathbb E\log T'(X)$.
In particular, for $c>0$, $h(cX)=h(X)+\log c$.
:::

:::{prf:proof}
The density of $U=T(X)$ at $T(x)$ is $f(x)/T'(x)$.
Substitution in its entropy integral gives
$-\mathbb E\log[f(X)/T'(X)]$; the integrability assumptions allow
the two terms to be separated. Set $T(x)=cx$ for the last assertion.
:::

The constraint uses a dimensionless logarithm. Changing units from $X$
to $cX$ changes $x_m$ by the same factor. A nonlinear change of coordinates
need not preserve an entropy ranking. For $T(x)=\log(x/x_m)$, the
correction is $-\mathbb E\log X=-\log x_m-m$, fixed by the constraint,
which is why maximizing in either coordinate gives the same candidate here.

(maximum-entropy-tsallis)=
## Tsallis entropy and escort constraints

For a local parameter $q>0$, $q\ne1$, define

$$

S_q(f)=\frac{1-\int f(x)^q\,dx}{q-1},\qquad
f_{[q]}(x)=\frac{f(x)^q}{\int f(y)^q\,dy},
$$

provided $0<\int f^q<\infty$. The second expression is an escort density.
Its mean $\int x f_{[q]}(x)\,dx$ is a different functional from
$\mathbb EX$. Differential densities can exceed one, so inequalities between
$f^q$ and $f$ require the value of $f$, as well as the sign of $q-1$.

:::{prf:proposition} Tsallis entropy of a product density
:label: prop-maximum-entropy-tsallis-product

Fix $q>0$, $q\ne1$. Let $f,g$ be densities of independent real
variables, with $0<\int f^q<\infty$ and $0<\int g^q<\infty$.
For $S_q(f)=(1-\int f^q)/(q-1)$ and joint density
$(f\otimes g)(x,y)=f(x)g(y)$,

$$

S_q(f\otimes g)=S_q(f)+S_q(g)+(1-q)S_q(f)S_q(g).
$$
:::

:::{prf:proof}
Tonelli gives $\int(f\otimes g)^q=(\int f^q)(\int g^q)$.
Substitute $\int f^q=1-(q-1)S_q(f)$ and the corresponding expression
for $g$, expand the product, and divide by $q-1$.
:::

This concerns the **joint** density of the pair. It is not an entropy
identity for the distribution of their sum. Since differential entropies
can be negative, a universal superadditive/subadditive label needs further
conditions. This distinction matters in §B.2, pp. 129–130.

:::{prf:proposition} Interior stationarity under an escort constraint
:label: prop-maximum-entropy-escort-stationarity

Fix $q>0$, $q\ne1$, and a support $D\subseteq\mathbb R$.
Let $f>0$ almost everywhere on $D$ be a normalized density with
$0<\int_D f^q<\infty$ and $\int_D |x|f^q<\infty$, and put
$m_q=\int_D xf^q/\int_D f^q$.
Suppose differentiation under the integrals is valid for the admissible
variations. If there are multipliers $\lambda_0,\lambda_1$ for which
the Lagrangian
$S_q(f)-\lambda_0(\int_D f-1)-\lambda_1(\int_D xf^q/\int_D f^q-m_q)$
has zero first variation against every bounded test variation supported
inside $D$, then, almost everywhere,

$$
-q f(x)^{q-1}\left[\frac1{q-1}
       +\frac{\lambda_1}{\int_D f^q}(x-m_q)\right]-\lambda_0=0.
$$
:::

:::{prf:proof}
The quotient rule gives

$$
\delta\left(\frac{\int_D xf^q}{\int_D f^q}\right)
=\frac q{\int_D f^q}\int_D (x-m_q)f(x)^{q-1}\delta f(x)\,dx.
$$

The first variation of $S_q$ is
$-q\int_D f^{q-1}\delta f/(q-1)$. Adding the normalization term
and using arbitrary test variations gives the displayed equation.
:::

This is the necessary interior stationarity calculation when the
Lagrange-multiplier condition holds. On an admissible branch it produces
a power of an affine function. The escort constraint is nonlinear in $f$;
this calculation proves neither multiplier existence nor a global maximum.

(maximum-entropy-q-family)=
## Normalize before interpreting the q-exponential

We prove normalization and distinguish the ordinary and escort means.

:::{prf:proposition} Normalization and moments of the q-exponential family
:label: prop-maximum-entropy-q-family

Let $\beta>0$ and $0<q<2$. For $q\ne1$, define

$$

f_{q,\beta}(x)=\beta(2-q)
 [1+(q-1)\beta x]^{-1/(q-1)}.
$$

For $1<q<2$ its support is $[0,\infty)$; for $0<q<1$ it is
$[0,1/((1-q)\beta)]$, with density zero beyond that endpoint.
At $q=1$ define $f_{1,\beta}(x)=\beta e^{-\beta x}$ for $x\ge0$.
These are normalized densities. For $q\ne1$, their survival on the
interior of the support is
$\bar F_{q,\beta}(x)=[1+(q-1)\beta x]^{-(2-q)/(q-1)}$;
at $q=1$ it is $e^{-\beta x}$.
For $1<q<2$, the survival exponent is $\alpha=(2-q)/(q-1)$,
and $\mathbb EX^p<\infty$ for $p>0$ exactly when $p<\alpha$,
with logarithmic divergence at equality. For $0<q\le1$, all positive
power moments exist. The ordinary and escort means satisfy

$$
\mathbb EX=\frac1{\beta(3-2q)}\quad(0<q<3/2),\qquad
\mathbb EX=\infty\quad(3/2\le q<2),
$$

$$
\int f_{q,\beta}^q=\beta^{q-1}(2-q)^q,\qquad
\mathbb E_{[q]}X=\frac1{\beta(2-q)}\quad(0<q<2).
$$

The unnormalized expression for $q\ge2$ on $[0,\infty)$ is not
integrable; the divergence is logarithmic at $q=2$.
:::

:::{prf:proof}
Substituting the affine factor $1+(q-1)\beta x$ as the integration variable
gives an unnormalized integral
$1/[\beta(2-q)]$ in both ranges. On the infinite support it converges
only for $q<2$; $q=2$ diverges logarithmically and $q>2$ also fails.
The survival on the interior of the support is

$$
\bar F_{q,\beta}(x)
=[1+(q-1)\beta x]^{-(2-q)/(q-1)}.
$$

For $1<q<2$, this is a Lomax survival with
$\alpha=(2-q)/(q-1)$ and scale $1/[(q-1)\beta]$.
The same power comparison as in the Pareto moment proof shows
$\mathbb EX^p<\infty$ exactly when $0<p<\alpha$; equality diverges.
All nonnegative power moments exist in the compact-support case.
Integrating this survival gives

$$
\mathbb EX=\frac1{\beta(3-2q)},\qquad 0<q<3/2.
$$

The ordinary mean is infinite for $3/2\le q<2$. In contrast, dividing
$f_{q,\beta}^q$ by its integral gives a density proportional to
$[1+(q-1)\beta x]^{-q/(q-1)}$. Integration gives

$$
\int f_{q,\beta}^q=\beta^{q-1}(2-q)^q,\qquad
\mathbb E_{[q]}X=\frac1{\beta(2-q)},\quad 0<q<2.
$$

At $q=1$, direct integration of the exponential density gives the same
values, also obtained by limits.
:::

For $q=5/4$ and $\beta=2$, the survival exponent is $3$, ordinary mean
is $1$, and escort mean is $2/3$. A zero escort mean on $[0,\infty)$
would force the escort law to concentrate at zero, impossible for this nondegenerate
density. Centering a coordinate shifts the support too; it cannot set that
mean to zero while preserving the same half-line. These restrictions qualify
the normalization step in §B.2, pp. 132–133.

(maximum-entropy-worked)=
## A worked comparison at fixed log-moment

Set $x_m=1$, $m=1/2$. In the coordinate $Z=\log X$, the maximizer is
exponential with rate $2$ and density $g_Z(z)=2e^{-2z}$ on $z\ge0$.
A competitor with the same mean is gamma with shape $2$ and rate $4$,
density $f_Z(z)=16z e^{-4z}$ on $z>0$.
Transforming back gives a log-gamma density $16\log(x)x^{-5}$ for $x\ge1$.
Both meet the log-moment constraint, but their ordinary means differ:
$2$ for the Pareto and $(4/3)^2=16/9$ for the competitor.

Numerical integration of the log-coordinate entropy plus the Jacobian
correction $m$ gives entropies approximately $0.806853$ and $0.690921$.
Their gap, $0.115932$, agrees with direct integration of
$f_Z\log(f_Z/g_Z)$ and is positive as the certificate requires.
Both calculations integrate to infinity; plotting endpoints do not truncate
either model.

```{figure} ../assets/figures/incerto/power-law-entropy.svg
:alt: Vertical panels compare two densities with the same mean log size, then q-exponential survival curves with compact, exponential and power tails.

Top: exact log-coordinate densities with mean $1/2$. Bottom: exact survival
functions at $\beta=2$, for $q=0.75,1,1.25,1.6$. The compact-support survival
reaches zero at $2$ and stays zero. The other laws retain infinite support;
the plotted range is finite. These are model calculations, with no sampled
or observed data.
```

Use [`entropy_log_constraint_example`](incerto-api-reference.md#api-entropy-log-constraint-example).

```python
from xmath.incerto.power_laws import entropy_log_constraint_example

result = entropy_log_constraint_example()
for name, value in result.items():
    print(f"{name}: {value:.4g}")
```

**Output**

```text
pareto_mass: 1
pareto_log_moment: 0.5
pareto_entropy: 0.8069
log_gamma_mass: 1
log_gamma_log_moment: 0.5
log_gamma_entropy: 0.6909
kl: 0.1159
```

This quadrature checks one worked case, not global optimality.

Figure regeneration: `uv run python demos/incerto/power_laws.py`. The helper checks the
constraint, entropy and KL integrals independently by adaptive quadrature,
and the demo evaluates 301 points for each panel. No fitted distribution or
uncertainty interval is involved. Selecting a log-moment constraint is a
modeling decision; the optimum alone supplies no empirical justification for it.

## References

- @taleb2025scoftv4, appendix B, §B.1, pp. 128–129, and §B.2, pp. 129–133:
  source context. The global Shannon proof and worked alternative above are
  developed here. The Tsallis section proves family properties and gives
  necessary stationarity, without asserting a general escort-constrained optimum.
- [Exponential families](https://xshi19.github.io/math/ig/information-geometry-exponential-families/):
  canonical sufficient statistics, natural parameters, and log partition.
- [Pareto moments](incerto-pareto-moment-existence.md) and
  [tail catalog](incerto-tail-class-catalog.md): moment and tail-class distinctions.
