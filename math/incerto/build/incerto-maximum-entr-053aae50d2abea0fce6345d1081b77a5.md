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

We prove the entropy bound and its equality condition. With $r=f/g$,
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
everywhere. This proves a global maximum, once a feasible candidate exists.
It does not assert existence for arbitrary moment constraints.

### Uniform and Gaussian constraints

On $[a,b]$ with finite $a<b$, normalization alone gives
$g=1/(b-a)$ and $h(g)=\log(b-a)$. Its log density is constant, so the
certificate applies to every competing density on that interval. There is
no corresponding uniform probability density on the whole real line.

On $\mathbb R$, fix $\mathbb EX=\mu$ and
$\mathbb E(X-\mu)^2=\sigma^2>0$. The Gaussian candidate has

$$
\log g(x)=-\tfrac12\log(2\pi\sigma^2)
           -\frac{(x-\mu)^2}{2\sigma^2},\qquad
h(g)=\tfrac12\log(2\pi e\sigma^2).
$$

Its normalization is the Gaussian integral, and taking the prescribed
second moment gives the displayed entropy. The certificate proves its
unique maximum among densities with those moments. A half-line support
would be a different optimization problem. These are the scoped versions
of the examples in §B.1.1 and §B.1.3, pp. 128–129.

(maximum-entropy-pareto)=
## A logarithmic constraint selects Pareto

Fix $x_m>0$, support $[x_m,\infty)$, and

$$
\mathbb E\log(X/x_m)=m,\qquad 0<m<\infty.
$$

Here $m$ is a local log-moment, not $\mathbb EX$. Put $\alpha=1/m$.
The [Pareto candidate](incerto-pareto.md) is
$g(x)=\alpha x_m^\alpha x^{-\alpha-1}$. To check feasibility, set
$z=\log(x/x_m)$; its transformed density is $\alpha e^{-\alpha z}$
on $z\ge0$. Thus $\mathbb E_g z=1/\alpha=m$. Also

$$
\log g(x)=\log(\alpha/x_m)-(\alpha+1)\log(x/x_m),
\qquad h(g)=\log(x_m m)+1+m.
$$

The entropy certificate proves that this Pareto law is the unique maximum
for the stated log-moment constraint. No ordinary mean is required: it is
finite only when $m<1$, by the [moment theorem](incerto-pareto-moment-existence.md).
This supplies a complete proof for the model considered in §B.1.2, p. 129.

The constraint uses a dimensionless logarithm. Changing units from $X$ to
$cX$ changes entropy by $\log c$ and changes $x_m$ by the same factor.
More generally, a differentiable increasing coordinate change $U=T(X)$
with positive derivative gives $h(U)=h(X)+\mathbb E\log T'(X)$ whenever
the terms are finite, by substitution in the density integral. A nonlinear
change of coordinates therefore need not preserve an entropy ranking.
For the logarithm above the correction is fixed by the constraint, which
is why maximizing in either coordinate gives the same candidate here.

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

For independent variables with densities $f,g$ and finite integrals of their
$q$th powers, Tonelli gives $\int(fg)^q=(\int f^q)(\int g^q)$.
Substitution proves

$$
S_q(f\otimes g)=S_q(f)+S_q(g)+(1-q)S_q(f)S_q(g).
$$

This concerns the **joint** density of the pair. It is not an entropy
identity for the distribution of their sum. Since differential entropies
can be negative, a universal superadditive/subadditive label needs further
conditions. This distinction matters in §B.2, pp. 129–130.

To see where the family in that section comes from, write $J=\int f^q$
and prescribe an escort mean $m_q$. For admissible variations at a positive
density, differentiation of the numerator and denominator gives

$$
\delta\left(\frac{\int xf^q}{J}\right)
=\frac qJ\int (x-m_q)f(x)^{q-1}\delta f(x)\,dx.
$$

A stationary density for $S_q$ subject to normalization and that constraint
therefore satisfies, for Lagrange multipliers $\lambda_0,\lambda_1$,

$$
-q f(x)^{q-1}\left[\frac1{q-1}
       +\frac{\lambda_1}{J}(x-m_q)\right]-\lambda_0=0.
$$

This is a necessary interior stationarity calculation, with variations for
which differentiation under the integral is justified. On an admissible
branch it produces a power of an affine function. The escort constraint
is nonlinear in $f$; we do not infer a global maximum from this calculation.

(maximum-entropy-q-family)=
## Normalize before interpreting the q-exponential

Take $\beta>0$. The following family is a probability density for $0<q<2$:

$$
f_{q,\beta}(x)=\beta(2-q)
 [1+(q-1)\beta x]^{-1/(q-1)}.
$$

For $1<q<2$ its support is $[0,\infty)$; for $0<q<1$ it is
$[0,1/((1-q)\beta)]$, with density zero beyond that endpoint.
At $q=1$ define $f_{1,\beta}(x)=\beta e^{-\beta x}$ for $x\ge0$.
We now prove normalization, the tail exponent, and the mean distinctions.

Substitution $v=1+(q-1)\beta x$ gives an unnormalized integral
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

These formulas also have the stated exponential values at $q=1$ by limits.
For $q=5/4$ and $\beta=2$, the survival exponent is $3$, ordinary mean
is $1$, and escort mean is $2/3$. A zero escort mean on $[0,\infty)$
would force the escort law to concentrate at zero, impossible for this nondegenerate
density. Centering a coordinate shifts the support too; it cannot set that
mean to zero while preserving the same half-line. These restrictions qualify
the normalization step in §B.2, pp. 132–133.

(maximum-entropy-worked)=
## A worked comparison at fixed log-moment

Set $x_m=1$, $m=1/2$. In the coordinate $Z=\log X$, the maximizer is
exponential with rate $2$. A competitor with the same mean is gamma with
shape $2$ and rate $4$, density $16z e^{-4z}$ on $z>0$.
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
