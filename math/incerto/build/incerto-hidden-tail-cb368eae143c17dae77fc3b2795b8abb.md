---
title: The tail beyond the sample maximum
options:
  concept:
    id: hidden-tail
    type: theorem
    depends_on:
      - pareto
      - pareto-moment-existence
      - frechet
    tags:
      - extreme-value-theory
      - estimation
---

A sample has a largest observation; an unbounded population has no largest
possible outcome. The probability beyond that observation can be small while
its contribution to a moment remains substantial. We derive both quantities
under iid sampling, then separate them from empirical CDF accuracy.
SCoFT v4 §§9.2–9.3, pp. 177–181, motivates the comparison
[@taleb2025scoftv4]. The calculations below use exact finite-sample laws.

(hidden-tail-missing-mass)=
## Missing probability

Let $X_1,\ldots,X_n$ be iid from a continuous CDF $F$, let
$M_n=\max_iX_i$, and write $\bar F=1-F$. Define the random missing mass
$U_n=\bar F(M_n)$. The probability integral transform makes the
$F(X_i)$ independent uniform variables, so, for $0\le u\le1$,

$$
\begin{aligned}
P(U_n>u)&=(1-u)^n,\\
U_n&\sim\operatorname{Beta}(1,n),\\
\mathbb E[U_n]&=\frac1{n+1}.
\end{aligned}
$$

The first identity follows by requiring every $F(X_i)$ to be below $1-u$;
integrating it from zero to one gives the expectation. It also proves that
an independent next draw sets a new upper record with probability $1/(n+1)$.
This result concerns probability, regardless of the sizes of the outcomes.
Continuity excludes ties; a discrete parent needs a different calculation.

For fixed $t\ge0$, eventually $t<n$ and

$$
P(nU_n>t)=(1-t/n)^n\longrightarrow e^{-t}.
$$

Thus $nU_n$ converges to an exponential law with rate one. An exponential
law for the unscaled missing mass is an approximation, not its exact
finite-$n$ distribution; the latter is supported on $[0,1]$.

Use [`maximum_missing_mass`](incerto-api-reference.md#api-maximum-missing-mass).

```python
from xmath.incerto.extremes import maximum_missing_mass

for n in (10, 100, 1000):
    law = maximum_missing_mass(n)
    print(f"n={n}: mean={law.mean():.4g}, 95% quantile={law.ppf(.95):.4g}")
```

**Output**

```text
n=10: mean=0.09091, 95% quantile=0.2589
n=100: mean=0.009901, 95% quantile=0.02951
n=1000: mean=0.000999, 95% quantile=0.002991
```

(hidden-tail-pareto)=
## Hidden Pareto moments

Now assume the entire parent is [Pareto Type I](incerto-pareto.md), with
lower cutoff $x_m>0$ and exponent $\alpha>0$. For a nonnegative order $p$
and fixed $t\ge x_m$, define the *unconditional* tail contribution

$$
T_p(t)=\mathbb E[X^p\mathbf1_{\{X>t\}}].
$$

When $p<\alpha$, integrating the Pareto density gives

$$
\begin{aligned}
T_p(t)&=\int_t^\infty\alpha x_m^\alpha x^{p-\alpha-1}\,dx\\
&=\frac{\alpha x_m^p}{\alpha-p}
       \left(\frac{x_m}{t}\right)^{\alpha-p}.
\end{aligned}
$$

For $p\ge\alpha$ this integral diverges for every finite $t$; the
[population moment](incerto-pareto-moment-existence.md) is infinite.
There is then no finite moment fraction to report. For $p<\alpha$, let
$r=1-p/\alpha>0$ and define the random fraction

$$
R_{n,p}=\frac{T_p(M_n)}{\mathbb E[X^p]}=U_n^r.
$$

This uses a new population draw $X$ independent of the sample defining
$M_n$. It does not assign unobserved values to that sample. Transforming
the Beta law proves, for $0\le h\le1$,

$$
P(R_{n,p}\le h)=1-(1-h^{1/r})^n.
$$

Integrating against the Beta density gives the exact expected fraction,
with $B$ and $\Gamma$ denoting beta and gamma functions:

$$
\begin{aligned}
\mathbb E[R_{n,p}]&=nB(1+r,n)\\
&=\frac{\Gamma(n+1)\Gamma(1+r)}{\Gamma(n+1+r)}\\
&\sim\Gamma(1+r)n^{-r}.
\end{aligned}
$$

The last equivalence follows from Stirling's gamma-ratio asymptotic for
fixed $r$. For the mean, $r=1-1/\alpha$ and $\alpha>1$ is required.
As $\alpha$ approaches one, this decay exponent approaches zero. The
probability fraction still has expectation $1/(n+1)$.

Use [`pareto_hidden_moment`](incerto-api-reference.md#api-pareto-hidden-moment)
and [`expected_pareto_hidden_fraction`](incerto-api-reference.md#api-expected-pareto-hidden-fraction).
The infinite moment and undefined fraction are visible in the output.

```python
from xmath.incerto.extremes import pareto_hidden_moment, expected_pareto_hidden_fraction

for alpha in (.8, 1, 1.2, 1.5, 2):
    tail = pareto_hidden_moment(100, alpha)
    fraction = expected_pareto_hidden_fraction(1000, alpha)
    print(f"alpha={alpha:g}: mean above 100={tail:.4g}, expected fraction={fraction:.4g}")
```

**Output**

```text
alpha=0.8: mean above 100=inf, expected fraction=nan
alpha=1: mean above 100=inf, expected fraction=nan
alpha=1.2: mean above 100=2.389, expected fraction=0.2933
alpha=1.5: mean above 100=0.3, expected fraction=0.08928
alpha=2: mean above 100=0.02, expected fraction=0.02801
```

(hidden-tail-figure)=
```{figure} ../assets/figures/incerto/hidden-tail-pareto.svg
:alt: Expected hidden mean fractions for Pareto exponents 1.1, 1.5 and 2 decrease much more slowly than missing probability as sample size increases. A second panel shows sampling quantiles at exponent 1.5.

Exact Beta-law calculations, $x_m=1$, $n=10$ to $10^6$ on logarithmic
axes. Top: expected hidden mean fractions and expected missing probability.
Bottom: median and pointwise 10th/90th sampling quantiles of the hidden
mean fraction for $\alpha=1.5$, with its expectation dashed. The band
is a distribution of sample-to-sample variation under the known parent,
not a confidence interval for a fitted law. No simulation is used.
```

A positive expected hidden contribution does **not** imply that the sample
mean is biased downward. For $\alpha>1$, linearity of expectation gives
$\mathbb E[\bar X_n]=\mathbb E[X]$. A sample maximum and an independent
future draw play different roles in $T_p(M_n)$. Adding that quantity to the
sample mean is not an unbiasedness correction. The
[plug-in note](incerto-plug-in-tail-estimation.md#plugin-hidden-tail)
explains the modeling required to estimate a missing contribution.

(hidden-tail-gaussian)=
## Comparison with a normal parent

For standard normal $X$, write $\phi$ for its density and $\Phi$ for its
CDF. Since $\phi'(x)=-x\phi(x)$, direct integration gives, for every real $t$,

$$
\mathbb E[X\mathbf1_{\{X>t\}}]=\phi(t).
$$

The exact maximum density is $n\phi(t)\Phi(t)^{n-1}$. Consequently,

$$
\begin{gathered}
\mathbb E[T_1(M_n)]\\
=\int_{-\infty}^{\infty}n\phi(t)^2\Phi(t)^{n-1}\,dt.
\end{gathered}
$$

All maxima are included, even negative ones. This is a positive upper-tail
contribution, not a fraction of the normal mean, which is zero. For a
centered normal with standard deviation $\sigma$, multiply this expression
by $\sigma$; dividing by $\sigma$ recovers the displayed quantity.

For stable numerical integration set $W=-n\log\Phi(M_n)$. Its survival
is $e^{-w}$ for $w\ge0$, since $\Phi(M_n)$ has CDF $v^n$. Thus $W$ is
exponential with rate one, and the integral can be evaluated as
$\mathbb E[\phi(\Phi^{-1}(e^{-W/n}))]$. This is an exact change of variables;
the helper's quadrature is numerical and reports an estimated integration error.

Use [`expected_gaussian_hidden_mean`](incerto-api-reference.md#api-expected-gaussian-hidden-mean).

```python
from xmath.incerto.extremes import expected_gaussian_hidden_mean

for n in (10, 100, 1000):
    result = expected_gaussian_hidden_mean(n)
    print(f"n={n}: contribution/sigma={result['value']:.4g}, error={result['abs_error']:.2g}")
```

**Output**

```text
n=10: contribution/sigma=0.1442, error=1.8e-11
n=100: contribution/sigma=0.02486, error=2.4e-11
n=1000: contribution/sigma=0.003238, error=5.9e-12
```

```{figure} ../assets/figures/incerto/hidden-tail-normal.svg
:alt: The normal upper-tail mean contribution in standard-deviation units decreases with sample size. Its ratio to expected missing probability grows, showing that event sizes matter even for a normal parent.

Numerical quadrature of the exact normal-maximum integral, $n=10$ to
$10^5$. Top: expected upper-tail mean contribution in standard-deviation
units. Bottom: that expectation divided by $1/(n+1)$; this is the
conditional mean of an independent normal draw given that it sets a new
record. It is a ratio of expectations, not the average of conditional
means at realized maxima. Both axes are logarithmic. These curves use
neither simulated maxima nor a Gumbel approximation.
```

(hidden-tail-empirical)=
## What the empirical distribution estimates

The empirical probability measure $\widehat P_n=n^{-1}\sum_i\delta_{X_i}$
is a valid empirical distribution. Its strict survival is

$$
\widehat{\bar F}_n(u)=\frac1n\sum_i\mathbf1_{\{X_i>u\}}.
$$

At a fixed, nonrandom $u$, independence makes the numerator binomial
with success probability $q=\bar F(u)$. The empirical survival fraction
has expectation $q$ and variance $q(1-q)/n$. For $0<q<1$ the ordinary Bernoulli CLT gives

$$
\begin{gathered}
\sqrt n\bigl(\widehat{\bar F}_n(u)-q\bigr)\\
\Rightarrow\mathcal N(0,q(1-q)).
\end{gathered}
$$

This is a pointwise CLT, not the functional Donsker theorem. The CDF value
$F(x)$ at a fixed $x$ is a number; it is $F(X)$ that is uniform for a
continuous parent. At the *random* threshold $M_n$, strict empirical
survival is zero while the population survival has the Beta law above.
The fixed-threshold binomial argument does not apply after replacing $u$
by the observed maximum.

Small absolute probability error need not control an unbounded payoff.
For instance, with $\delta_x$ a unit mass at $x$, define

$$
\begin{gathered}
P_0=\delta_0,\\
Q_\varepsilon=(1-\varepsilon)\delta_0
                  +\varepsilon\delta_{\varepsilon^{-2}},\\
0<\varepsilon<1.
\end{gathered}
$$

Their CDFs differ by at most $\varepsilon$, but their means differ by
$1/\varepsilon$. This proves that uniform CDF proximity alone does not
bound mean error; an integrability or tail bound must supply the missing
control. [Inference asymmetry](incerto-inference-asymmetry.md) develops
the broader finite-sample identification problem.

Rank plotting positions such as $1-i/(n+1)$ stay positive at the largest
ordered observation and can differ at ties. They are not strict empirical
survival. The existing [tied-count example](incerto-counting-exceedances.md#counting-exceedances-ties)
shows the distinction. Ordinary empirical bootstrap draws cannot exceed
$M_n$; a fitted-parent simulation can, conditional on its model assumptions.
See [max/sum calibration](incerto-max-to-sum-ratio.md) for the resampling
limits. None of these facts makes empirical CDFs invalid.

## Reproduction and source qualifications

Run `uv run python demos/incerto/extreme_values.py` for the two hidden-tail
figures and the linked Gaussian/Hill examples. The formulas here are proved
for exact Pareto or normal parents, not for every regularly varying tail at
an unspecified finite cutoff. No empirical stock-loss multiplier is inferred.

SCoFT v4 §9.2, pp. 178–179, motivates hidden moments. Our domain is
$p<\alpha$; the complementary finite integral below a cutoff must not be
identified with the divergent upper-tail integral when $p\ge\alpha$.
Section 9.2.1, pp. 179–180, supplies the normal comparison context;
§9.3, pp. 180–181, motivates the distinction between probability and
payoff accuracy [@taleb2025scoftv4]. The pointwise CLT used above is the
ordinary iid finite-variance CLT, as in @durrett2019probability,
Theorem 3.4.1, p. 144, in the January 2019 manuscript.
