---
title: Rayleigh speeds
options:
  concept:
    id: rayleigh-speed
    type: distribution
    related:
      - maximum-entropy
      - tail-class-catalog
      - mixture-uncertainty
    tags:
      - statistical-mechanics
      - distributions
      - sampling
---

The speed of a uniformly sampled particle in a two-dimensional equilibrium
gas has a **Rayleigh distribution**. Each signed velocity component is
Gaussian; taking the length of the velocity vector changes the law.
This answers Taleb's distribution challenge [@taleb2026speed]: the density
already written in the accompanying animation is Rayleigh [@matharium2026speed].
No extra physical parameter is needed.

The sampling rule matters. An event recorded when a particle crosses a small
hole favors faster particles. We derive the chamber law first, then the
effusion law and the finite-system correction. A seeded simulation compares
seven fitted families; it illustrates these distinctions without testing
whether the animation implements equilibrium dynamics.

(rayleigh-speed-density)=
## From components to speed

Assume a classical, nonrelativistic gas of equal-mass particles at canonical
equilibrium, with no bulk flow and no constraint on total momentum. Let
$m>0$ be particle mass, $T>0$ temperature, and $k$ Boltzmann's constant.
Let $d\ge1$ be an integer. Write the velocity as $\mathbf u\in\mathbb R^d$,
the speed as $V=\|\mathbf u\|$, and its realized value as $v$. Set

$$
\sigma^2=\frac{kT}{m},\qquad c=\frac1{2\sigma^2}.
$$

Here $\sigma$ is component standard deviation; it is not speed standard
deviation. Densities below use Lebesgue measure in their stated coordinates.
These symbols agree with the [shared notation](https://xshi19.github.io/math/notation/).

For $N$ particles at positions $\mathbf q_i$ and a velocity-independent
potential $U$, the Hamiltonian is
$H=\sum_{i=1}^N m\|\mathbf u_i\|^2/2+U(\mathbf q_1,\ldots,\mathbf q_N)$,
and the canonical weight $e^{-H/(kT)}$ separates into kinetic and positional
factors [@tongStatisticalPhysics, §2.3]. Integrating the positions and the
other particles leaves the single-particle density

$$
\begin{aligned}
\phi_d(\mathbf u)&=(2\pi\sigma^2)^{-d/2}\\
&\quad\times\exp\!\left(-\frac{\|\mathbf u\|^2}{2\sigma^2}\right).
\end{aligned}
$$

The factorization proves that the $d$ components are independent
$N(0,\sigma^2)$ variables. This follows from the equilibrium measure;
it does not require a central limit argument about collisions.
The [maximum-entropy note](incerto-maximum-entropy.md) explains why the
constraint and the coordinate measure must be specified in an entropy argument.

We now derive the speed density, its moments, and its mode. The surface area
of the unit sphere in $d$ dimensions is
$S_{d-1}=2\pi^{d/2}/\Gamma(d/2)$. Polar coordinates contribute the shell
factor $S_{d-1}v^{d-1}$, giving

$$
\boxed{
\begin{gathered}
f_d(v)=\frac{2c^{d/2}}{\Gamma(d/2)}v^{d-1}e^{-cv^2},\\
v>0.
\end{gathered}
}
$$

The density is zero for $v<0$, and $V/\sigma\sim\chi_d$.
Substitution $t=cv^2$ gives

$$
\begin{gathered}
\int_0^\infty v^{d+r-1}e^{-cv^2}\,dv\\
=\frac{\Gamma((d+r)/2)}{2c^{(d+r)/2}},\\
r>-d.
\end{gathered}
$$

Taking $r=0$ proves normalization. For every real $r>-d$,

$$
\begin{aligned}
&\mathbb E[V^r]\\
&\quad=(2\sigma^2)^{r/2}\frac{\Gamma((d+r)/2)}{\Gamma(d/2)}.
\end{aligned}
$$

For $r\le-d$ the integral diverges at zero. Every positive moment exists.
For $d>1$, differentiating the log density gives
$(d-1)/v-v/\sigma^2$, so the unique mode is $\sigma\sqrt{d-1}$.
For $d=1$ the law is half-normal and its mode is zero.

| Dimension | Law and density for $v>0$ | Mean | Variance | Mode |
| --- | --- | --- | --- | --- |
| $2$ | Rayleigh: $v\sigma^{-2}e^{-v^2/(2\sigma^2)}$ | $\sigma\sqrt{\pi/2}$ | $(4-\pi)\sigma^2/2$ | $\sigma$ |
| $3$ | Maxwell: $\sqrt{2/\pi}\,v^2\sigma^{-3}e^{-v^2/(2\sigma^2)}$ | $2\sigma\sqrt{2/\pi}$ | $(3-8/\pi)\sigma^2$ | $\sqrt2\sigma$ |

In two dimensions, integration also gives
$F(v)=1-e^{-v^2/(2\sigma^2)}$ for $v\ge0$.
This is a Weibull law with shape $2$ and scale $\sqrt2\sigma$.
Its right tail decays faster than any exponential, placing it on the light
side of the [tail-class catalog](incerto-tail-class-catalog.md).

(rayleigh-speed-families)=
## Generalized Gamma, not GIG

For a local shape $s>0$, define

$$
\begin{gathered}
h_s(v)=\frac{2c^s}{\Gamma(s)}v^{2s-1}e^{-cv^2},\\
v>0.
\end{gathered}
$$

This is a generalized Gamma density with power $2$; the chamber law has
$s=d/2$. The generalized inverse Gaussian
[GIG family](https://xshi19.github.io/math/normix-theory/normix-generalized-inverse-gaussian/)
is a different family. In the shared convention its kernel is
$v^{p-1}\exp[-(av+b/v)/2]$, with $p\in\mathbb R$ and $a,b>0$.

We can exclude Gamma, inverse Gamma, inverse Gaussian, and GIG by their
tails, without fitting any data. In the following kernels, shapes and
scales are positive, and all parameters are fixed as $v\to\infty$.

| Candidate | Density kernel on $v>0$ |
| --- | --- |
| Gamma, shape $s_0$, scale $\vartheta$ | $v^{s_0-1}e^{-v/\vartheta}$ |
| Inverse Gamma, shape $s_0$, scale $b_0$ | $v^{-s_0-1}e^{-b_0/v}$ |
| Inverse Gaussian, mean $\mu$, shape $\kappa$ | $v^{-3/2}e^{-\kappa v/(2\mu^2)-\kappa/(2v)}$ |
| $\operatorname{GIG}(p,a,b)$ | $v^{p-1}e^{-(av+b/v)/2}$ |

After taking logarithms, each candidate has only constant, logarithmic,
linear, and reciprocal terms. Consequently, for each such density $q$,

$$
\begin{gathered}
\lim_{v\to\infty}\frac{\log q(v)}{v^2}=0,\\
\lim_{v\to\infty}\frac{\log h_s(v)}{v^2}=-c<0.
\end{gathered}
$$

The unequal limits prove that none of these fixed-parameter laws equals
$h_s$. The argument includes the normalizable Gamma and inverse Gamma
boundaries of GIG. Log densities refer to a fixed choice of speed units;
changing units changes the negative constant but preserves the distinction.

Gaussian exclusion needs a separate argument: a nondegenerate Gaussian
assigns positive probability to negative speeds. Its log-density limit
is also negative, so the preceding tail test alone would not exclude it.
For $d\ge2$, even an ordinary Gaussian conditioned to be nonnegative fails:
its density has a positive right limit at zero, whereas $f_d(0+)=0$.
Positive support and skewness alone do not identify a family.

(rayleigh-speed-energy)=
## Squared speed and energy

Let $Q=V^2$ and let $\varepsilon=mV^2/2$ be kinetic energy. The change
of variable from $v$ to $q$ has Jacobian $1/(2\sqrt q)$, so under $h_s$,

$$
\begin{gathered}
\begin{aligned}
p_Q(q)&=\frac{h_s(\sqrt q)}{2\sqrt q}\\
&=\frac{c^s}{\Gamma(s)}q^{s-1}e^{-cq},
\end{aligned}\\
q>0.
\end{gathered}
$$

Using Gamma's **shape, scale** convention, this proves

$$
\begin{gathered}
V^2\sim\operatorname{Gamma}(s,2\sigma^2),\\
\varepsilon\sim\operatorname{Gamma}(s,kT).
\end{gathered}
$$

For the 2D chamber, $s=1$: squared speed is exponential with mean
$2kT/m$, and kinetic energy is exponential with mean $kT$.
For $R=1/Q$, a second Jacobian gives

$$
\begin{gathered}
p_R(r)=\frac{c^s}{\Gamma(s)}r^{-s-1}e^{-c/r},\\
r>0.
\end{gathered}
$$

Thus $1/V^2\sim\operatorname{InvGamma}(s,c)$.
Similarly $1/\varepsilon$ has inverse Gamma scale $1/(kT)$.
The reciprocal speed $1/V$ instead has an exponential factor $e^{-c/r^2}$;
it is not inverse Gamma.

If the normalizable boundaries are included,
$\operatorname{Gamma}(s,\vartheta)=\operatorname{GIG}(s,2/\vartheta,0)$
and $\operatorname{InvGamma}(s,b_0)=\operatorname{GIG}(-s,0,2b_0)$.
For the strict $a,b>0$ family these are limits. Transforming a Gamma
variable by a square root does not preserve membership in Gamma or GIG.

(rayleigh-speed-effusion)=
## Crossing a hole changes the sample

Assume the hole is small compared with the mean free path, the gas near it
is isotropic and equilibrated, and detection efficiency does not depend on
speed. In time $dt$, particles that can cross a small area come from a layer
of thickness $v\cos\theta\,dt$, where $\theta$ is the angle to the outward
normal and $\cos\theta>0$. This is the flux factor in the usual effusion
calculation [@fitzpatrickEffusion].

Multiplying the equilibrium velocity measure by this factor and integrating
over outward directions leaves one extra factor $v$. Normalizing proves

$$
\begin{gathered}
g_d(v)=\frac{vf_d(v)}{\mathbb E_f V}\\
=\frac{2c^{(d+1)/2}}{\Gamma((d+1)/2)}v^d e^{-cv^2},\\
v>0.
\end{gathered}
$$

The effusing speed divided by $\sigma$ therefore has law $\chi_{d+1}$.
In particular,

$$
g_2(v)=\sqrt{\frac2\pi}\frac{v^2}{\sigma^3}
e^{-v^2/(2\sigma^2)}=f_3(v).
$$

Two-dimensional effusion has the same speed law as a three-dimensional
chamber. The change comes from sampling, not an added spatial dimension.
Its mode is $\sigma\sqrt d$, and its squared speed and energy use
$s=(d+1)/2$ in the preceding Gamma formulas. Selecting only an outward
direction would not introduce the factor $v$; counting crossings does.

The animation shows particles leaving an opening and accumulating in a
histogram [@matharium2026speed]. If those are physical crossing events, the
effusion law is the appropriate equilibrium prediction. If the opening
merely conveys uniformly selected particles to a histogram, Rayleigh is
appropriate. The displayed formula specifies the chamber law; without the
sampling algorithm, the drawing does not resolve this distinction.

(rayleigh-speed-finite)=
## A finite isolated chamber

Canonical velocities have the preceding Gaussian components even for finite
$N$. An isolated finite system with fixed total kinetic energy has a different
joint measure. Assume $N>1$ ideal particles, or elastic hard disks or spheres,
in reflecting hard walls, with total kinetic energy $E>0$ and no imposed
total-momentum constraint. The velocity measure is uniform on the sphere

$$
\begin{gathered}
\sum_{i=1}^N\|\mathbf u_i\|^2=\frac{2E}{m}\\
\text{in }\mathbb R^{dN}.
\end{gathered}
$$

We derive its one-particle marginal. Integrating the remaining $d(N-1)$
coordinates against the energy-shell delta function supplies the factor
$(E-mv^2/2)^{d(N-1)/2-1}$. The first particle's radial element contributes
$v^{d-1}\,dv$. Put $v_{\max}=\sqrt{2E/m}$ and
$B(x,y)=\Gamma(x)\Gamma(y)/\Gamma(x+y)$. Substitution
$z=v^2/v_{\max}^2$ normalizes the density to

$$
\begin{gathered}
\begin{aligned}
&f_{N,d}(v)\\
&\quad=\frac{2v^{d-1}}{v_{\max}^{d}B(d/2,d(N-1)/2)}\\
&\qquad\times\left(1-\frac{v^2}{v_{\max}^2}\right)^{d(N-1)/2-1},
\end{aligned}\\
0<v<v_{\max}.
\end{gathered}
$$

Equivalently,

$$
\boxed{\frac{mV^2}{2E}\sim
\operatorname{Beta}\!\left(\frac d2,\frac{d(N-1)}2\right).}
$$

This is a scaled Beta energy law, with a finite upper endpoint
[@scalas2015velocities]. For the instantaneous effusion measure, multiplication
by $v$ changes the first Beta shape to $(d+1)/2$, leaving the second unchanged.

Define the kinetic temperature by $E=dNkT_{\rm kin}/2$. At fixed
$T_{\rm kin}$ and fixed speed, the finite-system factor tends to
$e^{-mv^2/(2kT_{\rm kin})}$ as $N\to\infty$, and the Gamma-ratio
asymptotic $\Gamma(z+a)/\Gamma(z)\sim z^a$ gives the canonical normalizer.
This proves the chi limit; it is not a uniform relative approximation all
the way to $v_{\max}$. Fixing total momentum as well would change the
available degrees of freedom and the Beta law.

Hard-core exclusion affects the allowed positions but introduces no
velocity-dependent potential, so it does not alter the canonical
factorization. Elastic collisions conserve energy; that alone does not
prove that an arbitrary initial condition has equilibrated. Noninteracting
point particles reflecting in a fixed box retain their individual speeds.

(rayleigh-speed-cooling)=
## Leaking cools the chamber

The moment formula gives mean energy $dkT/2$ inside and $(d+1)kT/2$
among effusing particles. A leak therefore removes more energy per particle
than the chamber average. In a large, unheated chamber that re-equilibrates
between losses, treat $N$ as continuous and use dots for time derivatives.
Energy balance gives

$$
\begin{gathered}
\frac d2 k(N\dot T+T\dot N)\\
=\frac{d+1}{2}kT\dot N,\\
N\dot T=\frac{T\dot N}{d}.
\end{gathered}
$$

Since $\dot N<0$, temperature falls. Starting at $N_0,T_0$, integration gives
$T/T_0=(N/N_0)^{1/d}$ within this quasistatic approximation.
If the outgoing event rate is $J(t)$, the accumulated speed
density over an observation interval is

$$
h_{\rm collected}(v)=
\frac{\int J(t)g_d(v;T(t))\,dt}{\int J(t)\,dt}.
$$

It generally mixes temperatures. The
[variance-mixture note](incerto-mixture-uncertainty.md) explains why averaging
over scales can change a marginal law. A fixed-temperature effusion density
does not describe an entire cooling history.

(rayleigh-speed-simulation)=
## A reproducible comparison

We draw 50,000 independent pairs of Gaussian components with $kT/m=1$, seed
20261022, and take their Euclidean lengths. All first components are drawn
before all second components. This samples the canonical chamber law directly;
it is neither molecular dynamics nor a sample extracted from the animation.

We fit Gaussian, Gamma, inverse Gamma, inverse Gaussian, GIG, Rayleigh, and
Weibull laws by maximum likelihood. Positive families have location fixed
at zero. Gaussian estimates mean and standard deviation; Rayleigh estimates
one scale; Weibull, Gamma, and the two inverse families estimate two parameters.
With $K$ estimated parameters and maximized log likelihood $\ell$, the
reported score is $\mathrm{AIC}=2K-2\ell$ in these fixed speed units.

```{figure} ../assets/figures/incerto/rayleigh-speed-fits.svg
:alt: Two vertical panels compare the simulated speed histogram with seven fitted densities on linear and logarithmic scales; the Gaussian extends below zero and the inverse families overstate the far right tail.

The same 50,000 speeds in 80 bins. Curves use 401 points from $-0.5$ to
the sample maximum. The log panel displays densities down to $10^{-5}$;
this plotting limit does not truncate the fitted laws. Rayleigh and Weibull
nearly coincide, as do Gamma and the GIG boundary fit.
```

The table's KS distance is $D=\sup_v|F_n(v)-F_{\hat\theta}(v)|$,
where $F_n$ is the empirical CDF. Its p-values are the ordinary two-sided
KS calculation **without correction for parameter fitting**. They are
descriptive diagnostics, not calibrated tests of the fitted families.
Calibration would require, for example, a parametric bootstrap with refitting
in each replicate [@scipy2026goodness]. Zeros below are numerical underflow,
not exact zero probabilities.

Use [`sample_chi_speed`](incerto-api-reference.md#api-sample-chi-speed),
[`fit_speed_models`](incerto-api-reference.md#api-fit-speed-models),
[`speed_fit_table`](incerto-api-reference.md#api-speed-fit-table), and
[`squared_speed_ks`](incerto-api-reference.md#api-squared-speed-ks).

:::{dropdown} Code and fit table

```python
from xmath.incerto.gas_speeds import (
    sample_chi_speed, fit_speed_models, speed_fit_table, squared_speed_ks,
)

v = sample_chi_speed(50_000, random_state=20261022)
fits = fit_speed_models(v)
print(speed_fit_table(fits))
energy = squared_speed_ks(v, thermal_variance=1)
print(f"V^2 vs Exp(mean 2): D={energy['statistic']:.4f}, p={energy['pvalue']:.4f}")
for fit in fits:
    if fit.boundary:
        print(f"{fit.name} boundary: {fit.boundary}")
```

**Output**

```text
Model              K        AIC     KS D   KS p (raw)
Rayleigh           1    93781.3   0.0034        0.618
Weibull            2    93783.3   0.0035        0.585
Gamma              2    95323.5   0.0327     5.43e-47
GIG                3    95325.5   0.0327     5.43e-47
Gaussian           2    99241.9   0.0471     5.06e-97
inverse Gaussian   2   110562.7   0.1170            0
inverse Gamma      2   123078.8   0.1338            0
V^2 vs Exp(mean 2): D=0.0058, p=0.0696
GIG boundary: Gamma
```
:::

The Rayleigh scale estimate is about $0.9967$. Weibull estimates shape
$2.0014$, close to Rayleigh's exact shape $2$; its extra parameter raises
AIC by about $2$. These are finite-sample fits, not a new derivation of the law.

(rayleigh-speed-gig-boundary)=
### Why the GIG fit lands on Gamma

The Gamma MLE has shape $\hat s=3.1513$ and scale
$\hat\vartheta=0.3964$. At its GIG representation
$(p,a,b)=(\hat s,2/\hat\vartheta,0)$, the inward derivative of log
likelihood per observation is

$$
\begin{aligned}
&\left.\frac1n\frac{\partial\ell}{\partial b}\right|_{b=0+}\\
&\quad=\frac12\Bigl(\frac1{(\hat s-1)\hat\vartheta}\\
&\qquad-\frac1n\sum_{i=1}^n\frac1{v_i}\Bigr)<0.
\end{aligned}
$$

This expression is finite because $\hat s>1$. Differentiating the
normalizing integral supplies the model reciprocal mean in the first term.
The other two score components vanish at the Gamma MLE. Log likelihood is
concave in the natural coordinates $(p-1,-a/2,-b/2)$: the Hessian of its
log normalizer is the covariance of $(\log V,V,1/V)$ in the interior.
The supporting-plane inequality therefore certifies a maximum on the
Gamma boundary. For strict $a,b>0$, this value is a supremum as $b\downarrow0$.

The helper uses that boundary law directly for its density and CDF. It
does not present tiny positive GIG parameters as a distinct distribution.
The table retains GIG's nominal $K=3$, hence a score exactly $2$ above
Gamma's. Ordinary regular-model AIC theory does not apply unchanged at
this boundary; the score here is a stated comparison convention.

### Quantiles and the squared-speed check

```{figure} ../assets/figures/incerto/rayleigh-speed-qq.svg
:alt: Two QQ panels compare five fitted speed laws with sample quantiles, then squared speeds with the specified exponential law. Rayleigh, Weibull and the squared-speed comparison follow the equality line closely.

Theoretical quantiles are horizontal and empirical quantiles vertical, at
199 equally spaced probabilities from $0.005$ to $0.995$. All displayed
quantiles are retained. These QQ
plots use the same observations as the fit table; they are not independent
validation data.
```

```{figure} ../assets/figures/incerto/rayleigh-speed-qq-tail.svg
:alt: Separate inverse-Gaussian and inverse-Gamma QQ panels retain their larger theoretical quantile ranges and show departures from the equality line.

The same probabilities and observations, with one panel per inverse family.
Their much larger fitted upper quantiles need separate horizontal ranges;
no quantiles are clipped to make the central fits look closer.
```

The squared-speed test compares against the **specified** exponential mean
$2$, without estimating it. Its KS statistic is $0.0058$ and its p-value is
$0.0696$, so it does not reject this simple null at the 5% level.
This p-value has the usual iid continuous-null calibration. It is not the
probability that the exponential model is true. The different Rayleigh
p-value in the table uses a fitted scale.

[`chi_speed`](incerto-api-reference.md#api-chi-speed) provides the model
density and CDF; [`speed_qq`](incerto-api-reference.md#api-speed-qq) constructs
quantile pairs. The three figures come from
[`speed_fit_figure`](incerto-api-reference.md#api-speed-fit-figure), using the
shared [`set_theme`](incerto-api-reference.md#api-set-theme),
[`vertical_figure`](incerto-api-reference.md#api-vertical-figure), and
[`savefig`](incerto-api-reference.md#api-savefig) helpers.
Regenerate them with `uv run python demos/incerto/rayleigh_speed.py`.
The page's text output is regenerated by
`uv run python scripts/regenerate_incerto_outputs.py`.

## Sources and context

- @taleb2026speed and @matharium2026speed: the challenge and animation.
- @tongStatisticalPhysics, §2.3: canonical velocity and Maxwell speed laws.
- @fitzpatrickEffusion: weighting wall crossings by normal velocity.
- @scalas2015velocities: finite microcanonical velocity and energy laws,
  including how boundary conditions change the constraints.
- @scipy2026goodness: goodness-of-fit calibration with estimated parameters.

Browse the [distribution catalog](incerto-distribution-concepts.md) for
related laws and the [API reference](incerto-api-reference.md) for the
reproducible calculations.
