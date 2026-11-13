---
title: Latent variables and ordinary EM
---

A latent-variable model specifies a joint law for an observation $X$ and an
unobserved variable $Z$. The likelihood of the observed data uses the marginal
law of $X$. The EM algorithm relates these two likelihoods through the
conditional distribution of $Z$ given the actual observations.

## Joint, marginal, and posterior

Let $P_\theta(x,z)$ be a joint density with respect to fixed measures
$\nu(dx)\lambda(dz)$. Its marginal and posterior densities are

```{math}
:label: ig-joint-marginal-posterior
p_\theta(x)=\int P_\theta(x,z)\,d\lambda(z),\qquad
r_\theta(z\mid x)=\frac{P_\theta(x,z)}{p_\theta(x)}.
```

The posterior is defined where $p_\theta(x)>0$. Integrals become sums for
discrete latent variables. Assume throughout the following derivation that the
densities have common positive support at the observations and that the log
densities, posterior entropies, and KL terms used below are integrable.

For independent observations $x_1,\ldots,x_n$, the objective is the average
observed log likelihood

```{math}
\ell_n(\theta)=\frac1n\sum_{i=1}^n\log p_\theta(x_i).
```

Even if $\log P_\theta(x,z)$ is easy to optimize, the integral inside
$\log p_\theta(x)$ may make direct optimization difficult.

## A variational identity at each observation

For any collection $q=(q_1,\ldots,q_n)$ of conditional densities $q_i(z)$
with the required support and finite terms, define

```{math}
\mathcal F(q,\theta)
=\frac1n\sum_{i=1}^n
\int q_i(z)\log\frac{P_\theta(x_i,z)}{q_i(z)}\,d\lambda(z).
```

For a comparison density $r$, KL divergence is
$D_{\mathrm{KL}}(q_i\|r)=\int q_i\log(q_i/r)\,d\lambda$, with natural logarithms.
It is nonnegative: Jensen's inequality applied to $-\log$ gives
$\mathbb E_{q_i}[-\log(r/q_i)]\ge-\log\mathbb E_{q_i}[r/q_i]=0$
when the supports agree. The more general support case gives the same
inequality, or infinite KL if $r$ vanishes on a set of positive $q_i$ mass.
Equality holds exactly when the two distributions agree.

Substituting $P_\theta(x_i,z)=p_\theta(x_i)r_\theta(z\mid x_i)$ into
$\mathcal F$ gives the exact identity

```{math}
:label: ig-em-lower-bound
\ell_n(\theta)
=\mathcal F(q,\theta)
 +\frac1n\sum_{i=1}^n
   D_{\mathrm{KL}}\!\left(q_i\|r_\theta(\cdot\mid x_i)\right).
```

Thus $\mathcal F$ is a lower bound on the average observed log likelihood,
tight when each $q_i$ is the posterior at the current parameter. This is the
free-energy formulation of EM developed by Neal and Hinton in
[A view of the EM algorithm that justifies incremental, sparse, and other variants](https://www.cs.utoronto.ca/~hinton/csc2535_06/readings/emk.pdf).

## The two steps and the likelihood inequality

At iteration $k$, the **E-step** sets
$q_k=(q_{k,1},\ldots,q_{k,n})$ with
$q_{k,i}=r_{\theta_k}(\cdot\mid x_i)$ and forms

```{math}
:label: ig-em-q-function
Q(\theta\mid\theta_k)
=\frac1n\sum_{i=1}^n
  \mathbb E_{\theta_k}\!\left[
    \log P_\theta(x_i,Z)\mid X=x_i
  \right].
```

The expectation uses $\theta_k$; the log joint density inside it uses the
candidate parameter $\theta$. During the **M-step**, keep the posterior fixed
and choose a maximizer of $Q(\theta\mid\theta_k)$, assuming one exists in the
specified parameter set. The posterior entropies are constant in this
optimization, so maximizing $Q$ also maximizes $\mathcal F(q_k,\theta)$.

Equation [](#ig-em-lower-bound) now gives

```{math}
:label: ig-em-monotonicity
\ell_n(\theta_{k+1})
\ge\mathcal F(q_k,\theta_{k+1})
\ge\mathcal F(q_k,\theta_k)
=\ell_n(\theta_k).
```

Increasing $Q$ instead of maximizing it is enough for this inequality; that is
a generalized EM step. Likelihood monotonicity does not establish a global
maximum, convergence of the parameter sequence, or a convergence rate.

The E-step is also not, in general, a substitution $Z\leftarrow\mathbb E[Z\mid X]$.
For a nonlinear term, such as $Z^2$ or $\log Z$, its posterior expectation must
be computed separately. In particular,
$\mathbb E[Z^2\mid X]=\operatorname{Var}(Z\mid X)+\mathbb E[Z\mid X]^2$
when the second moment exists.

## Example: one unknown mixing weight

Let $Z\in\{0,1\}$ have probability $\Pr(Z=1)=\pi\in(0,1)$. Given $Z=1$,
$X$ has known density $f_1$; given $Z=0$, it has known density $f_0$. Assume
both are positive and finite at each $x_i$. Then

```{math}
\begin{aligned}
P_\pi(x,z)&=[\pi f_1(x)]^z[(1-\pi)f_0(x)]^{1-z},\\
p_\pi(x)&=\pi f_1(x)+(1-\pi)f_0(x).
\end{aligned}
```

The E-step computes the posterior probabilities, often called responsibilities:

```{math}
\tau_{k,i}=\Pr_{\pi_k}(Z=1\mid X=x_i)
=\frac{\pi_k f_1(x_i)}{\pi_k f_1(x_i)+(1-\pi_k)f_0(x_i)}.
```

With these values fixed, let $\overline\tau_k=n^{-1}\sum_i\tau_{k,i}$. Then

```{math}
\begin{aligned}
Q(\pi\mid\pi_k)
&=\overline\tau_k\log\pi\\
&\quad +(1-\overline\tau_k)\log(1-\pi)+\text{constant}.
\end{aligned}
```

Its derivative vanishes at $\pi=\overline\tau_k$, and its second derivative
is negative. Hence the exact update is

```{math}
:label: ig-em-weight-update
\pi_{k+1}=\overline\tau_k.
```

Here a posterior mean suffices because the complete log likelihood depends
affinely on the binary variable $z$. For a concrete step, suppose $\pi_k=1/2$
and the likelihood ratios $f_1(x_i)/f_0(x_i)$ at two observations are $3$ and
$1/2$. The responsibilities are $3/4$ and $1/3$, giving $\pi_{k+1}=13/24$.

If $f_0=f_1$ everywhere, the marginal law does not depend on $\pi$.
Then $\tau_{k,i}=\pi_k$ and every update stays in place. A well-defined EM
formula alone does not establish identifiability.

## Exponential-family completion

Suppose the joint is a full regular minimal exponential family in natural
coordinates:

```{math}
P_\theta(x,z)=h(x,z)
\exp\{\theta^\mathsf{T}t(x,z)-\psi(\theta)\}.
```

Here $t(x,z)$ is the complete-data sufficient statistic, with model expectation
coordinate $\eta(\theta)=\mathbb E_\theta[t(X,Z)]=\nabla\psi(\theta)$.
The E-step averages the entire statistic vector:

```{math}
\widehat\eta_k=\frac1n\sum_{i=1}^n
\mathbb E_{\theta_k}[t(x_i,Z)\mid X=x_i].
```

Then $Q(\theta\mid\theta_k)=\theta^\mathsf{T}\widehat\eta_k-\psi(\theta)
+\text{constant}$. If an interior unconstrained M-step exists, the
[moment-matching equation](./information-geometry-exponential-families.md#ig-moment-matching)
becomes

```{math}
\eta(\theta_{k+1})=\nabla\psi(\theta_{k+1})=\widehat\eta_k.
```

Constraints can change this equation, and the posterior moments need not lie
in the attainable interior. These are optimization and model-domain questions,
not consequences of the EM inequality.

In the [GH EM specialization](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/),
the complete data are $(X,Y)$: the positive mixing variable $Y$ takes the
latent role of $Z$ here, while $Z$ in that model denotes Gaussian noise.

The derivation used actual observations throughout. For continuous data, the
empirical distribution is atomic and its KL to a continuous model is generally
infinite. Ordinary EM requires no such KL. A later population projection
construction must specify a suitable observable density and its support.

Continue with [conditional expectation as projection](./information-geometry-conditional-expectation.md).
The [duality note](./information-geometry-duality.md) introduces a different,
KL-based projection, and the [hub](./ig/index.md) places both in
the reading path.
