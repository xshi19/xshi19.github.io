# Lean Blueprint

Lean is a quality track for selected mathematical statements, not a gate on
ordinary content work. The Phase 9 MVP checks two algebraic facts in a
contained [Lake/Mathlib project](https://github.com/xshi19/incerto-wiki/tree/main/formalization/lean).
The remaining seed statements below are blueprint-only.

## Scope

The checked results concern real-valued expressions. They do not establish
convergence of a moment integral or construct a probability measure.
Pickands-Balkema-de Haan, asymptotic empirical processes, and general
subexponential theory remain deferred.

The blueprint accepts a statement when it has all of the following:

- a stable notation choice in the wiki;
- a concept page that already states and motivates the result;
- a proof short enough to audit by hand;
- a likely home in Mathlib without large new infrastructure.

## Checked Statements

The declarations are in
[`Incerto/Pareto.lean`](https://github.com/xshi19/incerto-wiki/blob/main/formalization/lean/Incerto/Pareto.lean),
imported by the default `Incerto` library target. They were checked with
Lean and Mathlib `v4.19.0` on 2026-09-05:

```bash
cd formalization/lean
lake exe cache get  # first-time dependency cache
lake build
```

`Incerto.moment_exponent_threshold` proves the exponent comparison for
all real $p$ and $\alpha$. The improper-integral convergence criterion
used in the moment proof is a separate, unformalized step.

`Incerto.pareto_survival_scale` uses the piecewise survival expression
from the [Pareto Distribution](../concepts/distributions/pareto.md).
It requires $x_m>0$, $\alpha>0$, $t>0$, $x\ge x_m$, and $tx\ge x_m$.
When $0<t<1$, the last condition ensures the scaled threshold remains in
the power-law branch; it cannot be dropped. Equality at either cutoff is
included.

## Seed Statements

The notation used below follows the shared [notation table](../notation/index.md)
where applicable.

| ID | Statement | Site page | Formalization status |
| -- | --------- | --------- | -------------------- |
| `pareto_survival_scale` | For $x \ge x_m>0$, $t>0$, $tx\ge x_m$, and $\alpha>0$, the Pareto survival ratio satisfies $\bar F(tx)/\bar F(x)=t^{-\alpha}$. | [Pareto Distribution](../concepts/distributions/pareto.md) | Checked: `Incerto.pareto_survival_scale` (real algebra) |
| `pareto_moment_integrand` | The raw moment integral for Pareto reduces to a power integral, $\alpha x_m^\alpha\int_{x_m}^{\infty}x^{p-\alpha-1}\,dx$. | [Pareto Moment Existence](../concepts/theorems/pareto-moment-existence.md) | Blueprint-only |
| `moment_exponent_threshold` | For real $p,\alpha$, the exponent condition $p-\alpha-1<-1$ is equivalent to $p<\alpha$. | [Pareto Moment Existence](../concepts/theorems/pareto-moment-existence.md) | Checked: `Incerto.moment_exponent_threshold` (algebra only) |
| `regular_variation_power` | The pure power survival function $x \mapsto x^{-\alpha}$ is [regularly varying](../concepts/theorems/regular-variation.md) with index $-\alpha$. | [Regular Variation](../concepts/theorems/regular-variation.md) | Blueprint-only |

These are deliberately modest. They connect the wiki's notation to exact
algebraic facts that recur across concept pages.

## Lean Conventions

- Use theorem names that match the wiki identifier when possible.
- Keep assumptions explicit: positivity, lower cutoffs, and threshold domains
  should not be hidden inside prose.
- Prefer real-valued statements before measure-theoretic probability
  statements.
- A page may link to Lean only after the Lean statement is checked in a Lean
  project or clearly marked as a blueprint item.

## Deferred

The following remain outside the Lean MVP:

- full proofs of Karamata's theorem;
- Pickands-Balkema-de Haan convergence;
- Hill estimator consistency;
- empirical-process or statistical-estimation theorems;
- automatic extraction of Lean statements from concept pages.

The right order is the patient one: small algebra, then integral facts, then
probability statements, then asymptotics.

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/formalization/lean-blueprint.md`. Last verified: 2026-09-05. The two named declarations passed `lake build`; other seed statements remain blueprint-only.
:::
<!-- incerto-provenance:end -->
