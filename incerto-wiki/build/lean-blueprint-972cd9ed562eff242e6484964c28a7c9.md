# Lean Blueprint

Lean is a quality track for selected mathematical statements, not a gate on
ordinary content work.  This page records the Phase 4 blueprint: start with
small facts that stabilize notation and proof obligations before attempting
anything from extreme value theory proper.

## Scope

The first Lean work should be local, elementary, and useful to nearby pages.
Do not begin with Pickands-Balkema-de Haan, asymptotic empirical processes, or
general subexponential theory.  Those topics need substantial supporting
libraries and would turn formalization into a blocker.

The blueprint accepts a statement when it has all of the following:

- a stable notation choice in the wiki;
- a concept page that already states and motivates the result;
- a proof short enough to audit by hand;
- a likely home in Mathlib without large new infrastructure.

## Seed Statements

| ID | Statement | Site page | Formalization status |
| -- | --------- | --------- | -------------------- |
| `pareto_survival_scale` | For $x \ge x_m>0$, $t>0$, and $\alpha>0$, the Pareto survival ratio satisfies $\bar F(tx)/\bar F(x)=t^{-\alpha}$. | [Pareto Distribution](../concepts/distributions/pareto.md) | Blueprinted |
| `pareto_moment_integrand` | The raw moment integral for Pareto reduces to a power integral, $\alpha x_m^\alpha\int_{x_m}^{\infty}x^{p-\alpha-1}\,dx$. | [Pareto Moment Existence](../concepts/theorems/pareto-moment-existence.md) | Blueprinted |
| `moment_exponent_threshold` | The exponent condition $p-\alpha-1<-1$ is equivalent to $p<\alpha$. | [Pareto Moment Existence](../concepts/theorems/pareto-moment-existence.md) | Blueprinted |
| `regular_variation_power` | The pure power survival function $x \mapsto x^{-\alpha}$ is regularly varying with index $-\alpha$. | [Regular Variation](../concepts/theorems/regular-variation.md) | Blueprinted |

These are deliberately modest.  They are useful because they connect the wiki's
notation to exact algebraic facts that recur across concept pages.

## Lean Conventions

- Use theorem names that match the wiki identifier when possible.
- Keep assumptions explicit: positivity, lower cutoffs, and threshold domains
  should not be hidden inside prose.
- Prefer real-valued statements before measure-theoretic probability
  statements.
- A page may link to Lean only after the Lean statement is checked in a Lean
  project or clearly marked as a blueprint item.

## Deferred

The following are not Phase 4 targets:

- full proofs of Karamata's theorem;
- Pickands-Balkema-de Haan convergence;
- Hill estimator consistency;
- empirical-process or statistical-estimation theorems;
- automatic extraction of Lean statements from concept pages.

The right order is the patient one: small algebra, then integral facts, then
probability statements, then asymptotics.

<!-- incerto-provenance:start -->
:::{div} incerto-provenance
**Provenance.** Source: `content/formalization/lean-blueprint.md`. Last verified: 2026-05-24. Scope checked as a Lean-blueprint item; Lean proofs are not implied.
:::
<!-- incerto-provenance:end -->
