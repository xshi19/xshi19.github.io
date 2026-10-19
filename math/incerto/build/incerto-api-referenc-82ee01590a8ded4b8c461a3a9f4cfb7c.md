---
title: Python API
---

The installed distribution is `xshi-math`; these helpers are imported as
`xmath.incerto`. Install the numerical and plotting extra with:

```sh
uv sync --extra incerto
```

The base `xmath` helpers require only NumPy. The Incerto extra adds SciPy and
Matplotlib; graph generation uses NetworkX and PyYAML from the development
group. Dataset fetching uses standard-library `urllib`. The Python modules
contain the full signatures and docstrings; there is no separate Sphinx build.
Browse the [module sources](https://github.com/xshi19/xshi-math/tree/main/src/math/incerto)
for implementations and use local `help()` for the installed version.

| Module | Entry points and purpose |
| --- | --- |
| `xmath.incerto.datasets` | `load_amazon_bookrank`, `load_nyc_dog_name_frequencies`, `load_sp500_prices`; explicit local-cache and refresh helpers |
| `xmath.incerto.distributions` | `pareto_type1`, `double_pareto`, `hypsecant2`, `simple_normal_mixture`, `univariate_variance_gamma`; SciPy-style distribution objects |
| `xmath.incerto.estimators` | `hill_estimator` (extreme-value index), `hill_alpha_estimator` (reciprocal exponent), `hill_stability`, `pareto_mean_plugin`, `mean_excess`, `running_mean`, `max_to_sum_ratio` |
| `xmath.incerto.stats` | Gaussian-calibrated `dispersion_ratios`, `mean_absolute_deviation`, `median_absolute_deviation`; normal and Student-t shoulder boundaries; `empirical_survival_function` (strict exceedance with ties) and `lognorm_params_from_mean_and_variance` |
| `xmath.incerto.tail_diagnostics` | `regular_variation_ratio`, `iid_two_sum_tail`, `subexponential_sum_ratio`; numerical calculations with support and integration checks |
| `xmath.incerto.tail_classes` | `bounded_spike_summary` (population moments), `log_survival_multiplier_ratio` (log-domain tail ratios); used by the [Prologue models](incerto-scoft-ch1.md) |
| `xmath.incerto.glossary` | `pareto_tail_quantities` (log survival, conditional means, tail contribution), `pareto_normalized_maximum_cdf` (exact iid maximum law); [chapter 2 worked cases](incerto-scoft-ch2.md#scoft-ch2-demo) |
| `xmath.incerto.lecture_risk` | `repeated_event_probability`, `survival_from_hazards`, `finite_payoff_mean`, `absorbing_resource_path`, `split_power_damage_logsf`, `block_failure_summary`; [chapter 3 worked models](incerto-scoft-ch3.md#scoft-ch3-demo) |
| `xmath.incerto.finite_moments` | `normal_mixture_log_density`, `normal_mixture_log_abs_survival`, `normal_mixture_absolute_moment`, `pareto_moment_estimator_summary`, `power_mean`; [finite-moment worked cases](incerto-scoft-ch4.md#worked-cases) |
| `xmath.incerto.figures` | `set_theme`, `style_axes`, `savefig`; shared figure style and deterministic export helpers |
| `xmath.incerto.power_laws` | `pareto_product_logsf`, `pareto_truncated_moment`, `student_double_pareto_scales`, `q_exponential_logsf`, `entropy_log_constraint_example`; [chapter 5 worked cases](incerto-scoft-ch5.md#scoft-ch5-demo) |

```python
import numpy as np
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.estimators import hill_alpha_estimator

rng = np.random.default_rng(20261017)
x = pareto_type1.rvs(2.5, size=1000, random_state=rng)
alpha_hat = hill_alpha_estimator(x, k=100)
```

The example is a simulation, not a guarantee about estimation accuracy.
The [Hill note](incerto-hill-estimator.md) states the model and threshold
assumptions. Use `help()` on an imported function for argument validation,
return types, and numerical limits.

The [empirical examples](incerto-empirical-example-concepts.md) link to committed
demo scripts and static figures. Dataset loading does not fetch by default.
The [S&P method](incerto-sp500-tail.md) explains the deliberately absent financial
cache. The separate [Normix API](https://xshi19.github.io/normix/) remains
maintained upstream; this educational package does not vendor Normix.

## Source and license

Adapted from `xshi19/incerto-wiki`, `content/api-reference.md`,
at revision `9717c9cf83aee18b47f423a14bce2f0d14bb8bcb`.
The original author-controlled material is used under MIT; the cited books
and external datasets retain their own rights. Calculations are static and
are not executed by the site build.

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
