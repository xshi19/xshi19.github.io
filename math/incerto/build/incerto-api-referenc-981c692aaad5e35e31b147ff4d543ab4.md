---
title: Python API
---

The installed distribution is `xshi-math`; these helpers are imported as
`xmath.incerto`. Install the numerical and plotting extra with:

```sh
uv sync --locked --extra incerto
```

Run page examples in Python after installing from a repository checkout.
Every fenced `python` block on the Incerto pages is executed by pytest;
examples do not fetch data or save files. Each **Output** block is captured
from its preceding example and checked against stdout, ignoring trailing
whitespace only. Bundled dataset examples require an
editable checkout (or a populated `INCERTO_DATA_DIR`). Longer examples expand
under a **Code** dropdown.

Numeric output uses about four significant digits or three to four decimal
places, with scientific notation for small probabilities. Counts remain
integers; vectors print as short lists and dictionaries as `key: value` lines.
Mathematical infinities and missing values print as `inf` and `nan`.

The base `xmath` helpers require only NumPy. The Incerto extra adds SciPy and
Matplotlib; graph generation uses NetworkX and PyYAML from the development
group. Dataset fetching uses standard-library `urllib`. The Python modules
contain the full signatures and docstrings; there is no separate Sphinx build.
Browse the [module sources](https://github.com/xshi19/xshi-math/tree/main/src/math/incerto)
for implementations and use local `help()` for the installed version.

| Module | Entry points and purpose |
| --- | --- |
| `xmath.incerto.concentration` | [`kappa_from_mads`](incerto-api-reference.md#api-kappa-from-mads), [`stable_kappa`](incerto-api-reference.md#api-stable-kappa), [`gaussian_sum_mad`](incerto-api-reference.md#api-gaussian-sum-mad), [`gamma_sum_mad`](incerto-api-reference.md#api-gamma-sum-mad), [`student_sum_mad`](incerto-api-reference.md#api-student-sum-mad), [`estimate_sum_mad`](incerto-api-reference.md#api-estimate-sum-mad), [`lognormal_sum_mad`](incerto-api-reference.md#api-lognormal-sum-mad), [`normal_location_mixture_sum_mad`](incerto-api-reference.md#api-normal-location-mixture-sum-mad), [`gaussian_equivalent_n`](incerto-api-reference.md#api-gaussian-equivalent-n); [finite-sample concentration](incerto-kappa.md) and [sample sufficiency](incerto-moment-estimation.md#moment-estimation-sufficiency) |
| `xmath.incerto.gas_speeds` | [`chi_speed`](incerto-api-reference.md#api-chi-speed), [`sample_chi_speed`](incerto-api-reference.md#api-sample-chi-speed), [`fit_speed_models`](incerto-api-reference.md#api-fit-speed-models), [`speed_fit_table`](incerto-api-reference.md#api-speed-fit-table), [`speed_qq`](incerto-api-reference.md#api-speed-qq), [`squared_speed_ks`](incerto-api-reference.md#api-squared-speed-ks), [`speed_fit_figure`](incerto-api-reference.md#api-speed-fit-figure); [Rayleigh chamber speeds](incerto-rayleigh-speed.md) and fitted comparisons |
| `xmath.incerto.datasets` | [`load_amazon_bookrank`](incerto-api-reference.md#api-load-amazon-bookrank), [`load_nyc_dog_name_frequencies`](incerto-api-reference.md#api-load-nyc-dog-name-frequencies), [`load_sp500_prices`](incerto-api-reference.md#api-load-sp500-prices); explicit local-cache and refresh helpers |
| `xmath.incerto.distributions` | [`pareto_type1`](incerto-api-reference.md#api-pareto-type1), [`double_pareto`](incerto-api-reference.md#api-double-pareto), [`hypsecant2`](incerto-api-reference.md#api-hypsecant2), [`simple_normal_mixture`](incerto-api-reference.md#api-simple-normal-mixture), [`univariate_variance_gamma`](incerto-api-reference.md#api-univariate-variance-gamma), [`generalized_pareto`](incerto-api-reference.md#api-generalized-pareto), [`generalized_extreme_value`](incerto-api-reference.md#api-generalized-extreme-value), [`frechet`](incerto-api-reference.md#api-frechet); SciPy-style distribution objects |
| `xmath.incerto.estimators` | [`hill_estimator`](incerto-api-reference.md#api-hill-estimator) (extreme-value index), [`hill_alpha_estimator`](incerto-api-reference.md#api-hill-alpha-estimator) (reciprocal exponent), [`hill_stability`](incerto-api-reference.md#api-hill-stability), [`pareto_mean_plugin`](incerto-api-reference.md#api-pareto-mean-plugin), [`mean_excess`](incerto-api-reference.md#api-mean-excess), [`running_mean`](incerto-api-reference.md#api-running-mean), [`max_to_sum_ratio`](incerto-api-reference.md#api-max-to-sum-ratio), [`max_to_sum_path`](incerto-api-reference.md#api-max-to-sum-path), [`gpd_threshold_fit`](incerto-api-reference.md#api-gpd-threshold-fit) |
| `xmath.incerto.stats` | Gaussian-calibrated [`dispersion_ratios`](incerto-api-reference.md#api-dispersion-ratios), [`mean_absolute_deviation`](incerto-api-reference.md#api-mean-absolute-deviation), [`median_absolute_deviation`](incerto-api-reference.md#api-median-absolute-deviation); normal and Student-t shoulder boundaries; [`empirical_survival_function`](incerto-api-reference.md#api-empirical-survival-function) (strict exceedance with ties) and [`lognorm_params_from_mean_and_variance`](incerto-api-reference.md#api-lognorm-params-from-mean-and-variance) |
| `xmath.incerto.tail_diagnostics` | [`regular_variation_ratio`](incerto-api-reference.md#api-regular-variation-ratio), [`iid_two_sum_tail`](incerto-api-reference.md#api-iid-two-sum-tail), [`subexponential_sum_ratio`](incerto-api-reference.md#api-subexponential-sum-ratio); numerical calculations with support and integration checks |
| `xmath.incerto.tail_classes` | [`bounded_spike_summary`](incerto-api-reference.md#api-bounded-spike-summary) (population moments), [`log_survival_multiplier_ratio`](incerto-api-reference.md#api-log-survival-multiplier-ratio) (log-domain tail ratios); used by the [Prologue models](incerto-scoft-ch1.md) |
| `xmath.incerto.glossary` | [`pareto_tail_quantities`](incerto-api-reference.md#api-pareto-tail-quantities) (log survival, conditional means, tail contribution), [`pareto_normalized_maximum_cdf`](incerto-api-reference.md#api-pareto-normalized-maximum-cdf) (exact iid maximum law); [chapter 2 worked cases](incerto-scoft-ch2.md#scoft-ch2-demo) |
| `xmath.incerto.lecture_risk` | [`repeated_event_probability`](incerto-api-reference.md#api-repeated-event-probability), [`survival_from_hazards`](incerto-api-reference.md#api-survival-from-hazards), [`finite_payoff_mean`](incerto-api-reference.md#api-finite-payoff-mean), [`absorbing_resource_path`](incerto-api-reference.md#api-absorbing-resource-path), [`split_power_damage_logsf`](incerto-api-reference.md#api-split-power-damage-logsf), [`block_failure_summary`](incerto-api-reference.md#api-block-failure-summary); [chapter 3 worked models](incerto-scoft-ch3.md#scoft-ch3-demo) |
| `xmath.incerto.finite_moments` | [`normal_mixture_log_density`](incerto-api-reference.md#api-normal-mixture-log-density), [`normal_mixture_log_abs_survival`](incerto-api-reference.md#api-normal-mixture-log-abs-survival), [`normal_mixture_absolute_moment`](incerto-api-reference.md#api-normal-mixture-absolute-moment), [`pareto_moment_estimator_summary`](incerto-api-reference.md#api-pareto-moment-estimator-summary), [`power_mean`](incerto-api-reference.md#api-power-mean); [finite-moment worked cases](incerto-scoft-ch4.md#worked-cases) |
| `xmath.incerto.multivariate` | [`student_entropy`](incerto-api-reference.md#api-student-entropy), [`student_mutual_information`](incerto-api-reference.md#api-student-mutual-information), [`bivariate_student`](incerto-api-reference.md#api-bivariate-student), [`sample_correlation`](incerto-api-reference.md#api-sample-correlation), [`symmetric_spectrum`](incerto-api-reference.md#api-symmetric-spectrum), [`regression_population_r2`](incerto-api-reference.md#api-regression-population-r2), [`normal_location_mixture_kurtosis`](incerto-api-reference.md#api-normal-location-mixture-kurtosis); [chapter 6 worked cases](incerto-scoft-ch6.md#scoft-ch6-demo). Student helpers deliberately restrict degrees of freedom to [0.5, 1000]. |
| `xmath.incerto.limits` | [`pareto_capped_mean`](incerto-api-reference.md#api-pareto-capped-mean), [`symmetric_stable_mean_abs`](incerto-api-reference.md#api-symmetric-stable-mean-abs), [`repeated_centered_sums`](incerto-api-reference.md#api-repeated-centered-sums), [`pareto_plugin_nonfinite_probability`](incerto-api-reference.md#api-pareto-plugin-nonfinite-probability), [`bernoulli_chernoff_log_bound`](incerto-api-reference.md#api-bernoulli-chernoff-log-bound); [chapter 7 averaging and limit laws](incerto-scoft-ch7.md) |
| `xmath.incerto.figures` | [`set_theme`](incerto-api-reference.md#api-set-theme), [`style_axes`](incerto-api-reference.md#api-style-axes), [`savefig`](incerto-api-reference.md#api-savefig), [`vertical_figure`](incerto-api-reference.md#api-vertical-figure), [`distribution_card`](incerto-api-reference.md#api-distribution-card); shared figure style and deterministic export helpers |
| `xmath.incerto.powerlaw` | [`powerlaw_mle_continuous`](incerto-api-reference.md#api-powerlaw-mle-continuous), [`powerlaw_mle_discrete`](incerto-api-reference.md#api-powerlaw-mle-discrete), [`ks_distance_powerlaw`](incerto-api-reference.md#api-ks-distance-powerlaw), [`select_xmin`](incerto-api-reference.md#api-select-xmin), [`powerlaw_standard_error`](incerto-api-reference.md#api-powerlaw-standard-error), [`sample_powerlaw`](incerto-api-reference.md#api-sample-powerlaw), [`powerlaw_bootstrap_se`](incerto-api-reference.md#api-powerlaw-bootstrap-se), [`powerlaw_gof_bootstrap`](incerto-api-reference.md#api-powerlaw-gof-bootstrap), [`fit_tail_model`](incerto-api-reference.md#api-fit-tail-model), [`likelihood_ratio_test`](incerto-api-reference.md#api-likelihood-ratio-test); power-tail fitting, bootstrap checks, and conditional model comparison |
| `xmath.incerto.power_laws` | [`pareto_product_logsf`](incerto-api-reference.md#api-pareto-product-logsf), [`pareto_truncated_moment`](incerto-api-reference.md#api-pareto-truncated-moment), [`student_double_pareto_scales`](incerto-api-reference.md#api-student-double-pareto-scales), [`q_exponential_logsf`](incerto-api-reference.md#api-q-exponential-logsf), [`entropy_log_constraint_example`](incerto-api-reference.md#api-entropy-log-constraint-example); [chapter 5 worked cases](incerto-scoft-ch5.md#scoft-ch5-demo) |

Use [`pareto_type1`](incerto-api-reference.md#api-pareto-type1) and
[`hill_alpha_estimator`](incerto-api-reference.md#api-hill-alpha-estimator).

```python
import numpy as np
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.estimators import hill_alpha_estimator

rng = np.random.default_rng(20261017)
x = pareto_type1.rvs(2.5, size=1000, random_state=rng)
alpha_hat = hill_alpha_estimator(x, k=100)
print(f"Estimated alpha: {alpha_hat:.3f}")
```

**Output**

```text
Estimated alpha: 2.405
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

## Function reference

Signatures omit type annotations for readability; keyword-only markers and
defaults match the installed helpers. See local `help()` for validation and
return details. Distribution objects use SciPy’s frozen-law interface.

### Datasets

Module: `xmath.incerto.datasets`.

(api-datasetnotavailable)=
#### [`DatasetNotAvailable`](incerto-api-reference.md#api-datasetnotavailable)

Raised when a dataset is missing or a fetch cannot provide usable data.

(api-priceseries)=
#### [`PriceSeries`](incerto-api-reference.md#api-priceseries)

Daily close-price series used by empirical examples.

(api-priceseries-log-returns)=
#### [`PriceSeries.log_returns`](incerto-api-reference.md#api-priceseries-log-returns)

`PriceSeries.log_returns()`

```python
import numpy as np
from xmath.incerto.datasets import PriceSeries

prices = PriceSeries(
    name="synthetic", date=np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
    close=np.array([100., 90., 99.]), source="synthetic example",
)
dates, values = prices.log_returns()
print("Ending dates: [" + ", ".join(str(day) for day in dates) + "]")
print("Log returns: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
Ending dates: [2020-01-02, 2020-01-03]
Log returns: [-0.1054, 0.09531]
```


(api-priceseries-negative-log-returns)=
#### [`PriceSeries.negative_log_returns`](incerto-api-reference.md#api-priceseries-negative-log-returns)

`PriceSeries.negative_log_returns()`

```python
import numpy as np
from xmath.incerto.datasets import PriceSeries

prices = PriceSeries(
    name="synthetic", date=np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
    close=np.array([100., 90., 99.]), source="synthetic example",
)
dates, values = prices.negative_log_returns()
print("Ending dates: [" + ", ".join(str(day) for day in dates) + "]")
print("Positive losses: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
Ending dates: [2020-01-02]
Positive losses: [0.1054]
```


(api-bookrankdata)=
#### [`BookRankData`](incerto-api-reference.md#api-bookrankdata)

Amazon book sales-rank observations.

(api-dognamefrequencydata)=
#### [`DogNameFrequencyData`](incerto-api-reference.md#api-dognamefrequencydata)

Normalized NYC dog-name frequencies from license-period records.

(api-dognamefrequencydata-frequency-for)=
#### [`DogNameFrequencyData.frequency_for`](incerto-api-reference.md#api-dognamefrequencydata-frequency-for)

`DogNameFrequencyData.frequency_for(name)`

```python
import numpy as np
from xmath.incerto.datasets import DogNameFrequencyData

names = DogNameFrequencyData(np.array(["SAFARI"]), np.array([3]), "synthetic example")
for name in ("Safari", " safari ", "MISSING"):
    print(f"{name!r}: {names.frequency_for(name):d}")
```

**Output**

```text
'Safari': 3
' safari ': 3
'MISSING': 0
```


(api-data-root)=
#### [`data_root`](incerto-api-reference.md#api-data-root)

`data_root(data_dir=None)`

```python
from xmath.incerto.datasets import data_root

print(data_root(data_dir="/tmp/incerto-example-data").as_posix())
```

**Output**

```text
/tmp/incerto-example-data
```


(api-snapshot-path)=
#### [`snapshot_path`](incerto-api-reference.md#api-snapshot-path)

`snapshot_path(filename, data_dir=None)`

```python
from xmath.incerto.datasets import snapshot_path

print(snapshot_path("example.csv", data_dir="/tmp/incerto-example-data").as_posix())
```

**Output**

```text
/tmp/incerto-example-data/snapshots/example.csv
```


(api-cache-path)=
#### [`cache_path`](incerto-api-reference.md#api-cache-path)

`cache_path(filename, data_dir=None)`

```python
from xmath.incerto.datasets import cache_path

print(cache_path("example.csv", data_dir="/tmp/incerto-example-data").as_posix())
```

**Output**

```text
/tmp/incerto-example-data/cache/example.csv
```


(api-load-sp500-prices)=
#### [`load_sp500_prices`](incerto-api-reference.md#api-load-sp500-prices)

`load_sp500_prices(*, data_dir=None, fetch=False)`

Load S&P 500 daily closes from the local cache, optionally fetching FRED.

(api-load-amazon-bookrank)=
#### [`load_amazon_bookrank`](incerto-api-reference.md#api-load-amazon-bookrank)

`load_amazon_bookrank(*, data_dir=None)`

Load the CC0 Amazon book sales-rank snapshot.

(api-load-nyc-dog-name-frequencies)=
#### [`load_nyc_dog_name_frequencies`](incerto-api-reference.md#api-load-nyc-dog-name-frequencies)

`load_nyc_dog_name_frequencies(*, data_dir=None)`

Load the NYC Open Data dog-name frequency snapshot.

(api-write-sp500-fred-cache)=
#### [`write_sp500_fred_cache`](incerto-api-reference.md#api-write-sp500-fred-cache)

`write_sp500_fred_cache(destination, *, url=FRED_SP500_CSV_URL)`

Fetch and clean the public FRED S&P 500 CSV into an untracked cache.

(api-write-sp500-cache-from-csv)=
#### [`write_sp500_cache_from_csv`](incerto-api-reference.md#api-write-sp500-cache-from-csv)

`write_sp500_cache_from_csv(source, destination)`

Clean a local S&P 500-like CSV into the standard untracked cache.

(api-write-amazon-bookrank-snapshot)=
#### [`write_amazon_bookrank_snapshot`](incerto-api-reference.md#api-write-amazon-bookrank-snapshot)

`write_amazon_bookrank_snapshot(destination, *, url=KAGGLE_AMAZON_BOOKRANK_DOWNLOAD_URL)`

Fetch and clean the CC0 Kaggle Amazon bookrank dataset.

(api-write-nyc-dog-name-frequency-snapshot)=
#### [`write_nyc_dog_name_frequency_snapshot`](incerto-api-reference.md#api-write-nyc-dog-name-frequency-snapshot)

`write_nyc_dog_name_frequency_snapshot(destination, *, url=NYC_DOG_NAME_FREQUENCY_API_URL)`

Fetch the server-aggregated NYC dog-name frequencies into a snapshot.

### Distributions

Module: `xmath.incerto.distributions`.

Call an object below with its shape parameters to freeze a law; then use
`pdf(x)`, `logpdf(x)`, `cdf(x)`, `sf(x)`, `logsf(x)`, `ppf(q)`,
`rvs(size=..., random_state=rng)` or `stats(moments="mv")`.
These methods belong to the linked distribution entry. `scale` is a
positive scale multiplier, and `loc` shifts the support.

(api-pareto-type1)=
#### [`pareto_type1`](incerto-api-reference.md#api-pareto-type1)

`pareto_type1(alpha, loc=0, scale=1)`

Pareto Type I with positive exponent `alpha`; at `loc=0`, `scale` is the lower cutoff.

(api-double-pareto)=
#### [`double_pareto`](incerto-api-reference.md#api-double-pareto)

`double_pareto(alpha, loc=0, scale=1)`

Symmetric shifted double Pareto with positive exponent `alpha`; ordinary mean exists only above one.

(api-hypsecant2)=
#### [`hypsecant2`](incerto-api-reference.md#api-hypsecant2)

`hypsecant2(loc=0, scale=1)`

Density `0.5 / cosh(x)**2` at default location and scale (logistic scale one half).

(api-simple-normal-mixture)=
#### [`simple_normal_mixture`](incerto-api-reference.md#api-simple-normal-mixture)

`simple_normal_mixture(a, loc=0, scale=1)`

Equal centered normal mixture with component variances `1-a` and `1+a`, for `0 <= a < 1`.

(api-univariate-variance-gamma)=
#### [`univariate_variance_gamma`](incerto-api-reference.md#api-univariate-variance-gamma)

`univariate_variance_gamma(alpha, beta, lam, loc=0, scale=1)`

Variance–gamma law requiring `alpha > abs(beta)` and `lam > 0`; this `alpha` is a decay parameter, not a Pareto exponent.

(api-hypsecant2-chf)=
#### [`hypsecant2.chf`](incerto-api-reference.md#api-hypsecant2-chf)

`hypsecant2.chf(t, loc=0.0, scale=1.0)`

```python
from xmath.incerto.distributions import hypsecant2

values = hypsecant2.chf([0, 1, 2]).real
print("[" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
[1, 0.6826, 0.272]
```



(api-generalized-pareto)=
#### [`generalized_pareto`](incerto-api-reference.md#api-generalized-pareto)

`generalized_pareto(xi, beta=1.0)`

Returns a frozen zero-location GPD with finite scalar shape $\xi$ and
finite scale $\beta>0$. Methods include `pdf`, `cdf`, `sf`, `logsf`,
`ppf`, `rvs` and `support`. Support is $[0,\infty)$ for $\xi\ge0$ and
$[0,-\beta/\xi]$ for $\xi<0$. CDF/survival extend continuously at
support boundaries. Endpoint densities can be infinite when $\xi<-1$.
Invalid parameters raise `ValueError`.

```python
from xmath.incerto.distributions import generalized_pareto

law = generalized_pareto(-0.4)
print("Survival: [" + ", ".join(f"{v:.4g}" for v in law.sf([-1, 0, 1, 2.5, 3])) + "]")
```

**Output**

```text
Survival: [1, 1, 0.2789, 0, 0]
```

(api-generalized-extreme-value)=
#### [`generalized_extreme_value`](incerto-api-reference.md#api-generalized-extreme-value)

`generalized_extreme_value(xi, mu=0.0, sigma=1.0)`

Returns a frozen GEV with the [page's shape convention](incerto-generalized-extreme-value.md):
positive $\xi$ is heavy-tailed, zero is Gumbel, negative has a finite
right endpoint. The adapter uses `scipy.stats.genextreme(c=-xi)` and
exposes the same methods as the GPD adapter. Require finite scalar
parameters and $\sigma>0$. Outside support, density is zero and
CDF/survival have their limiting values. For $\xi<-1$, the density
diverges as the right endpoint is approached, although SciPy 1.18 returns
zero at the endpoint itself. This single-point convention does not change
probabilities. Invalid parameters raise `ValueError`.

```python
from xmath.incerto.distributions import generalized_extreme_value

for xi in (-0.3, 0, 0.3):
    law = generalized_extreme_value(xi)
    print(f"xi={xi:.4g}: cdf(0)={law.cdf(0):.4g}, sf(4)={law.sf(4):.4g}")
```

**Output**

```text
xi=-0.3: cdf(0)=0.3679, sf(4)=0
xi=0: cdf(0)=0.3679, sf(4)=0.01815
xi=0.3: cdf(0)=0.3679, sf(4)=0.06966
```

(api-frechet)=
#### [`frechet`](incerto-api-reference.md#api-frechet)

`frechet(alpha, scale=1.0)`

Returns a frozen zero-location Fréchet law with CDF
$\exp[-(y/\mathrm{scale})^{-\alpha}]$ for $y>0$. Both parameters are
finite positive scalars. Uses SciPy `invweibull`, including `pdf`, `cdf`,
`sf`, `logsf`, `ppf`, `rvs` and `support`. At and below zero,
density and CDF are zero, survival one. Use `sf` rather than subtracting
`cdf` from one when the right tail is small. Invalid parameters raise `ValueError`.
The adapter inherits SciPy's numerical limits: for example, SciPy 1.18
returns `nan` for `frechet(3).pdf(1e-100)` because intermediate powers
overflow, although the true density rounds to zero. The P1 figure grids
avoid this range.

```python
from xmath.incerto.distributions import frechet

law = frechet(1.5)
for y in (0, 1, 10):
    print(f"y={y:d}: pdf={law.pdf(y):.4g}, cdf={law.cdf(y):.4g}, sf={law.sf(y):.4g}")
```

**Output**

```text
y=0: pdf=0, cdf=0, sf=1
y=1: pdf=0.5518, cdf=0.3679, sf=0.6321
y=10: pdf=0.004596, cdf=0.9689, sf=0.03113
```

### Estimators

Module: `xmath.incerto.estimators`.

(api-hill-estimator)=
#### [`hill_estimator`](incerto-api-reference.md#api-hill-estimator)

`hill_estimator(x, k)`

```python
from xmath.incerto.estimators import hill_estimator

print(f"xi: {hill_estimator([1, 2, 4, 8], k=2):.4g}")
```

**Output**

```text
xi: 1.04
```


(api-hill-alpha-estimator)=
#### [`hill_alpha_estimator`](incerto-api-reference.md#api-hill-alpha-estimator)

`hill_alpha_estimator(x, k)`

```python
from xmath.incerto.estimators import hill_alpha_estimator

for sample in ([1, 2, 4, 8], [1, 1, 1, 1]):
    print(sample, "alpha:", f"{hill_alpha_estimator(sample, k=2):.4g}")
```

**Output**

```text
[1, 2, 4, 8] alpha: 0.9618
[1, 1, 1, 1] alpha: inf
```


(api-pareto-mean-plugin)=
#### [`pareto_mean_plugin`](incerto-api-reference.md#api-pareto-mean-plugin)

`pareto_mean_plugin(x, *, x_m, axis=None)`

```python
from xmath.incerto.estimators import pareto_mean_plugin

for sample in ([1, 1], [1, 2], [4, 4]):
    print(f"{sample} {pareto_mean_plugin(sample, x_m=1):.4g}")
```

**Output**

```text
[1, 1] 1
[1, 2] 1.53
[4, 4] inf
```

The cutoff is known; statistical calibration assumes an exact iid Pareto model.


(api-hill-stability)=
#### [`hill_stability`](incerto-api-reference.md#api-hill-stability)

`hill_stability(x, k_values=None, *, num=100)`

```python
from xmath.incerto.estimators import hill_stability

for name, values in hill_stability([1, 2, 4, 8], [1, 2]).items():
    fmt = ".0f" if name == "k" else ".4g"
    print(f"{name}: [" + ", ".join(format(v, fmt) for v in values) + "]")
```

**Output**

```text
k: [1, 2]
xi: [0.6931, 1.04]
alpha: [1.443, 0.9618]
```


(api-running-mean)=
#### [`running_mean`](incerto-api-reference.md#api-running-mean)

`running_mean(x, axis=None)`

```python
from xmath.incerto.estimators import running_mean

values = running_mean([1, 2, 6])
print("Vector: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
print("Rows:")
for row in running_mean([[1, 2, 6], [2, 4, 12]], axis=1):
    print("[" + ", ".join(f"{v:.4g}" for v in row) + "]")
```

**Output**

```text
Vector: [1, 1.5, 3]
Rows:
[1, 1.5, 3]
[2, 3, 6]
```


(api-max-to-sum-ratio)=
#### [`max_to_sum_ratio`](incerto-api-reference.md#api-max-to-sum-ratio)

`max_to_sum_ratio(x, axis=None)`

```python
from xmath.incerto.estimators import max_to_sum_ratio

print(f"{max_to_sum_ratio([1, 2, 7]):.4g}")
```

**Output**

```text
0.7
```



(api-max-to-sum-path)=
#### [`max_to_sum_path`](incerto-api-reference.md#api-max-to-sum-path)

`max_to_sum_path(x)`

Returns the maximum divided by the sum for every prefix of a nonempty,
finite, real, nonnegative one-dimensional list. Zero-total prefixes return
`nan`. Log accumulation avoids overflow of the sum and avoids scaling early
prefixes by a much larger future observation. Negative/nonfinite/complex
values and wrong dimensions raise `ValueError`; ordinary rounding remains.

```python
from xmath.incerto.estimators import max_to_sum_path

path = max_to_sum_path([0, 1, 1, 8])
print("Prefix ratios: [" + ", ".join(f"{v:.4g}" for v in path) + "]")
```

**Output**

```text
Prefix ratios: [nan, 1, 0.5, 0.8]
```

(api-gpd-threshold-fit)=
#### [`gpd_threshold_fit`](incerto-api-reference.md#api-gpd-threshold-fit)

`gpd_threshold_fit(x, threshold, *, min_exceedances=30)`

Numerically fits a zero-location GPD to strict excesses above a finite
threshold. Returns `threshold`, integer `count`, `xi`, `beta`,
`modified_scale` ($\widehat\beta-\widehat\xi u$), `success` and `status`.
Finite real one-dimensional data are required; invalid input raises.
Insufficient or constant excesses, overflow when forming excesses, and failed
numerical fits retain counts but report `nan` parameters and an explanatory
status. The optimizer stops
at 2,000 iterations or 4,000 evaluations. Checks cover numerical convergence,
finite parameters/log density, positive scale and every excess's support.
Success is no guarantee of a global MLE, a valid tail model or an appropriate
threshold. No confidence interval is returned.

```python
from xmath.incerto.estimators import gpd_threshold_fit

fit = gpd_threshold_fit([1, 2, 2, 4], 2, min_exceedances=2)
print(f"Count: {fit['count']:d}; status: {fit['status']}")
print(f"xi: {fit['xi']:.4g}; beta: {fit['beta']:.4g}")
```

**Output**

```text
Count: 1; status: insufficient count
xi: nan; beta: nan
```

(api-mean-excess)=
#### [`mean_excess`](incerto-api-reference.md#api-mean-excess)

`mean_excess(x, thresholds, *, min_exceedances=1)`

```python
from xmath.incerto.estimators import mean_excess

for name, values in mean_excess([1, 2, 2, 4], [1, 2, 4]).items():
    fmt = ".0f" if name == "exceedances" else ".4g"
    print(f"{name}: [" + ", ".join(format(v, fmt) for v in values) + "]")
```

**Output**

```text
threshold: [1, 2, 4]
mean_excess: [1.667, 2, nan]
exceedances: [3, 1, 0]
```


### Stats

Module: `xmath.incerto.stats`.

(api-mean-absolute-deviation)=
#### [`mean_absolute_deviation`](incerto-api-reference.md#api-mean-absolute-deviation)

`mean_absolute_deviation(x, axis=None)`

```python
from xmath.incerto.stats import mean_absolute_deviation

print(f"Gaussian-calibrated MeanAD: {mean_absolute_deviation([1, 2, 3]):.4g}")
```

**Output**

```text
Gaussian-calibrated MeanAD: 0.8355
```


(api-median-absolute-deviation)=
#### [`median_absolute_deviation`](incerto-api-reference.md#api-median-absolute-deviation)

`median_absolute_deviation(x, axis=None)`

```python
from xmath.incerto.stats import median_absolute_deviation

print(f"Gaussian-calibrated MedianAD: {median_absolute_deviation([1, 2, 3]):.4g}")
```

**Output**

```text
Gaussian-calibrated MedianAD: 1.483
```


(api-normal-peak-shoulder-tail)=
#### [`normal_peak_shoulder_tail`](incerto-api-reference.md#api-normal-peak-shoulder-tail)

`normal_peak_shoulder_tail(mu=0.0, sigma=1.0)`

```python
from xmath.incerto.stats import normal_peak_shoulder_tail

values = normal_peak_shoulder_tail()
print("Normal scale boundaries: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
Normal scale boundaries: [-2.136, -0.6622, 0.6622, 2.136]
```


(api-normal-variance-peak-shoulder-tail)=
#### [`normal_variance_peak_shoulder_tail`](incerto-api-reference.md#api-normal-variance-peak-shoulder-tail)

`normal_variance_peak_shoulder_tail(mu=0.0, variance=1.0)`

```python
from xmath.incerto.stats import normal_variance_peak_shoulder_tail

values = normal_variance_peak_shoulder_tail()
print("Normal variance boundaries: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
Normal variance boundaries: [-2.334, -0.742, 0.742, 2.334]
```


(api-t-peak-shoulder-tail)=
#### [`t_peak_shoulder_tail`](incerto-api-reference.md#api-t-peak-shoulder-tail)

`t_peak_shoulder_tail(df, mu=0.0, sigma=1.0)`

```python
from xmath.incerto.stats import t_peak_shoulder_tail

values = t_peak_shoulder_tail(3)
print("Student-3 scale boundaries: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
Student-3 scale boundaries: [-2.758, -0.6281, 0.6281, 2.758]
```


(api-lognorm-params-from-mean-and-variance)=
#### [`lognorm_params_from_mean_and_variance`](incerto-api-reference.md#api-lognorm-params-from-mean-and-variance)

`lognorm_params_from_mean_and_variance(mean, variance)`

```python
from xmath.incerto.stats import lognorm_params_from_mean_and_variance

mu, sigma = lognorm_params_from_mean_and_variance(mean=2, variance=3)
print(f"Log-location: {mu:.4f}; log-scale: {sigma:.4f}")
```

**Output**

```text
Log-location: 0.4133; log-scale: 0.7481
```


(api-empirical-survival-function)=
#### [`empirical_survival_function`](incerto-api-reference.md#api-empirical-survival-function)

`empirical_survival_function(x)`

```python
from xmath.incerto.stats import empirical_survival_function

support, survival = empirical_survival_function([1, 2, 2, 4])
print("Support: [" + ", ".join(f"{v:.4g}" for v in support) + "]")
print("Strict survival: [" + ", ".join(f"{v:.4g}" for v in survival) + "]")
```

**Output**

```text
Support: [1, 2, 4]
Strict survival: [0.75, 0.25, 0]
```


(api-dispersion-ratios)=
#### [`dispersion_ratios`](incerto-api-reference.md#api-dispersion-ratios)

`dispersion_ratios(sample)`

```python
from xmath.incerto.stats import dispersion_ratios

for sample in ([2, 2, 2], [0, 0, 10], [1, 2, 3]):
    mean_ratio, median_ratio = dispersion_ratios(sample)
    print(f"{sample}: Std/MeanAD={mean_ratio:.4g}, Std/MedianAD={median_ratio:.4g}")
```

**Output**

```text
[2, 2, 2]: Std/MeanAD=nan, Std/MedianAD=nan
[0, 0, 10]: Std/MeanAD=0.8463, Std/MedianAD=inf
[1, 2, 3]: Std/MeanAD=0.9772, Std/MedianAD=0.5507
```

Standard deviation uses `ddof=0`; both absolute deviations are Gaussian-calibrated.


### Tail diagnostics

Module: `xmath.incerto.tail_diagnostics`.

(api-regular-variation-ratio)=
#### [`regular_variation_ratio`](incerto-api-reference.md#api-regular-variation-ratio)

`regular_variation_ratio(survival, thresholds, *, multiplier=2.0)`

```python
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.tail_diagnostics import regular_variation_ratio

result = regular_variation_ratio(pareto_type1(2).sf, [2, 4], multiplier=2)
for name, values in result.items():
    print(f"{name}: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
threshold: [2, 4]
survival: [0.25, 0.0625]
multiplied_survival: [0.0625, 0.01562]
ratio: [0.25, 0.25]
```


(api-iid-two-sum-tail)=
#### [`iid_two_sum_tail`](incerto-api-reference.md#api-iid-two-sum-tail)

`iid_two_sum_tail(rv, thresholds, *, rtol=1e-10, maxlevel=12)`

```python
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.tail_diagnostics import iid_two_sum_tail

result = iid_two_sum_tail(pareto_type1(1.5), [3, 10, 100])
print("Fields:", ", ".join(result))
print("Two-sum ratio: [" + ", ".join(f"{v:.4g}" for v in result["ratio"]) + "]")
print("Quadrature success: [" + ", ".join(str(v) for v in result["success"]) + "]")
```

**Output**

```text
Fields: x, threshold, log_tail, tail, log_ratio, ratio, success, log_abs_error, nfev
Two-sum ratio: [1.769, 1.391, 1.044]
Quadrature success: [True, True, True]
```


(api-subexponential-sum-ratio)=
#### [`subexponential_sum_ratio`](incerto-api-reference.md#api-subexponential-sum-ratio)

`subexponential_sum_ratio(sampler, survival, thresholds, *, sample_size=100000, random_state=None)`

```python
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.tail_diagnostics import subexponential_sum_ratio

law = pareto_type1(1.5)
result = subexponential_sum_ratio(
    law.rvs, law.sf, [3, 10], sample_size=10000, random_state=20261020,
)
for name, values in result.items():
    fmt = ".0f" if name == "sum_exceedances" else ".4g"
    print(f"{name}: [" + ", ".join(format(v, fmt) for v in values) + "]")
```

**Output**

```text
threshold: [3, 10]
single_tail: [0.1925, 0.03162]
sum_tail: [0.6793, 0.0875]
ratio: [1.765, 1.383]
sum_exceedances: [6793, 875]
```

These seeded Monte Carlo values are finite-sample diagnostics.


### Tail classes

Module: `xmath.incerto.tail_classes`.

(api-bounded-spike-summary)=
#### [`bounded_spike_summary`](incerto-api-reference.md#api-bounded-spike-summary)

`bounded_spike_summary(epsilon)`

```python
from xmath.incerto.tail_classes import bounded_spike_summary

for name, value in bounded_spike_summary(0.01).items():
    print(f"{name}: {value:.4g}")
```

**Output**

```text
support_radius: 10
mean: 0
variance: 1
kurtosis: 100
```

The symmetric three-point law has total nonzero probability `epsilon`.


(api-log-survival-multiplier-ratio)=
#### [`log_survival_multiplier_ratio`](incerto-api-reference.md#api-log-survival-multiplier-ratio)

`log_survival_multiplier_ratio(log_survival, thresholds, *, multiplier=2.0)`

```python
from xmath.incerto.distributions import pareto_type1
from xmath.incerto.tail_classes import log_survival_multiplier_ratio

values = log_survival_multiplier_ratio(pareto_type1(2).logsf, [2, 10, 100])
print("[" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
[-1.386, -1.386, -1.386]
```


### Glossary

Module: `xmath.incerto.glossary`.

(api-pareto-tail-quantities)=
#### [`pareto_tail_quantities`](incerto-api-reference.md#api-pareto-tail-quantities)

`pareto_tail_quantities(alpha, x_m, threshold)`

```python
from xmath.incerto.glossary import pareto_tail_quantities

for alpha in (1, 1.5):
    result = pareto_tail_quantities(alpha, x_m=1, threshold=4)
    print(f"alpha={alpha:.4g}")
    for name, value in result.items():
        print(f"  {name}: {value:.4g}")
```

**Output**

```text
alpha=1
  log_survival: -1.386
  mean_excess: inf
  conditional_mean: inf
  tail_first_moment: inf
alpha=1.5
  log_survival: -2.079
  mean_excess: 8
  conditional_mean: 12
  tail_first_moment: 1.5
```


(api-pareto-normalized-maximum-cdf)=
#### [`pareto_normalized_maximum_cdf`](incerto-api-reference.md#api-pareto-normalized-maximum-cdf)

`pareto_normalized_maximum_cdf(y, *, alpha, n)`

```python
from xmath.incerto.glossary import pareto_normalized_maximum_cdf

for n in (1, 4, 16):
    values = pareto_normalized_maximum_cdf([0.5, 1, 2], alpha=1.5, n=n)
    print(f"n={n:d}: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
n=1: [0, 0, 0.6464]
n=4: [0.007359, 0.3164, 0.6906]
n=16: [0.04449, 0.3561, 0.6994]
```

The maximum is divided by $x_m n^{1/\alpha}$ under an exact iid Pareto model.


### Lecture risk

Module: `xmath.incerto.lecture_risk`.

(api-repeated-event-probability)=
#### [`repeated_event_probability`](incerto-api-reference.md#api-repeated-event-probability)

`repeated_event_probability(probability, trials)`

```python
from xmath.incerto.lecture_risk import repeated_event_probability

for trials in (0, 1, 1000):
    print(f"{trials:d} {repeated_event_probability(1e-4, trials):.3e}")
```

**Output**

```text
0 0.000e+00
1 1.000e-04
1000 9.517e-02
```

Trials are independent with the same event probability.


(api-survival-from-hazards)=
#### [`survival_from_hazards`](incerto-api-reference.md#api-survival-from-hazards)

`survival_from_hazards(hazards)`

```python
from xmath.incerto.lecture_risk import survival_from_hazards

values = survival_from_hazards([0.02] * 3)
print("[" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
[1, 0.98, 0.9604, 0.9412]
```

The initial survival is included; inputs are conditional hazards.


(api-finite-payoff-mean)=
#### [`finite_payoff_mean`](incerto-api-reference.md#api-finite-payoff-mean)

`finite_payoff_mean(probabilities, payoffs)`

```python
from xmath.incerto.lecture_risk import finite_payoff_mean

print(f"{finite_payoff_mean([0.98, 0.02], [1, -80]):.4g}")
```

**Output**

```text
-0.62
```

Probabilities must sum to one; there is no implicit normalization.


(api-absorbing-resource-path)=
#### [`absorbing_resource_path`](incerto-api-reference.md#api-absorbing-resource-path)

`absorbing_resource_path(initial, increments)`

```python
from xmath.incerto.lecture_risk import absorbing_resource_path

for increments in ([-4, 4], [4, -4]):
    path = absorbing_resource_path(3, increments)
    print(f"{increments}: [" + ", ".join(f"{v:.4g}" for v in path) + "]")
```

**Output**

```text
[-4, 4]: [3, 0, 0]
[4, -4]: [3, 7, 3]
```

The initial resource level is included and ruin is absorbing.


(api-split-power-damage-logsf)=
#### [`split_power_damage_logsf`](incerto-api-reference.md#api-split-power-damage-logsf)

`split_power_damage_logsf(threshold, *, alpha, power, modules, exposure=1.0, x_m=1.0)`

```python
from xmath.incerto.lecture_risk import split_power_damage_logsf

for modules in (1, 16):
    value = split_power_damage_logsf(4, alpha=1.5, power=2, modules=modules)
    print(f"{modules:d} log survival: {value:.4g}")
```

**Output**

```text
1 log survival: -1.04
16 log survival: -3.119
```

The model splits a Pareto power-damage exposure into equal modules.


(api-block-failure-summary)=
#### [`block_failure_summary`](incerto-api-reference.md#api-block-failure-summary)

`block_failure_summary(probability, blocks, exposure=1.0)`

```python
from xmath.incerto.lecture_risk import block_failure_summary

for name, value in block_failure_summary(0.02, blocks=10).items():
    print(f"{name}: {value:.4g}")
```

**Output**

```text
mean: 0.02
variance: 0.00196
any_failure: 0.1829
total_failure: 1.024e-17
```

Blocks fail independently and have equal all-or-nothing exposures.


### Finite moments

Module: `xmath.incerto.finite_moments`.

(api-normal-mixture-log-density)=
#### [`normal_mixture_log_density`](incerto-api-reference.md#api-normal-mixture-log-density)

`normal_mixture_log_density(x, variances, weights)`

```python
from xmath.incerto.finite_moments import normal_mixture_log_density

values = normal_mixture_log_density([0, 1, 4], [0.5, 50.5], [0.99, 0.01])
print("[" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
[-0.5814, -1.58, -7.643]
```

Components are centered Gaussian laws with the stated variances and weights.


(api-normal-mixture-log-abs-survival)=
#### [`normal_mixture_log_abs_survival`](incerto-api-reference.md#api-normal-mixture-log-abs-survival)

`normal_mixture_log_abs_survival(u, variances, weights)`

```python
from xmath.incerto.finite_moments import normal_mixture_log_abs_survival

values = normal_mixture_log_abs_survival([0, 1, 4], [0.5, 50.5], [0.99, 0.01])
print("[" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
[0, -1.804, -5.161]
```

Thresholds are finite and nonnegative; exceedances are two-sided absolute values.


(api-normal-mixture-absolute-moment)=
#### [`normal_mixture_absolute_moment`](incerto-api-reference.md#api-normal-mixture-absolute-moment)

`normal_mixture_absolute_moment(q, variances, weights)`

```python
from xmath.incerto.finite_moments import normal_mixture_absolute_moment

for order in (2, 4):
    print(f"{order} {normal_mixture_absolute_moment(order, [0.5, 50.5], [0.99, 0.01]):.4g}")
```

**Output**

```text
2 1
4 77.25
```

All positive-order absolute moments exist for a finite Gaussian mixture.


(api-pareto-moment-estimator-summary)=
#### [`pareto_moment_estimator_summary`](incerto-api-reference.md#api-pareto-moment-estimator-summary)

`pareto_moment_estimator_summary(alpha, q, n, x_m=1.0)`

```python
from xmath.incerto.finite_moments import pareto_moment_estimator_summary

for alpha in (1.5, 3, 6):
    print(f"alpha={alpha:.4g}")
    for name, value in pareto_moment_estimator_summary(alpha, q=2, n=200).items():
        print(f"  {name}: {value:.4g}")
```

**Output**

```text
alpha=1.5
  moment: inf
  estimator_variance: nan
alpha=3
  moment: 3
  estimator_variance: inf
alpha=6
  moment: 1.5
  estimator_variance: 0.00375
```

The empirical moment averages `n` iid exact Pareto observations.


(api-power-mean)=
#### [`power_mean`](incerto-api-reference.md#api-power-mean)

`power_mean(sample, order)`

```python
from xmath.incerto.finite_moments import power_mean

for order in (1, 2, 4, float("inf")):
    print(f"{order} {power_mean([0, 0, 0, -4], order):.4g}")
```

**Output**

```text
1 1
2 2
4 2.828
inf 4
```

Orders are at least one; powers use absolute values and normalization by sample size.


### Multivariate

Module: `xmath.incerto.multivariate`.

(api-student-entropy)=
#### [`student_entropy`](incerto-api-reference.md#api-student-entropy)

`student_entropy(nu, dimension, logdet=0.0)`

```python
from xmath.incerto.multivariate import student_entropy

for dimension in (1, 2):
    print(f"{dimension} {student_entropy(3, dimension, logdet=0):.4g}")
```

**Output**

```text
1 1.773
2 3.505
```

Entropy is in nats; `logdet` is the log determinant of scatter.


(api-student-mutual-information)=
#### [`student_mutual_information`](incerto-api-reference.md#api-student-mutual-information)

`student_mutual_information(nu, rho=0.0)`

```python
from xmath.incerto.multivariate import student_mutual_information

for nu in (1, 5):
    print(f"{nu} {student_mutual_information(nu, rho=0):.4g}")
```

**Output**

```text
1 0.2242
5 0.01713
```

Mutual information is in nats for unit diagonal scatter.


(api-bivariate-student)=
#### [`bivariate_student`](incerto-api-reference.md#api-bivariate-student)

`bivariate_student(rng, nu, rho, size)`

```python
import numpy as np
from xmath.incerto.multivariate import bivariate_student

rng = np.random.default_rng(20261020)
for row in bivariate_student(rng, nu=3, rho=0.6, size=3):
    print("[" + ", ".join(f"{v:.4g}" for v in row) + "]")
```

**Output**

```text
[-1.722, -1.126]
[-1.263, -0.0244]
[7.266, 2.572]
```

Each pair shares one gamma precision, with unit diagonal scatter and off-diagonal `rho`.


(api-sample-correlation)=
#### [`sample_correlation`](incerto-api-reference.md#api-sample-correlation)

`sample_correlation(x, y, *, centered=True)`

```python
from xmath.incerto.multivariate import sample_correlation

for centered in (True, False):
    print(f"{centered} {sample_correlation([1, 2, 3], [3, 2, 1], centered=centered):.4g}")
print(f"Constant centered column: {sample_correlation([1, 1, 1], [1, 2, 3]):.4g}")
```

**Output**

```text
True -1
False 0.7143
Constant centered column: nan
```

`centered=True` uses Pearson correlation; `False` uses the uncentered cosine along the last axis.


(api-symmetric-spectrum)=
#### [`symmetric_spectrum`](incerto-api-reference.md#api-symmetric-spectrum)

`symmetric_spectrum(matrix)`

```python
from xmath.incerto.multivariate import symmetric_spectrum

values = symmetric_spectrum([[0, 2], [2, 0]])
print("[" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
[-2, 2]
```

The real square input must be finite, nonempty, and exactly symmetric.


(api-regression-population-r2)=
#### [`regression_population_r2`](incerto-api-reference.md#api-regression-population-r2)

`regression_population_r2(slope, error_variance)`

```python
from xmath.incerto.multivariate import regression_population_r2

print(f"{regression_population_r2(slope=1, error_variance=3):.4g}")
```

**Output**

```text
0.25
```

The error is independent and the predictor has variance one.


(api-normal-location-mixture-kurtosis)=
#### [`normal_location_mixture_kurtosis`](incerto-api-reference.md#api-normal-location-mixture-kurtosis)

`normal_location_mixture_kurtosis(separation, sigma=1.0)`

```python
from xmath.incerto.multivariate import normal_location_mixture_kurtosis

for separation in (0, 2):
    print(f"{separation:d} {normal_location_mixture_kurtosis(separation, sigma=1):.4g}")
```

**Output**

```text
0 3
2 1.72
```

Equal Gaussian components are centered at plus and minus `separation`, with common standard deviation `sigma`.


### Power laws

Module: `xmath.incerto.power_laws`.

(api-pareto-product-logsf)=
#### [`pareto_product_logsf`](incerto-api-reference.md#api-pareto-product-logsf)

`pareto_product_logsf(log_threshold, a, b)`

```python
import numpy as np
from xmath.incerto.power_laws import pareto_product_logsf

for a, b in ((1, 2), (1, 1)):
    values = pareto_product_logsf(np.log([1, 10, 100]), a, b)
    print(f"a={a:d}, b={b:d}: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
a=1, b=2: [-0, -1.661, -3.917]
a=1, b=1: [-0, -1.108, -2.881]
```

The factors are independent unit-minimum Paretos; thresholds enter on the log scale.


(api-pareto-truncated-moment)=
#### [`pareto_truncated_moment`](incerto-api-reference.md#api-pareto-truncated-moment)

`pareto_truncated_moment(cutoff, alpha, order, *, x_m=1.0)`

```python
from xmath.incerto.power_laws import pareto_truncated_moment

for order in (1, 1.5, 2):
    values = pareto_truncated_moment([10, 100], alpha=1.5, order=order)
    print(f"order={order:.4g}: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
order=1: [2.051, 2.7]
order=1.5: [3.454, 6.908]
order=2: [6.487, 27]
```

Truncation uses the indicator below `cutoff`, without conditional renormalization.


(api-student-double-pareto-scales)=
#### [`student_double_pareto_scales`](incerto-api-reference.md#api-student-double-pareto-scales)

`student_double_pareto_scales(alpha)`

```python
from xmath.incerto.power_laws import student_double_pareto_scales

for name, value in student_double_pareto_scales(3).items():
    print(f"{name}: {value:.4g}")
```

**Output**

```text
student_tail_constant: 1.103
tail_scale: 0.7683
variance_scale: 0.5774
tail_matched_variance: 1.771
```

Matching uses a unit shifted double Pareto and requires `alpha > 2`.


(api-q-exponential-logsf)=
#### [`q_exponential_logsf`](incerto-api-reference.md#api-q-exponential-logsf)

`q_exponential_logsf(x, q, beta=1.0)`

```python
from xmath.incerto.power_laws import q_exponential_logsf

for q in (0.75, 1, 1.25):
    values = q_exponential_logsf([0, 1, 2], q, beta=2)
    print(f"q={q:.4g}: [" + ", ".join(f"{v:.4g}" for v in values) + "]")
```

**Output**

```text
q=0.75: [-0, -3.466, -inf]
q=1: [-0, -2, -4]
q=1.25: [-0, -1.216, -2.079]
```

The normalized law requires `x >= 0` and `0 < q < 2`; `-inf` means zero survival.


(api-entropy-log-constraint-example)=
#### [`entropy_log_constraint_example`](incerto-api-reference.md#api-entropy-log-constraint-example)

`entropy_log_constraint_example()`

```python
from xmath.incerto.power_laws import entropy_log_constraint_example

for name, value in entropy_log_constraint_example().items():
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

Independent quadrature uses $x_m=1$ and $\mathbb E\log X=1/2$.


### Gas speeds

Module: `xmath.incerto.gas_speeds`. These helpers simulate equilibrium
observations, not collision trajectories. The
[Rayleigh note](incerto-rayleigh-speed.md) derives the chamber, effusion, and
finite-system laws and gives the seeded seven-family comparison.

(api-chi-speed)=
#### [`chi_speed`](incerto-api-reference.md#api-chi-speed)

`chi_speed(dimension=2, thermal_variance=1.0)`

Return the frozen chi law with scale $\sqrt{kT/m}$. `dimension` is a positive
integer; `thermal_variance` is the finite positive component variance $kT/m$.
The object exposes SciPy `pdf`, `cdf`, `sf`, `ppf`, `moment`, and `rvs`.
Support is $[0,\infty)$; dimensions 1, 2, and 3 are half-normal, Rayleigh,
and Maxwell. Density and CDF are zero below zero.

```python
from xmath.incerto.gas_speeds import chi_speed

for d in (2, 3):
    law = chi_speed(d)
    print(f"d={d}: mean={law.mean():.4f}, variance={law.var():.4f}")
```

**Output**

```text
d=2: mean=1.2533, variance=0.4292
d=3: mean=1.5958, variance=0.4535
```

(api-sample-chi-speed)=
#### [`sample_chi_speed`](incerto-api-reference.md#api-sample-chi-speed)

`sample_chi_speed(size, *, dimension=2, thermal_variance=1.0, random_state=None)`

Draw independent normal components in an array of shape `(dimension, size)`
and reduce their lengths with `hypot`. Return a one-dimensional speed array.
Counts must be positive integers. `random_state` accepts an integer seed,
a NumPy Generator, or `None`. All first components precede all second
components in the random stream; changing `size` changes that pairing.
Zero or nonfinite numerical draws raise rather than being replaced.

(api-speedfit)=
#### [`SpeedFit`](incerto-api-reference.md#api-speedfit)

An immutable record returned by `fit_speed_models`. Fields are `name`,
`law` (frozen SciPy distribution), `parameters` (number estimated),
`log_likelihood`, `aic`, `ks_statistic`, `ks_pvalue`, and `boundary`.
The last field is empty for an interior fit, or names the GIG boundary law.
Reported p-values are uncorrected for fitting. Parameters of `law` follow
SciPy's conventions; in particular inverse Gaussian shape is $\mu/\kappa$
and scale is $\kappa$ for physical mean $\mu$ and shape $\kappa$.

(api-fit-speed-models)=
#### [`fit_speed_models`](incerto-api-reference.md#api-fit-speed-models)

`fit_speed_models(sample)`

Return an AIC-sorted tuple of `SpeedFit` records for Gaussian, Rayleigh,
Weibull, Gamma, inverse Gamma, inverse Gaussian, and GIG. Input must be a
one-dimensional vector of at least three finite positive observations and
must not be constant. Locations of positive families are fixed at zero.
Gaussian has two estimated parameters, Rayleigh one, GIG three, and the
others two. Inverse Gamma is fitted through the Gamma MLE of reciprocals;
the transformation's Jacobian is independent of its parameters.

The GIG fit includes normalizable Gamma and inverse Gamma boundaries.
An inward score test certifies a boundary maximum using concavity in natural
parameters, as proved in the [note](incerto-rayleigh-speed.md#rayleigh-speed-gig-boundary).
Otherwise SciPy's interior optimizer must converge within 4,000 iterations
and 8,000 evaluations, and its likelihood must match or beat both boundaries
to within $10^{-7}$. This numerical branch does not certify a global optimum.
The frozen interior law uses SciPy's `geninvgauss(p, b, scale=scale)`:
the note's canonical parameters are $a=b/\mathrm{scale}$ and
$b_{\rm canonical}=b\,\mathrm{scale}$.

Nonfinite diagnostics and optimizer failures raise. Large interior GIG
samples may be slow because their CDF uses quadrature. GIG's AIC retains
the nominal three-parameter penalty on a boundary; its usual regular-model
interpretation is not asserted there. KS uses the ordinary exact finite-$n$
distribution for a specified continuous CDF, without adjusting for the
estimated parameters. Very small p-values may underflow to zero.

(api-speed-fit-table)=
#### [`speed_fit_table`](incerto-api-reference.md#api-speed-fit-table)

`speed_fit_table(fits)`

Return a text table sorted by AIC: model, parameter count, AIC (one decimal),
KS distance (four decimals), and raw p-value (three significant digits).
This function formats the supplied records; it performs no fitting or test.

(api-speed-qq)=
#### [`speed_qq`](incerto-api-reference.md#api-speed-qq)

`speed_qq(sample, law, probabilities=None)`

Return `(theoretical, empirical)` quantile arrays. Input observations are
finite, positive, and one-dimensional. The law must expose `ppf`.
The default uses 199 equally spaced probabilities from $0.005$ to $0.995$;
an explicit nonempty vector must increase strictly inside $(0,1)$.
Empirical quantiles use NumPy's linear interpolation. Nonfinite theoretical
quantiles raise. No observations are removed and no confidence band is added.

(api-squared-speed-ks)=
#### [`squared_speed_ks`](incerto-api-reference.md#api-squared-speed-ks)

`squared_speed_ks(sample, thermal_variance=1.0)`

Return a dictionary with float `statistic` and `pvalue` for a two-sided KS
comparison of squared speeds against $\operatorname{Exp}(2kT/m)$, where
the argument is the mean. The helper estimates no parameters.
Usual calibration requires iid observations and `thermal_variance` specified
independently of them. The input must be a finite positive vector and the
variance finite and positive. Arithmetic overflow raises.

(api-speed-fit-figure)=
#### [`speed_fit_figure`](incerto-api-reference.md#api-speed-fit-figure)

`speed_fit_figure(sample, fits, *, kind="density", thermal_variance=1.0)`

Return a Matplotlib figure, applying the shared DejaVu Serif theme at scale
1.1 and five-inch single-column layout. Supply the sample used to produce
the seven fits. `kind="density"` draws two panels with 80-bin histograms
and 401-point fitted curves on linear and log axes. `kind="qq"` draws two
panels: five fitted families, then squared speeds against the fixed exponential
law. `kind="qq-tail"` gives the inverse Gaussian and inverse Gamma fits their
own two-panel figure. Both use `speed_qq`'s default grid and
does not clip the displayed quantiles. `thermal_variance` supplies the
exponential null and the density viewport scale, not a new fit.
The caller closes or saves the figure; the helper writes no files.

### Averaging and limit laws

Module: `xmath.incerto.limits`.

(api-pareto-capped-mean)=
#### [`pareto_capped_mean`](incerto-api-reference.md#api-pareto-capped-mean)

`pareto_capped_mean(cap, alpha, *, x_m=1.0)`

Return $E[\min(X,c)]$ for exact Pareto exponent `alpha > 0` and cutoff
`x_m > 0`. Accept finite nonnegative scalar or array caps; return a float or
matching array. A cap at or below the cutoff returns the cap. The boundary
`alpha=1` uses logarithmic growth. This differs from the indicator-truncated
moment: observations above the cap contribute the cap. Invalid inputs raise
`ValueError`; nonrepresentable finite answers raise `FloatingPointError`.

```python
from xmath.incerto.limits import pareto_capped_mean

for alpha in [.8, 1., 1.3]:
    print(f"alpha={alpha:g}: " + ", ".join(f"{v:.3f}" for v in pareto_capped_mean([1, 10, 1000], alpha)))
```

**Output**

```text
alpha=0.8: 1.000, 3.924, 15.905
alpha=1: 1.000, 3.303, 7.908
alpha=1.3: 1.000, 2.663, 3.914
```

(api-symmetric-stable-mean-abs)=
#### [`symmetric_stable_mean_abs`](incerto-api-reference.md#api-symmetric-stable-mean-abs)

`symmetric_stable_mean_abs(alpha, *, scale=1.0, n=1)`

Return the mean absolute deviation of an iid sample mean from its location,
for the symmetric stable characteristic function
$\exp(i\mu t-|\mathtt{scale}\,t|^\alpha)$, `0 < alpha <= 2`.
The scale is positive and finite; `n` is an integer in $[1,2^{53}]$.
At exponent 2, the parent variance is twice the squared scale. Invalid
arguments raise `ValueError`; finite-result overflow raises `FloatingPointError`.
The [derivation](incerto-generalized-central-limit-theorem.md#stable-mean-deviation)
fixes the parameterization and the moment boundary.

```python
from xmath.incerto.limits import symmetric_stable_mean_abs

for alpha in [.8, 1., 1.5, 2.]:
    print(f"alpha={alpha:g}: {symmetric_stable_mean_abs(alpha, n=100):.4g}")
```

**Output**

```text
alpha=0.8: inf
alpha=1: inf
alpha=1.5: 0.3674
alpha=2: 0.1128
```

(api-repeated-centered-sums)=
#### [`repeated_centered_sums`](incerto-api-reference.md#api-repeated-centered-sums)

`repeated_centered_sums(rv, sizes, *, center=0.0, repetitions=600, random_state=0, batch=32)`

Draw independent samples at each size from a frozen law with an `rvs` method.
Return an array of shape `(len(sizes), repetitions)` containing sums of
`X - center`. The finite center is specified per observation; the function
does not estimate it or normalize sums. `sizes` is a nonempty positive integer
vector. Repetitions and batch are positive integers. A seed creates a PCG64
stream; a passed NumPy Generator is used as supplied. Each size is at most one million and the experiment
at most 50 million observations. Invalid settings raise `ValueError`;
nonfinite draws or arithmetic raise `FloatingPointError`, without removing
observations. Batching limits temporary memory, not the parent law's support.

```python
from scipy.stats import norm
from xmath.incerto.limits import repeated_centered_sums

sums = repeated_centered_sums(norm(loc=2), [10, 100], center=2, repetitions=400, random_state=17)
print("Shape:", sums.shape)
for n, row in zip([10, 100], sums):
    print(f"n={n}: normalized SD={row.std(ddof=1)/n**.5:.3f}")
```

**Output**

```text
Shape: (2, 400)
n=10: normalized SD=1.047
n=100: normalized SD=1.042
```

(api-pareto-plugin-nonfinite-probability)=
#### [`pareto_plugin_nonfinite_probability`](incerto-api-reference.md#api-pareto-plugin-nonfinite-probability)

`pareto_plugin_nonfinite_probability(alpha, n)`

Return the exact probability that the known-cutoff Pareto mean fit is infinite,
for iid data with true `alpha > 0`. The log-excess sum is Gamma with shape `n`
and rate `alpha`; its survival at `n` gives the answer. The cutoff cancels.
This describes the continuous mathematical model, including when finite
estimates are common although the population mean is infinite.
`n` is an integer in $[1,2^{53}]$. Invalid arguments raise `ValueError`;
very small tail probabilities can underflow to zero.

```python
from xmath.incerto.limits import pareto_plugin_nonfinite_probability

for alpha in [.8, 1., 1.5]:
    print(f"alpha={alpha:g}: {pareto_plugin_nonfinite_probability(alpha, 100):.4f}")
```

**Output**

```text
alpha=0.8: 0.9829
alpha=1: 0.4867
alpha=1.5: 0.0000
```

(api-bernoulli-chernoff-log-bound)=
#### [`bernoulli_chernoff_log_bound`](incerto-api-reference.md#api-bernoulli-chernoff-log-bound)

`bernoulli_chernoff_log_bound(p, q, n)`

Return the logarithm of the Chernoff upper bound on $P(S_n\ge nq)$ for
an iid Bernoulli(`p`) sum. Require `0 < p < 1`, `0 <= q <= 1` and integer
`n` in $[1,2^{53}]$. At `q <= p` return zero (bound one); at `q=1` the
all-success probability is exact. This is a bound, not a tail-probability
approximation. Invalid inputs raise `ValueError`.

```python
import math
from xmath.incerto.limits import bernoulli_chernoff_log_bound

for q in [.2, .5, 1.]:
    log_bound = bernoulli_chernoff_log_bound(.2, q, 20)
    print(f"q={q:g}: log bound={log_bound:.4f}, bound={math.exp(log_bound):.4g}")
```

**Output**

```text
q=0.2: log bound=0.0000, bound=1
q=0.5: log bound=-4.4629, bound=0.01153
q=1: log bound=-32.1888, bound=1.049e-14
```

### Figures

Module: `xmath.incerto.figures`.

(api-set-theme)=
#### [`set_theme`](incerto-api-reference.md#api-set-theme)

`set_theme(*, scale=1.0)`

Apply the Incerto Matplotlib theme.

(api-style-axes)=
#### [`style_axes`](incerto-api-reference.md#api-style-axes)

`style_axes(axes, *, grid_axis='y', legend=True)`

Apply final Incerto styling to one or more axes.

(api-savefig)=
#### [`savefig`](incerto-api-reference.md#api-savefig)

`savefig(fig, path, **kwargs)`

Save a figure with Incerto defaults.

(api-vertical-figure)=
#### [`vertical_figure`](incerto-api-reference.md#api-vertical-figure)

`vertical_figure(panels, *, panel_height=3.8)`

Returns a figure and a flat array of vertically stacked axes, five inches
wide. `panels` is a positive integer; `panel_height` is positive and finite.
Call [`set_theme`](incerto-api-reference.md#api-set-theme) with `scale=1.1`
for the P1 mobile layout. This adds no drawing or export side effect.

```python
import matplotlib.pyplot as plt
from xmath.incerto.figures import set_theme, vertical_figure

set_theme(scale=1.1)
fig, axes = vertical_figure(3)
print(f"Panels: {len(axes):d}; width: {fig.get_figwidth():.4g} inches")
plt.close(fig)
```

**Output**

```text
Panels: 3; width: 5 inches
```

(api-distribution-card)=
#### [`distribution_card`](incerto-api-reference.md#api-distribution-card)

`distribution_card(models, x, tail_x, *, title, support_note, coordinate="x", tail_label="Survival")`

Returns a three-panel density/CDF/log-log survival figure. `models` contains
`(frozen_law, label)` pairs exposing `pdf`, `cdf`, `sf` and `support`;
`x` and `tail_x` are increasing finite vectors, with positive `tail_x`.
Finite support endpoints get dotted markers in the corresponding curve color.
Exact zero survival is masked on log axes; the visible probability range
starts at $10^{-8}$. Nonfinite evaluations raise, including a grid point
on a singular density endpoint. The caller supplies support text and explicit
parameters; the helper performs no moment matching.

Use [`savefig`](incerto-api-reference.md#api-savefig) for deterministic SVG export.
This example constructs the card without writing files:

:::{dropdown} Code

```python
import matplotlib.pyplot as plt
import numpy as np
from xmath.incerto.distributions import generalized_pareto
from xmath.incerto.figures import distribution_card, set_theme

set_theme(scale=1.1)
fig = distribution_card(
    [(generalized_pareto(-0.4), "xi = -0.4")],
    np.linspace(-0.2, 4, 101), np.geomspace(0.05, 10, 101),
    title="GPD", support_note="Support: 0 ≤ y ≤ 2.5", coordinate="Excess y",
)
print(f"Card panels: {len(fig.axes):d}")
plt.close(fig)
```

**Output**

```text
Card panels: 3
```
:::

### Finite-sample concentration

(api-kappa-from-mads)=
#### [`kappa_from_mads`](incerto-api-reference.md#api-kappa-from-mads)

`concentration.kappa_from_mads(mad_n0, mad_n, n, *, n0=1)` evaluates the
two-size population metric. Counts are integers with
`1 <= n0 < n <= 10000000`; MADs must be nonnegative real scalars and
positive finite sum MADs must not decrease. No clipping to [0,1] occurs.

```python
from xmath.incerto.concentration import kappa_from_mads

for pair in ((1, 1.5), (2/3, 14/15), (1, 1), (0, 0), (float("inf"), float("inf"))):
    print(f"kappa: {kappa_from_mads(*pair, 2):.4g}")
try:
    kappa_from_mads(1, 0.9, 2)
except ValueError as error:
    print(type(error).__name__ + ": " + str(error))
```

**Output**

```text
kappa: 0.2905
kappa: -0.06004
kappa: -inf
kappa: nan
kappa: nan
ValueError: population sum MAD must not decrease
```

(api-stable-kappa)=
#### [`stable_kappa`](incerto-api-reference.md#api-stable-kappa)

`concentration.stable_kappa(alpha)` requires `0 < alpha <= 2`.
For an integrable stable law it evaluates `2-alpha`; the sum growth
exponent is `1/alpha`.

```python
from xmath.incerto.concentration import stable_kappa

for alpha in (0.8, 1, 1.5, 2):
    print(f"alpha={alpha:g}: {stable_kappa(alpha):.3f}")
```

**Output**

```text
alpha=0.8: nan
alpha=1: nan
alpha=1.5: 0.500
alpha=2: 0.000
```

(api-gaussian-sum-mad)=
#### [`gaussian_sum_mad`](incerto-api-reference.md#api-gaussian-sum-mad)

`concentration.gaussian_sum_mad(n, *, sigma=1)` uses the parent standard
deviation, not the stable CF scale. Requires positive finite sigma and
integer `1 <= n <= 10000000`.

```python
from xmath.incerto.concentration import gaussian_sum_mad

for n in (1, 2, 100):
    print(f"n={n}: sum MAD={gaussian_sum_mad(n):.4f}")
```

**Output**

```text
n=1: sum MAD=0.7979
n=2: sum MAD=1.1284
n=100: sum MAD=7.9788
```

(api-gamma-sum-mad)=
#### [`gamma_sum_mad`](incerto-api-reference.md#api-gamma-sum-mad)

`concentration.gamma_sum_mad(n, *, shape=1, scale=1)` is the closed form
for sums of independent Gamma(shape, scale) variables. Shape one gives
exponential summands. Shape is in `[0.000001, 1000000]`; scale is positive
and finite, and counts have the same domain as the Gaussian helper.
Log-domain evaluation and a Stirling remainder avoid large-shape cancellation.

```python
from xmath.incerto.concentration import gamma_sum_mad

for n in (1, 2, 100):
    print(f"n={n}: sum MAD={gamma_sum_mad(n):.4f}")
try:
    gamma_sum_mad(10, scale=1e308)
except FloatingPointError:
    print("FloatingPointError: finite MAD exceeds floating-point range")
```

**Output**

```text
n=1: sum MAD=0.7358
n=2: sum MAD=1.0827
n=100: sum MAD=7.9722
FloatingPointError: finite MAD exceeds floating-point range
```

(api-student-sum-mad)=
#### [`student_sum_mad`](incerto-api-reference.md#api-student-sum-mad)

`concentration.student_sum_mad(n, *, nu=3, scale=1)` uses ordinary
Student degrees of freedom `nu` in `[0.5, 1000]` and positive finite scale.
It evaluates closed forms at sizes one and two for all admitted degrees of
freedom, and at every integer size up to ten million for the cubic law.

```python
from xmath.incerto.concentration import student_sum_mad

for nu in (1, 1.5, 3, 5):
    print(f"nu={nu:g}: two-sum MAD={student_sum_mad(2, nu=nu):.4f}")
print(f"Cubic at n=30: {student_sum_mad(30):.4f}")
try:
    student_sum_mad(3, nu=5)
except ValueError as error:
    print(type(error).__name__ + ": " + str(error))
```

**Output**

```text
nu=1: two-sum MAD=inf
nu=1.5: two-sum MAD=3.4126
nu=3: two-sum MAD=1.6540
nu=5: two-sum MAD=1.3840
Cubic at n=30: 7.2218
ValueError: closed form requires n <= 2 or nu=3
```

(api-estimate-sum-mad)=
#### [`estimate_sum_mad`](incerto-api-reference.md#api-estimate-sum-mad)

`concentration.estimate_sum_mad(rv, n, *, mean, repetitions=20000, random_state=0)`
uses a frozen SciPy-style law and the **known population mean**. The returned
dictionary contains `mad` and `mc_se`. Replications are independent and
all draws are retained; nonfinite observations raise. It requires at least
two replications, at most a million summands, and at most 50 million draws.
The estimated SE has its usual interpretation only if the absolute centered
sum has finite variance; it is not a model-validity or coverage guarantee.

```python
from scipy.stats import t
from xmath.incerto.concentration import estimate_sum_mad, student_sum_mad

result = estimate_sum_mad(t(3), 8, mean=0, random_state=20261089)
print(f"Estimate: {result['mad']:.3f}, MC SE: {result['mc_se']:.3f}")
print(f"Exact: {student_sum_mad(8):.3f}")
```

**Output**

```text
Estimate: 3.564, MC SE: 0.025
Exact: 3.578
```

(api-lognormal-sum-mad)=
#### [`lognormal_sum_mad`](incerto-api-reference.md#api-lognormal-sum-mad)

`concentration.lognormal_sum_mad(n, sigma, *, mu=0, repetitions=20000, random_state=0)`
uses log-location `mu` and log-scale `sigma`, with sigma in `[0.0001, 10]`.
The single-summand result is exact. Larger sizes use a bounded conditional
Monte Carlo payoff and return `mad` and `mc_se`. With at least two
replications, `(n-1)*repetitions` must not exceed 50 million and n must not
exceed a million.
The underlying normal generator is PCG64; no samples are discarded.

```python
from xmath.incerto.concentration import lognormal_sum_mad

for n in (1, 2, 16):
    result = lognormal_sum_mad(n, 1, random_state=20261088)
    print(f"n={n}: MAD={result['mad']:.3f}, MC SE={result['mc_se']:.3f}")
try:
    lognormal_sum_mad(1, 1, mu=1000)
except FloatingPointError:
    print("FloatingPointError: finite MAD exceeds floating-point range")
```

**Output**

```text
n=1: MAD=1.263, MC SE=0.000
n=2: MAD=1.973, MC SE=0.009
n=16: MAD=6.422, MC SE=0.052
FloatingPointError: finite MAD exceeds floating-point range
```

(api-normal-location-mixture-sum-mad)=
#### [`normal_location_mixture_sum_mad`](incerto-api-reference.md#api-normal-location-mixture-sum-mad)

`concentration.normal_location_mixture_sum_mad(n, separation, *, sigma=1)`
evaluates the binomial mixture for independent copies of
`separation*R + sigma*Z`, with equiprobable signs R and independent standard
normal Z. Separation is nonnegative, sigma positive, both finite;
integer n is at most 10,000. The exact formula has n+1 terms.

```python
from xmath.incerto.concentration import normal_location_mixture_sum_mad

for n in (1, 2, 8):
    print(f"n={n}: sum MAD={normal_location_mixture_sum_mad(n, 3):.4f}")
```

**Output**

```text
n=1: sum MAD=3.0008
n=2: sum MAD=3.5642
n=8: sum MAD=7.1947
```

(api-gaussian-equivalent-n)=
#### [`gaussian_equivalent_n`](incerto-api-reference.md#api-gaussian-equivalent-n)

`concentration.gaussian_equivalent_n(sum_mad, n_g, *, max_n=1000000)`
returns the smallest matching integer, using doubling then binary search.
Both counts are integers in `[1, 10000000]`. The callable must be a
deterministic population sum-MAD curve for an iid integrable law with
positive MAD; noisy estimates cannot support the search. The criterion is
relative mean absolute error with equal one-observation MADs.
An eight-machine-epsilon tolerance, scaled by the absolute log target
or one, handles rounded equalities; closer crossings cannot be distinguished.

```python
from xmath.incerto.concentration import gaussian_equivalent_n, student_sum_mad

print("Gaussian reference 30:", gaussian_equivalent_n(student_sum_mad, 30))
print("Insufficient search cap:", gaussian_equivalent_n(student_sum_mad, 30, max_n=10))
print("Reference size one:", gaussian_equivalent_n(student_sum_mad, 1))
```

**Output**

```text
Gaussian reference 30: 44
Insufficient search cap: None
Reference size one: 1
```

### Base package

(api-exceedance-fraction)=
#### [`exceedance_fraction`](incerto-api-reference.md#api-exceedance-fraction)

`xmath.exceedance_fraction(samples, threshold)`

```python
from xmath import exceedance_fraction

for threshold in (0, 2, 4):
    print(f"{threshold:d} {exceedance_fraction([1, 2, 2, 4], threshold):.4g}")
```

**Output**

```text
0 1
2 0.25
4 0
```

Exceedances are strict; the sample must be nonempty and finite.

## Power-law fitting and model checks

The functions in `xmath.incerto.powerlaw` are also re-exported from
`xmath.incerto`. Their `alpha` is the survival exponent; the CSN density/mass
exponent is `alpha + 1`. Inputs are positive finite real one-dimensional
samples of at least two observations, with integer values in discrete mode.
Every observation is validated before the inclusive `x >= xmin` tail is
selected. No nonfinite observations are silently dropped. See
[fitting](incerto-power-law-fitting.md) and
[goodness of fit](incerto-power-law-goodness-of-fit.md) for statistical assumptions.

(api-powerlaw-mle-continuous)=
### [`powerlaw_mle_continuous`](incerto-api-reference.md#api-powerlaw-mle-continuous)

`powerlaw_mle_continuous(x, xmin) -> float` fits a fixed positive threshold,
using `n_tail / sum(log(x/xmin))`. It needs at least two retained observations
and a positive log-excess sum. An all-cutoff tail raises `ValueError` because
there is no finite MLE. Logarithmic arithmetic avoids forming extreme ratios.

```python
from xmath.incerto.powerlaw import powerlaw_mle_continuous

print(f"alpha: {powerlaw_mle_continuous([2, 3, 5, 9], 2):.3f}")
```

**Output**

```text
alpha: 1.416
```

(api-powerlaw-mle-discrete)=
### [`powerlaw_mle_discrete`](incerto-api-reference.md#api-powerlaw-mle-discrete)

`powerlaw_mle_discrete(x, xmin, *, approximate=False) -> float` fits the
normalized integer mass `k**(-(alpha+1))/zeta(alpha+1,xmin)`. Thresholds and
observations must be positive integers represented exactly, at most
`2**53-1`. The default maximizes the zeta likelihood numerically.
`approximate=True` instead uses the half-integer correction, which can be
poor at small cutoffs or large exponents. The exact fit expands its upper
exponent bracket at most 12 times, with 300 iterations per scalar search;
exponents below about `exp(-16)` and zeta underflow fail explicitly.

```python
from xmath.incerto.powerlaw import powerlaw_mle_discrete

x = [1, 1, 1, 2, 2, 3, 5, 9, 12]
print(f"Exact alpha: {powerlaw_mle_discrete(x, 1):.3f}")
print(f"Approximate alpha: {powerlaw_mle_discrete(x, 1, approximate=True):.3f}")
```

**Output**

```text
Exact alpha: 0.693
Approximate alpha: 0.599
```

(api-ks-distance-powerlaw)=
### [`ks_distance_powerlaw`](incerto-api-reference.md#api-ks-distance-powerlaw)

`ks_distance_powerlaw(x, alpha, xmin, *, discrete=False) -> float` returns
the supremum CDF distance on the retained tail. It checks both sides of
empirical jumps; integer mode also accounts for gaps in the support. It
returns a statistic only, with no fitted-model KS p-value.

```python
from xmath.incerto.powerlaw import ks_distance_powerlaw

for discrete in (False, True):
    d = ks_distance_powerlaw([1, 1, 2, 5], 2, 1, discrete=discrete)
    print(f"discrete={discrete}: KS={d:.3f}")
```

**Output**

```text
discrete=False: KS=0.500
discrete=True: KS=0.332
```

(api-select-xmin)=
### [`select_xmin`](incerto-api-reference.md#api-select-xmin)

`select_xmin(x, *, discrete=False, min_tail=50, candidates=None) -> dict`
refits the exponent and minimizes KS over distinct observed values by default.
An explicit nonempty candidate vector replaces that rule with a fixed grid.
`min_tail` must be an integer at least two. Equal minima choose the lowest
threshold. Results contain `xmin`, `alpha`, `n_tail`, `ks`, `discrete`, and
candidate arrays `xmins`, `alphas`, `counts`, `distances`. Ineligible or
all-cutoff tails have `nan` estimates/distances; an entirely ineligible grid
raises `ValueError`. Numerical failures abort. The default search has
quadratic sample-size cost; declare a smaller grid for large samples.

```python
from xmath.incerto.powerlaw import select_xmin

fit = select_xmin([1, 1, 2, 3, 6], min_tail=2)
print(f"xmin={fit['xmin']:.1f}; alpha={fit['alpha']:.3f}; n={fit['n_tail']}")
print("Counts:", [int(v) for v in fit["counts"]])
```

**Output**

```text
xmin=2.0; alpha=1.995; n=3
Counts: [5, 3, 2, 1]
```

(api-powerlaw-standard-error)=
### [`powerlaw_standard_error`](incerto-api-reference.md#api-powerlaw-standard-error)

`powerlaw_standard_error(alpha, n_tail, *, xmin=1, discrete=False) -> float`
returns the fixed-threshold asymptotic error: `alpha/sqrt(n_tail)` for
continuous data, inverse square-root total Fisher information for integer
data. Discrete information is the variance of log observations, computed
from scaled log-moment sums with an Euler–Maclaurin tail correction.
Refining the sum checks convergence, and a roundoff check rejects unresolved
information with `FloatingPointError`. This error excludes threshold selection,
dependence, and model misspecification. Adding one to the exponent leaves its SE unchanged.

```python
from xmath.incerto.powerlaw import powerlaw_standard_error

for discrete in (False, True):
    se = powerlaw_standard_error(2, 400, xmin=1, discrete=discrete)
    print(f"discrete={discrete}: SE={se:.3f}")
```

**Output**

```text
discrete=False: SE=0.100
discrete=True: SE=0.120
```

(api-sample-powerlaw)=
### [`sample_powerlaw`](incerto-api-reference.md#api-sample-powerlaw)

`sample_powerlaw(alpha, xmin, size, *, discrete=False, random_state=None)`
returns a vector of continuous Pareto or exact integer zeta draws. `size` is
a positive integer; `random_state` accepts a seed or NumPy generator. Continuous
sampling reuses [`pareto_type1`](incerto-api-reference.md#api-pareto-type1).
Discrete sampling inverts the zeta CDF by bracket expansion and integer
bisection; it does not round continuous draws. Nonfinite draws or integer
draws beyond `2**53-1` fail instead of being clipped.

```python
from xmath.incerto.powerlaw import sample_powerlaw

x = sample_powerlaw(2, 3, 5, random_state=381)
k = sample_powerlaw(2, 3, 5, discrete=True, random_state=381)
print("Continuous:", [round(float(v), 3) for v in x])
print("Discrete:", [int(v) for v in k])
```

**Output**

```text
Continuous: [6.84, 12.742, 9.057, 4.72, 7.573]
Discrete: [6, 11, 8, 4, 6]
```

(api-powerlaw-bootstrap-se)=
### [`powerlaw_bootstrap_se`](incerto-api-reference.md#api-powerlaw-bootstrap-se)

`powerlaw_bootstrap_se(x, *, replicates=199, random_state=None,
discrete=False, min_tail=50, candidates=None) -> dict` resamples the whole
empirical sample and refits the threshold and exponent in every replicate.
At least two replicates are required. It returns `fit`, `alpha_se`, `xmin_se`,
`covariance` in `(alpha,xmin)` order, and paired `alphas`, `xmins` arrays.
Standard deviations and covariance use divisor `replicates-1`. Failed
replicates abort with their index. Discontinuous threshold selection can
invalidate naive confidence-interval interpretations of these summaries.

```python
from xmath.incerto.powerlaw import sample_powerlaw, powerlaw_bootstrap_se

x = sample_powerlaw(2, 1, 100, random_state=382)
b = powerlaw_bootstrap_se(x, replicates=8, min_tail=25, random_state=383)
print(f"SE(alpha)={b['alpha_se']:.3f}; SE(xmin)={b['xmin_se']:.3f}")
```

**Output**

```text
SE(alpha)=0.168; SE(xmin)=0.042
```

(api-powerlaw-gof-bootstrap)=
### [`powerlaw_gof_bootstrap`](incerto-api-reference.md#api-powerlaw-gof-bootstrap)

`powerlaw_gof_bootstrap(x, *, replicates=499, random_state=None,
discrete=False, min_tail=50, candidates=None) -> dict` mixes the empirical
below-cutoff body with the fitted tail, with tail probability `n_tail/len(x)`.
Each replicate has the original total size, a random generated tail count,
and its own fitted threshold/exponent. The observed-value candidate rule is
recomputed from the replicate; an explicit grid is reused. Returns `fit`,
`p_value`, `mc_se`, `worst_case_mc_se`, `ks_samples`, `alphas`, `xmins`,
`tail_counts` after refitting, and `generated_tail_counts` before refitting.
The p-value is the fraction with KS at least the observed KS. At least two
replicates are required; failures abort, with none discarded. The worst-case
Monte Carlo SE is `0.5/sqrt(replicates)`; a plug-in `mc_se=0` at an endpoint
does not imply exact knowledge. The tiny example below illustrates the API.

```python
from xmath.incerto.powerlaw import sample_powerlaw, powerlaw_gof_bootstrap

x = sample_powerlaw(2, 1, 100, random_state=384)
b = powerlaw_gof_bootstrap(x, replicates=8, min_tail=25, random_state=385)
print(f"p={b['p_value']:.3f}; worst-case MC SE={b['worst_case_mc_se']:.3f}")
```

**Output**

```text
p=0.375; worst-case MC SE=0.177
```

(api-fit-tail-model)=
### [`fit_tail_model`](incerto-api-reference.md#api-fit-tail-model)

`fit_tail_model(x, xmin, model="powerlaw", *, discrete=False) -> dict`
fits one common conditional tail. Models are `powerlaw`, `lognormal`,
`exponential`, `stretched_exponential` (the full Weibull family), and
`truncated_powerlaw` (an exponential cutoff, no finite upper endpoint).
Discrete mode supports `powerlaw` only and rejects continuous alternatives.
The result contains `model`, `xmin`, `discrete`, `n_tail`, `parameters`,
per-observation `loglik` in retained input order, `loglikelihood`, `success`,
`boundary`, and `status`. Below-cutoff values do not enter the scores.

Parameters are `alpha` for a pure power law; `mu`, `sigma` for the parent
lognormal; `rate` for exponential; `shape`, `scale` for Weibull; and density
exponent `a`, `rate` for the cutoff family. The interior cutoff exponent `a`
can be any real number mathematically. Lognormal/Weibull/cutoff fits use
multi-start numerical optimization; convergence is not a global-MLE proof.

Numerical search boxes use dimensionless coordinates: lognormal
`mu-log(xmin)` in `[-100,100]` and `log(sigma)` in `[-6,5]`; Weibull
`log(shape)` in `[-5,5]`, profiling out scale. The cutoff search starts with
`a-1` in `[-20,100]` and `log(rate*xmin)` covering both `[-18,8]` and the
sample's inverse mean excess scale. It expands the bounds over at most eight
search rounds; log rates stay within `[-700,700]`. Each start permits 2000
iterations. An unresolved search or artificial bound sets `success=False`.
These bounds are numerical limits, not definitions of the families.
The pure-law boundary `rate=0` is accepted only after verifying its inward
score, returning `boundary=True` with status `pure-power-law boundary`.
An inferior numerical fit can return the pure law with `success=False` and
an explicit uncertified-boundary status. Cutoff quadrature is centered and
scaled around its mode; unresolved integration raises or fails the fit.
Exponential fitting accepts constant observations above the threshold when
their mean excess is positive. Lognormal, Weibull, and cutoff fitting reject
constant tails.

```python
from xmath.incerto.powerlaw import fit_tail_model

fit = fit_tail_model([0.5, 1, 2, 3, 6], 1, "exponential")
print(f"Retained: {fit['n_tail']}; rate: {fit['parameters']['rate']:.3f}")
print(f"Conditional log likelihood: {fit['loglikelihood']:.3f}")
```

**Output**

```text
Retained: 4; rate: 0.500
Conditional log likelihood: -6.773
```

(api-likelihood-ratio-test)=
### [`likelihood_ratio_test`](incerto-api-reference.md#api-likelihood-ratio-test)

`likelihood_ratio_test(loglik1, loglik2, *, nested=False,
calibration=None, df=1) -> dict` accepts paired finite real score vectors of
length at least two, from the same ordered observations and reference measure.
Positive `R=sum(loglik1-loglik2)` favors model 1. Non-nested mode returns
`normalized_R=R/(sqrt(n)*std(d,ddof=0))` and a two-sided normal `p_value`.
Zero contrast variance gives `nan` statistic/p with an explanatory status.
Results also contain `n`, `statistic`, and `status`; `statistic` is the nested
LR statistic and is `nan` for non-nested comparisons.

For `nested=True`, model 1 must be the smaller model. Model 2 fitting worse
beyond numerical tolerance raises `ValueError`. The statistic is `-2R`, while
the default p-value is `nan` pending calibration. Explicit `calibration="chi2"`
uses a chi-square reference with positive integer `df`;
`calibration="boundary"` requires `df=1` and uses the regular half-point-mass,
half-chi-square reference, with inclusive p-value one at statistic zero.
These options assert conditions; they do not diagnose them. See the
[boundary and infinite-information caveats](incerto-power-law-goodness-of-fit.md#csn-nested-cutoff).
No automatic AIC/BIC correction, multiple-testing adjustment, dependence
correction, or selected-threshold calibration is performed.

```python
from xmath.incerto.powerlaw import likelihood_ratio_test

lr = likelihood_ratio_test([1, 2, 5, 1], [2, 1, 1, 0])
print(f"R={lr['R']:.3f}; normalized R={lr['normalized_R']:.3f}")
print(f"Two-sided p: {lr['p_value']:.3f}")
```

**Output**

```text
R=5.000; normalized R=1.400
Two-sided p: 0.161
```
