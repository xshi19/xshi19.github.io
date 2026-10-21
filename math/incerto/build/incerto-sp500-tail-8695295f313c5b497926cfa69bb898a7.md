---
options:
  concept:
    id: sp500-tail
    type: empirical-example
    depends_on:
    - hill-estimator
    - tail-threshold-selection
    - mean-excess-function
    - pickands-balkema-de-haan
    tags:
    - finance
    - tail-risk
title: S&P 500 tail diagnostics
---

For closing index levels $P_t>0$, define a daily log loss by

$$
\ell_t=-\log(P_t/P_{t-1}).
$$

The [Hill estimator](incerto-hill-estimator.md) uses only $\ell_t>0$;
zero-return and rising-price days are excluded. Here $P_t$ denotes an index
level, not the probability law $P$ in the
[shared notation](https://xshi19.github.io/math/notation/).

## Data availability

No S&P index levels or derived empirical figures are distributed with this
page. The source repository records redistribution restrictions for the
S&P 500 series supplied through FRED [@fred2026sp500].
Use a local data file whose terms permit your intended analysis.

The method below works with a locally supplied cache. Site builds do not
read that cache or make network requests. To prepare the cache explicitly:

```sh
python scripts/fetch_incerto_data.py sp500 --source /path/to/authorized.csv
# Alternatively, where source terms permit downloading for local use:
python scripts/fetch_incerto_data.py sp500
```

Both commands write `data/incerto/cache/sp500_daily.csv` in an editable
checkout. The directory is ignored by Git. `INCERTO_DATA_DIR` can select
another data root; a wheel installation defaults to the user's cache.

## Reproducible method

Use [`DatasetNotAvailable`](incerto-api-reference.md#api-datasetnotavailable), [`load_sp500_prices`](incerto-api-reference.md#api-load-sp500-prices), [`hill_alpha_estimator`](incerto-api-reference.md#api-hill-alpha-estimator), [`mean_excess`](incerto-api-reference.md#api-mean-excess), [`PriceSeries.negative_log_returns`](incerto-api-reference.md#api-priceseries-negative-log-returns).

:::{dropdown} Code: run this calculation

```python
import numpy as np
from xmath.incerto.datasets import DatasetNotAvailable, load_sp500_prices
from xmath.incerto.estimators import hill_alpha_estimator, mean_excess

try:
    prices = load_sp500_prices(fetch=False)
except DatasetNotAvailable:
    print("No local S&P cache; supply an authorized price file to run this example.")
else:
    dates, losses = prices.negative_log_returns()
    print(f"Window / loss days: {prices.date[0]} {prices.date[-1]} {len(losses):d}")
    for k in (25, 50, 100, 200, 400):
        if k < len(losses):
            print(f"{k:d} {hill_alpha_estimator(losses, k):.4g}")
    if len(losses) >= 20:
        thresholds = np.quantile(losses, [0.8, 0.9, 0.95])
        for name, values in mean_excess(losses, thresholds, min_exceedances=5).items():
            fmt = ".0f" if name == "exceedances" else ".3e"
            print(f"{name}: [" + ", ".join(format(v, fmt) for v in values) + "]")
```

**Output**

```text
No local S&P cache; supply an authorized price file to run this example.
```
:::

The output above is for an absent local cache; results with a cache depend
on that local sample.
The same diagnostics are available as `uv run python demos/incerto/sp500_tail.py`.

The rank-based plotting position at sorted losses $\ell_{(i)}$ is
$1-i/(n+1)$; it keeps plotted probabilities positive at the largest loss.
The strict empirical survival at a threshold $u$ is instead
$n^{-1}\sum_{j=1}^n\mathbf1_{\{\ell_j>u\}}$, which handles ties and is zero
at the sample maximum. State which convention is plotted. Inspect the
survival plot on log-log axes alongside the Hill curve and
[mean excess](incerto-mean-excess-function.md). Report the number of
observations, threshold, $k$, sample window, and transformation with each
estimate. [Threshold selection](incerto-tail-threshold-selection.md) discusses
the bias and sampling variation that change as the threshold moves.

## Interpretation and caveats

A straight segment in a survival plot does not establish a Pareto tail.
The [peaks-over-threshold theorem](incerto-pickands-balkema-de-haan.md)
requires a domain-of-attraction assumption; it does not verify that assumption
for equity returns. Daily losses are dependent and volatility-clustered,
whereas a basic Hill calculation treats observations as iid. These diagnostics
also use price-index closes rather than dividend-adjusted total returns.

No numerical conclusion about S&P losses is asserted here without a dated
local sample. A fitted exponent is conditional on that sample and threshold;
it is not a forecast of the next daily loss.

## References

- @fred2026sp500, S&P 500 through FRED.
- @hill1975simple, tail-index estimation.
- @davison1990models, threshold exceedances.
