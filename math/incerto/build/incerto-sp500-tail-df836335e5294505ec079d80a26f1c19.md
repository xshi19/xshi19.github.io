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

The [chapter-10 diagnostics](incerto-sp500.md) reuse this cache for
aggregation, drawdowns, moment concentration, records, and gain/loss
comparisons. The [pipeline note](incerto-ml-econometrics.md) adds a frozen
chronological risk-estimator comparison.

(sp500-data)=
## Data source and window

The observed calculation uses **Yahoo Finance `^GSPC` daily Close**, fetched
locally on **2026-10-30** with `auto_adjust=False` [@yahoo2026gspc]. It is a
**price index, not total return**: dividends are not reinvested. The default
return window is **1928-01-03 through 2026-10-29**, comprising **24,824**
log returns. The cache has **24,825** closes, beginning with the
**1927-12-30** anchor. The supplied **2026-10-30 intraday row is dropped**.
Of the returns, **11,493** are losses, **13,018** are gains and **313** are
zero. The Hill and mean-excess calculations below condition on loss days.

The full available history retains early crises and 1987. This choice also
pools different regimes and historical index coverage, so the estimates
describe this sample rather than a stationary population. Only derived
statistics and plots are distributed. The raw cache is not redistributed
under Yahoo's source-use terms [@yahoo2026terms]. Site rendering uses the
recorded outputs and SVGs; it does not fetch or read prices.

Prepare the local cache with the script, using the same inclusive cutoff:

```sh
python scripts/fetch_incerto_data.py sp500 --source /path/to/authorized.csv --end-date 2026-10-29
# Fresh Yahoo fetch, after installing the optional dependency:
uv sync --extra yahoo
python scripts/fetch_incerto_data.py sp500 --provider yahoo --end-date 2026-10-29
# FRED remains the default and an explicit alternate (shorter history):
python scripts/fetch_incerto_data.py sp500 --provider fred --end-date 2026-10-29
```

These commands write `data/incerto/cache/sp500_daily.csv` in an editable
checkout. The directory is ignored by Git. `INCERTO_DATA_DIR` can select
another data root; a wheel installation defaults to the user's cache.
FRED's SP500 price index is an alternate source [@fred2026sp500], not the
input to the results below. A new fetch can revise history; the dated
local snapshot determines the recorded output. Without that cache, the
page's empirical snippet test skips with a data-availability reason.

## Reproducible method

Use [`load_sp500_prices`](incerto-api-reference.md#api-load-sp500-prices),
[`hill_alpha_estimator`](incerto-api-reference.md#api-hill-alpha-estimator),
[`mean_excess`](incerto-api-reference.md#api-mean-excess), and
[`PriceSeries.negative_log_returns`](incerto-api-reference.md#api-priceseries-negative-log-returns).
The [Yahoo writer](incerto-api-reference.md#api-write-sp500-yahoo-cache)
handles the optional download separately from this calculation.

:::{dropdown} Code: run this calculation

```python
import numpy as np
from xmath.incerto.datasets import load_sp500_prices
from xmath.incerto.estimators import hill_alpha_estimator, mean_excess

prices = load_sp500_prices(fetch=False, end_date="2026-10-29")
_, losses = prices.negative_log_returns()
print(f"Return window: {prices.date[1]} to {prices.date[-1]}")
print(f"Closes: {len(prices.close)}; returns: {len(prices.close) - 1}; loss days: {len(losses)}")
for k in (25, 50, 100, 200, 400):
    if k < len(losses):
        print(f"Hill k={k}: alpha={hill_alpha_estimator(losses, k):.2f}")
thresholds = np.quantile(losses, [0.8, 0.9, 0.95])
result = mean_excess(losses, thresholds, min_exceedances=5)
for u, excess, count in zip(result["threshold"], result["mean_excess"], result["exceedances"]):
    print(f"threshold={100 * u:.2f}% log, mean excess={100 * excess:.2f}% log, exceedances={count:.0f}")
```

**Output**

```text
Return window: 1928-01-03 to 2026-10-29
Closes: 24825; returns: 24824; loss days: 11493
Hill k=25: alpha=4.50
Hill k=50: alpha=3.43
Hill k=100: alpha=3.09
Hill k=200: alpha=2.88
Hill k=400: alpha=2.89
threshold=1.18% log, mean excess=1.03% log, exceedances=2299
threshold=1.76% log, mean excess=1.23% log, exceedances=1150
threshold=2.49% log, mean excess=1.43% log, exceedances=575
```
:::

The Hill exponent falls from **4.50 at $k=25$** to **2.89 at $k=400$**;
the intermediate estimates are **3.43, 3.09, 2.88**. The final two are
close, but this does not identify a stable population exponent. The broader
5%, 10%, and 20% loss-tail fractions on the
[diagnostics page](incerto-sp500.md) give **2.59, 2.26, 1.94**, showing how
far the estimate changes as more of the distribution enters the fit.

At the loss sample's 80th, 90th, and 95th percentiles, thresholds are
**1.18%, 1.76%, and 2.49% in log-loss units**. Mean excesses are
**1.03%, 1.23%, and 1.43%**, from **2,299, 1,150, and 575** strict
exceedances. Increasing mean excess across these three thresholds is
descriptive evidence; it does not establish an exactly linear Pareto
mean-excess function.

`uv run python demos/incerto/sp500_tail.py` prints these diagnostics and
regenerates the derived tail figure. Its grids are fixed before fitting.

```{figure} ../assets/figures/incerto/sp500-observed-tail.svg
:alt: Three stacked panels show conditional gain and loss survival on fixed thresholds, loss Hill estimates as the retained count changes, and loss mean excess on fixed quantiles with selected exceedance counts.

Yahoo Close-derived returns, 1928-01-03–2026-10-29. Survival is the strict
fraction above each of 80 fixed thresholds from 0.1% to 30% in log-change
units, conditional on the indicated sign; zero fractions are omitted from
log axes. Hill uses 100 integer counts spanning 25–2,298; mean excess uses
30 loss quantiles spanning 80%–98%, with counts marked at 80%, 90%, and 95%.
These are diagnostic curves, not confidence bands or accepted tail models.
```

For comparison, the rank-based plotting position at sorted losses $\ell_{(i)}$ is
$1-i/(n+1)$; it keeps plotted probabilities positive at the largest loss.
The strict empirical survival at a threshold $u$ is instead
$n^{-1}\sum_{j=1}^n\mathbf1_{\{\ell_j>u\}}$, which handles ties and is zero
at the sample maximum. The figure above uses **strict survival**, not rank
plotting positions. Inspect the
survival plot on log-log axes alongside the Hill curve and
[mean excess](incerto-mean-excess-function.md). Report the number of
observations, threshold, $k$, sample window, and transformation with each
estimate. [Threshold selection](incerto-tail-threshold-selection.md) discusses
the bias and sampling variation that change as the threshold moves.

## Interpretation and caveats

A straight segment in a survival plot does not establish a Pareto tail.
The [peaks-over-threshold theorem](incerto-pickands-balkema-de-haan.md)
requires a domain-of-attraction assumption; it does not verify that assumption
for equity returns. Neither independence nor stationarity is established
for this pooled history. Treating loss days as iid would ignore their
chronological spacing and volatility changes. No iid standard errors or
moment-existence verdicts are attached to these fits. A fitted exponent
above four at one threshold and below two at another cannot settle the
existence of the population variance or fourth moment.

A fitted exponent is conditional on this dated sample and threshold;
it is not a forecast of the next daily loss. The
[half-window diagnostics](incerto-sp500.md#sp500-local-run) expose further
sensitivity to which decades are included.

## References

- @yahoo2026gspc, locally fetched price-index Close history.
- @yahoo2026terms, source-use terms; no redistribution of raw data.
- @fred2026sp500, S&P 500 through FRED.
- @hill1975simple, tail-index estimation.
- @davison1990models, threshold exceedances.
