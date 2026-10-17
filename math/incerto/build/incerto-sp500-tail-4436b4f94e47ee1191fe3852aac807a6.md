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
S&P Dow Jones Indices series supplied through FRED [@fred2026sp500].
Financial data to be provided later via the Portfolio Manager bot.

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

Run `python demos/incerto/sp500_tail.py`. Without data, it prints an
availability message and exits successfully. With data, it reports the
sample dates, loss count, Hill estimates, and mean excess across thresholds.
It does not publish index data or produce figures during the site build.

```python
import numpy as np
from xmath.incerto.datasets import load_sp500_prices
from xmath.incerto.estimators import hill_alpha_estimator, mean_excess

prices = load_sp500_prices(fetch=False)
dates, losses = prices.negative_log_returns()
for k in (25, 50, 100, 200, 400):
    if k < len(losses):
        print(k, hill_alpha_estimator(losses, k))
if len(losses) >= 20:
    thresholds = np.quantile(losses, [0.8, 0.9, 0.95])
    print(mean_excess(losses, thresholds, min_exceedances=5))
```

The empirical survival estimate at sorted losses $\ell_{(i)}$ is
$1-i/(n+1)$. Inspect it on log-log axes alongside the Hill curve and
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

- S&P Dow Jones Indices, S&P 500 through FRED [@fred2026sp500].
- Hill, tail-index estimation [@hill1975simple].
- Davison and Smith, threshold exceedances [@davison1990models].

## Source and license

Adapted from `xshi19/incerto-wiki`, `content/concepts/examples/sp500-tail.md`,
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
