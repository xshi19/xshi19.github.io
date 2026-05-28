---
kernelspec:
  name: python3
  display_name: Python 3
---

# S&P 500 Tail Fit

## Statement

This empirical example applies one-sided tail diagnostics to daily S&P 500
close-to-close log losses,

$$
L_t=-\log(P_t/P_{t-1})\mathbf 1_{\{P_t<P_{t-1}\}},
$$

where $P_t$ is the closing index level.  The goal is modest: inspect whether
large daily losses behave more like a thin-tail sample or a heavy right-tail
sample over the observed window.

The raw index series is not committed to the repository.  The loader reads an
untracked local cache at `data/cache/sp500_daily.csv`; the refresh script can
populate that cache from FRED or from a locally obtained CSV.  This keeps the
site source reproducible without redistributing S&P index data.

## Intuition

Financial returns are two-sided, so a right-tail method has to be applied to a
one-sided transformation.  Here the transformed variable is daily loss.  A
large positive value of $L_t$ means the index fell sharply from one close to
the next.

The plot and estimates below are diagnostics, not a trading model.  The point
is to connect empirical losses to the wiki's tail machinery: survival plots,
Hill stability, mean excess, and threshold exceedances.

## Proof

There is no theorem proving that equity-index losses follow a Pareto law.  The
computation has three reproducible steps:

1. Clean daily closes into a date/close cache.
2. Transform closes into positive log losses.
3. Apply the [Hill Estimator](../methods/hill-estimator.md) and
   [Mean Excess Function](../theorems/mean-excess-function.md) to the loss
   sample.

Any claim made from this page is therefore empirical and conditional on the
sample window, the one-day horizon, and the threshold choices.

## Python

Run `python scripts/fetch_data.py sp500` to fetch the FRED path, or
`python scripts/fetch_data.py sp500 --source path/to/local.csv` if you already
have a licensed local CSV.  The code cell skips gracefully when the cache is
not present.

```{code-cell} python
:label: sp500-tail-python-check

import numpy as np
import matplotlib.pyplot as plt

from incerto.datasets import DatasetNotAvailable, load_sp500_prices
from incerto.estimators import hill_alpha_estimator, hill_stability, mean_excess
from incerto.figures import COLORS, FIGURE_SIZES, set_theme, style_axes

set_theme()

try:
    prices = load_sp500_prices()
except DatasetNotAvailable as exc:
    print(exc)
else:
    loss_dates, losses = prices.negative_log_returns()
    print(
        f"{prices.name}: {prices.date[0]} to {prices.date[-1]}, "
        f"{prices.close.size:,} closes, {losses.size:,} negative-return days"
    )

    order = np.sort(losses)
    survival = 1 - np.arange(1, order.size + 1) / (order.size + 1)

    ks = np.unique(
        np.geomspace(10, min(500, max(11, losses.size // 3)), 80).astype(int)
    )
    stability = hill_stability(losses, ks)

    thresholds = np.quantile(losses, np.linspace(0.80, 0.99, 35))
    me = mean_excess(losses, thresholds, min_exceedances=20)

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["three_panel"])

    axes[0].loglog(order, survival, color=COLORS["accent"])
    axes[0].set_xlabel("daily log loss")
    axes[0].set_ylabel("empirical survival")
    axes[0].set_title("Loss survival")

    axes[1].semilogx(stability["k"], stability["alpha"], color=COLORS["umber"])
    axes[1].set_xlabel("upper order statistics k")
    axes[1].set_ylabel("Hill alpha")
    axes[1].set_title("Hill stability")

    axes[2].plot(me["threshold"], me["mean_excess"], color=COLORS["green"])
    axes[2].set_xlabel("threshold u")
    axes[2].set_ylabel("mean excess")
    axes[2].set_title("Mean excess")

    style_axes(axes, grid_axis="both")
    fig.tight_layout()
    plt.show()

    for k in [25, 50, 100, 200, 400]:
        if k < losses.size:
            print(f"Hill alpha for k={k:3d}: {hill_alpha_estimator(losses, k):.2f}")

    largest = np.argsort(losses)[-5:][::-1]
    print("Largest one-day log losses:")
    for i in largest:
        print(f"  {loss_dates[i]}: {losses[i]:.3f}")
```

The Hill estimates should be read as a sensitivity range, not a single fitted
truth.  The largest observations usually identify crisis days, which is exactly
why iid threshold models need caveats for dependence and volatility clustering.

## Caveats

- The S&P 500 cache is local and untracked because the underlying index data
  has redistribution restrictions.  Do not commit raw S&P index levels without
  checking the license.
- Daily index losses are dependent and volatility-clustered.  A plain iid tail
  fit ignores that structure.
- The analysis uses price-index closes, not total returns with dividends.
- Threshold choice dominates finite-sample tail estimates.  Report $k$,
  threshold, sample window, and transformation.
- A heavy-tail diagnostic is not a forecast of tomorrow's loss.

## References

- S&P 500 data as made available through FRED, Federal Reserve Bank of St.
  Louis [@fred2026sp500].
- Hill, "A Simple General Approach to Inference About the Tail of a
  Distribution" [@hill1975simple].
- Davison and Smith, "Models for Exceedances over High Thresholds"
  [@davison1990models].
- Taleb, *Statistical Consequences of Fat Tails* [@taleb2020scoft].

## Backlinks

- Depends on: [Hill Estimator](../methods/hill-estimator.md),
  [Mean Excess Function](../theorems/mean-excess-function.md), and
  [Pickands-Balkema-de Haan Theorem](../theorems/pickands-balkema-de-haan.md).
- Used by: [Chapter 3 reading guide](../../reading-guides/taleb-scoft/ch3.md)
  and future applied tail-risk examples.

<!-- incerto-provenance:start -->
:::{div}
:class: incerto-provenance

**Provenance.** Source: `content/concepts/examples/sp500-tail.md`. Last verified: 2026-05-24. Checked against cited sources, page proof or computation, and executable examples.
:::
<!-- incerto-provenance:end -->
