---
title: S&P 500 — what return diagnostics can establish
options:
  concept:
    id: sp500
    type: empirical-example
    depends_on:
      - sp500-tail
      - kappa
      - max-to-sum-ratio
      - moment-estimation
    tags:
      - finance
      - diagnostics
---

Return diagnostics ask different questions. Aggregation examines changes with
the holding period; a drawdown uses the order of prices; a tail fit uses large
observations. Agreement among them can motivate a model comparison, but no
finite collection of these plots proves that a population moment is infinite.

This page develops the diagnostics in SCoFT v4, chapter 10, §§10.1–10.3,
printed pp. 193–205 [@taleb2025scoftv4]. The
[existing S&P tail page](incerto-sp500-tail.md) owns data access and the
Hill/mean-excess workflow. Here we add chronological and aggregation checks.
The [pipeline note](incerto-ml-econometrics.md) examines what these checks
mean for a risk estimator.

## Data, clock, and estimands

No observed S&P sample or empirical result is distributed with this page.
The code reads the same authorized local cache as the existing tail page,
with fetching disabled. FRED's SP500 series is a daily closing **price**
index, without dividends; its redistribution restrictions remain applicable
[@fred2026sp500]. The currently documented FRED window contains ten years
of history, so it cannot reproduce the source's much longer historical
experiment or its discussion of 1987.

For strictly positive, chronologically ordered closes $P_t$, define

$$
\begin{gathered}
r_t=\log P_t-\log P_{t-1},\\
\ell_t=-r_t.
\end{gathered}
$$

Here $P_t$ is an index level; $P$ without a time subscript denotes probability,
as in the [notation canon](https://xshi19.github.io/math/notation/).
All returns, including zero returns, remain in aggregation and record counts.
Only the conditional gain/loss fits select $r_t>0$ or $\ell_t>0$.
Joining loss days before aggregation would change the time interval.

The loader's cleaned cache omits unavailable closes. We do not fill gaps:
one return joins consecutive **available** closes, which can span a weekend,
holiday, or missing observation. The report includes the largest calendar
gap and rejects unordered or duplicated dates. A gap count alone cannot
identify missing trading sessions. No total-return adjustment, intraday
sampling, or volatility standardization is performed.

(sp500-aggregation)=
## Kurtosis under aggregation

For a nonconstant finite sample $x_1,\ldots,x_N$, let $\widehat m_q$ be its
central moment with divisor $N$. We use excess kurtosis

$$
\begin{gathered}
\widehat m_q=\frac1N\sum_{j=1}^N(x_j-\bar x)^q,\\
\widehat\gamma_2=\frac{\widehat m_4}{\widehat m_2^2}-3.
\end{gathered}
$$

There is no finite-sample bias correction. Pearson kurtosis is this number
plus three. An undefined population fourth moment does not prevent the
sample statistic from returning a finite number.

**Aggregation identity.** Suppose $X_i$ are iid, have variance
$0<\sigma^2<\infty$, and have a finite fourth central moment $\mu_4$.
For $S_h=\sum_{i=1}^hX_i$, the population excess kurtosis satisfies

$$
\gamma_2(S_h)=\frac{\gamma_2(X_1)}h.
$$

We prove this identity, not a convergence claim about a particular market
sample. Set $Z_i=X_i-\mathbb E[X_i]$. In the expansion of the fourth
power of the sum, independence and zero means eliminate terms with a
singleton index. The surviving terms give

$$
\begin{gathered}
\mathbb E\!\left[\left(\sum_{i=1}^h Z_i\right)^4\right]\\
=h\mu_4+3h(h-1)\sigma^4,\\
\operatorname{Var}(S_h)=h\sigma^2.
\end{gathered}
$$

Dividing by $h^2\sigma^4$ and subtracting three proves the formula.
Dependence can introduce additional terms. Finite samples also lose blocks
as $h$ increases. A slowly falling or rising sample curve therefore does
not, by itself, distinguish infinite fourth moments from dependence,
changing volatility, or sampling variation.

Use [`aggregate_returns`](incerto-api-reference.md#api-aggregate-returns),
[`aggregation_diagnostics`](incerto-api-reference.md#api-aggregation-diagnostics),
and [`sample_excess_kurtosis`](incerto-api-reference.md#api-sample-excess-kurtosis).

```python
from xmath.incerto.diagnostics import aggregate_returns, sample_excess_kurtosis

print("Disjoint:", ", ".join(f"{x:g}" for x in aggregate_returns([1, -2, 4, -8, 16], 2)))
print("Overlapping:", ", ".join(f"{x:g}" for x in aggregate_returns([1, -2, 4, -8, 16], 2, overlap=True)))
for sample in ([2, 2], [-1, 1], [-3, -1, 0, 2, 8]):
    print(f"Excess kurtosis: {sample_excess_kurtosis(sample):.4g}")
```

**Output**

```text
Disjoint: -1, -4
Overlapping: -1, 2, -4, 8
Excess kurtosis: nan
Excess kurtosis: -2
Excess kurtosis: -0.531
```

```{figure} ../assets/figures/incerto/diagnostics-aggregation.svg
:name: sp500-aggregation-figure
:alt: Two stacked panels show sample excess kurtosis across aggregation sizes for normal and Student t with three degrees of freedom, with broad sampling ranges.

Synthetic iid samples: 128 paths of length 4,096 per parent, PCG64 seed 1010.
Lines are sample medians; shading spans the 10th–90th percentiles, not
confidence intervals. Blocks are disjoint. The normal population excess
kurtosis is zero; Student t(3) has no finite fourth moment. Declining sample
values in the lower panel do not supply that missing moment.
```

The [kappa note](incerto-kappa.md) compares mean absolute deviations of
sums. Our report plugs the daily and aggregated **sample** mean absolute
deviations into
[`kappa_from_mads`](incerto-api-reference.md#api-kappa-from-mads).
Each horizon uses its own sample mean; the daily baseline uses all returns.
The result is descriptive. It neither estimates a unique tail exponent
nor carries an iid confidence interval for dependent market returns.

```python
from xmath.incerto.diagnostics import aggregation_diagnostics

for sample in ([1, 1, -1, -1], [1, -1, 1, -1], [0, 0, 0, 0]):
    row = aggregation_diagnostics(sample, (2,))[0]
    print(f"blocks={row['blocks']}, MAD={row['mad']:.3f}, kappa={row['kappa']:.3f}")
```

**Output**

```text
blocks=2, MAD=2.000, kappa=1.000
blocks=2, MAD=0.000, kappa=nan
blocks=2, MAD=0.000, kappa=nan
```

(sp500-drawdown)=
## Drawdowns and conditional tail means

Define the running price peak $P_j^\star$ and maximum fractional drawdown $D_n$ by

$$
\begin{gathered}
P_j^\star=\max_{0\le i\le j}P_i,\\
D_n=\max_{0\le j\le n}\left(1-\frac{P_j}{P_j^\star}\right).
\end{gathered}
$$

This equals the largest fractional decline across ordered peak/trough pairs:
for fixed $j$, the largest previous $P_i$ maximizes the decline. Including
$i=j$ makes the statistic nonnegative. This positive fractional convention
differs from a negative log-drawdown convention. A distribution of drawdowns
also depends on the window and overlap rules; a whole-series value is one
path statistic, not a sample of iid losses.

For a positive threshold $u$ with $P(X>u)>0$, finite conditional mean, and
mean excess $e(u)$, define

$$
\begin{gathered}
\phi(u)=\frac{\mathbb E[X\mid X>u]}u,\\
\phi(u)=1+\frac{e(u)}u.
\end{gathered}
$$

The second line follows by subtracting $u$ inside the conditional
expectation. Thus $\phi(u)>1$; it cannot tend to zero. For an exact
[Pareto law](incerto-pareto.md), $u\ge x_m$ and $\alpha>1$ give
$\phi(u)=\alpha/(\alpha-1)$. A unit exponential instead gives
$\phi(u)=1+1/u\to1$. These are model identities, not a characterization
of every possible tail class. For the left tail, apply the same definition
to $-r_t$ and condition on $-r_t>u$.

Use [`maximum_drawdown`](incerto-api-reference.md#api-maximum-drawdown),
[`conditional_tail_ratio`](incerto-api-reference.md#api-conditional-tail-ratio),
and its underlying [`mean_excess`](incerto-api-reference.md#api-mean-excess).

```python
from xmath.incerto.diagnostics import maximum_drawdown, conditional_tail_ratio

print(f"Drawdown: {maximum_drawdown([100, 120, 90, 140, 84, 110]):.3f}")
result = conditional_tail_ratio([-4, 0, 2, 2, 4, 8], [2, 4, 8], min_exceedances=2)
for count, ratio in zip(result["exceedances"], result["ratio"]):
    print(f"exceedances={count:.0f}, conditional mean / threshold={ratio:.3f}")
```

**Output**

```text
Drawdown: 0.400
exceedances=2, conditional mean / threshold=3.000
exceedances=1, conditional mean / threshold=nan
exceedances=0, conditional mean / threshold=nan
```

```{figure} ../assets/figures/incerto/diagnostics-drawdown-ratio.svg
:alt: Reordering six fixed prices changes maximum drawdown; a separate panel compares constant Pareto and decreasing exponential conditional-mean ratios.

The upper panel uses invented prices and shows order sensitivity. The lower
panel shows exact population ratios for a unit-minimum Pareto with exponent
1.5 and a unit exponential. Neither panel contains S&P observations.
```

(sp500-ms)=
## Fourth-moment concentration and the MS plot

For $p>0$, the raw absolute-power maximum-to-sum share is

$$
Q_n^{(p)}=\frac{\max_{i\le n}|X_i|^p}
{\sum_{i=1}^n|X_i|^p}.
$$

The [finite-mean max-to-sum argument](incerto-max-to-sum-ratio.md) proves
$Q_n^{(p)}\to0$ almost surely under iid sampling and
$0<\mathbb E|X|^p<\infty$. Apply that argument to $|X|^p$; its
nonnegativity and finite positive mean are exactly the needed assumptions.
An all-zero finite prefix has no defined ratio.

A high $Q_n^{(4)}$ says one observation dominates the raw fourth-power
total. It is not literally that observation's share of *central kurtosis*:
recentering and the variance denominator change when an observation is
removed. The report keeps those quantities separate and repeats them on
the first and second halves of the available series.

Finite-path instability cannot identify an infinite moment. It also cannot
establish regular variation. Even at the population level, an infinite
moment need not imply regular variation: the survival

$$
\begin{gathered}
\bar F(x)=x^{-1}\left(1+\tfrac14\sin(\log x)\right),\\
x\ge1
\end{gathered}
$$

is decreasing, since its derivative has the sign of
$-1+(\cos(\log x)-\sin(\log x))/4<0$. It starts at one and is bounded
below by $3/(4x)$, so its mean is infinite by survival integration.
But $\bar F(e^\pi x)/\bar F(x)$ oscillates without a limit, ruling out
regular variation. This is an explicit counterexample to treating missing
moments and power tails as equivalent classes.

Use [`moment_share_path`](incerto-api-reference.md#api-moment-share-path).

```python
from xmath.incerto.diagnostics import moment_share_path

for power in (1, 2, 4):
    values = moment_share_path([0, 1, -2, 8], power)
    print(f"p={power}: " + ", ".join(f"{v:.3f}" for v in values))
```

**Output**

```text
p=1: nan, 1.000, 0.667, 0.727
p=2: nan, 1.000, 0.800, 0.928
p=4: nan, 1.000, 0.941, 0.996
```

(sp500-records)=
## Records, extrema, and the two tails

A strict upper record occurs when $X_j$ exceeds every earlier observation.
The first observation counts as a record. Let $N_n^+$ denote the cumulative
count; define $N_n^-$ using strict new minima. For iid observations from a
continuous distribution, each side separately satisfies

$$
\begin{gathered}
H_n=\sum_{j=1}^n\frac1j,\\
\mathbb E[N_n^+]=H_n,\\
\operatorname{Var}(N_n^+)
=\sum_{j=1}^n\frac1j\left(1-\frac1j\right).
\end{gathered}
$$

Here is a proof of the expectation and variance. All $n!$ rank orders are
equally likely. The sequence of insertion ranks has $j$ choices at step
$j$ and gives a bijection to those orders. Its components are therefore
independent and uniform on $\{1,\ldots,j\}$. An upper record occurs at
rank $j$, with probability $1/j$. Adding independent Bernoulli means and
variances proves both formulas. Lower records use rank one and the same
argument; upper and lower counts are **not** mutually independent.

Rounded data can contain ties, which do not create strict records. The
continuous-iid benchmark then need not apply. Volatility clustering and
nonstationarity also change the reference experiment. Agreement with $H_n$
cannot establish independence, and independence alone does not establish
a [maximum domain of attraction](incerto-generalized-extreme-value.md).

Use [`record_counts`](incerto-api-reference.md#api-record-counts).

```python
from xmath.incerto.diagnostics import record_counts

records = record_counts([2, 2, 1, 3, 3, 0])
for key in ("upper", "lower", "iid_mean", "iid_variance"):
    print(key + ": " + ", ".join(f"{v:.3f}" for v in records[key]))
```

**Output**

```text
upper: 1.000, 1.000, 1.000, 2.000, 2.000, 2.000
lower: 1.000, 1.000, 2.000, 2.000, 2.000, 3.000
iid_mean: 1.000, 1.500, 1.833, 2.083, 2.283, 2.450
iid_variance: 0.000, 0.250, 0.472, 0.660, 0.820, 0.959
```

```{figure} ../assets/figures/incerto/diagnostics-moments-records.svg
:alt: Fourth-power concentration along normal and Student paths; upper and lower Student record counts compared with the harmonic expectation.

Two synthetic iid paths of length 3,000, PCG64 seed 1011. The MS curves use
raw absolute fourth powers; the record panel uses the same Student t(3)
path. All continuous iid parents share the harmonic record-count expectation,
despite having different moment properties. These are individual paths.
```

For gain/loss asymmetry, compare positive $r_t$ with positive $-r_t$, using
the same threshold rules and showing each side's retained counts. Our report
uses [`hill_alpha_estimator`](incerto-api-reference.md#api-hill-alpha-estimator)
at three retained fractions and conditional means at three quantiles.
The two sides' quantile thresholds need not have equal magnitudes.
[Threshold sensitivity](incerto-tail-threshold-selection.md) and
[model comparison](incerto-power-law-goodness-of-fit.md) remain necessary;
a few different fitted exponents do not establish population asymmetry.

(sp500-local-run)=
## Run on the existing local cache

Use [`DatasetNotAvailable`](incerto-api-reference.md#api-datasetnotavailable),
[`load_sp500_prices`](incerto-api-reference.md#api-load-sp500-prices), and
[`sp500_diagnostics`](incerto-api-reference.md#api-sp500-diagnostics).
The last helper reuses [`PriceSeries.log_returns`](incerto-api-reference.md#api-priceseries-log-returns).

:::{dropdown} Code: inspect a local return series

```python
from xmath.incerto.datasets import DatasetNotAvailable, load_sp500_prices
from xmath.incerto.diagnostics import sp500_diagnostics

try:
    prices = load_sp500_prices(fetch=False)
except DatasetNotAvailable:
    print("No local S&P cache; empirical diagnostics remain uncomputed.")
else:
    result = sp500_diagnostics(prices)
    print(f"Window: {result['start']} to {result['end']}; returns={result['returns']}")
    full = result["windows"]["full"]
    print(f"Maximum drawdown: {full['drawdown']:.4f}")
    for row in full["aggregation"]:
        print(f"h={row['horizon']}: blocks={row['blocks']}, kurtosis={row['excess_kurtosis']:.4g}, kappa={row['kappa']:.4g}")
    for side, tail in full["tails"].items():
        print(f"{side}: {tail['n']} positive observations")
        for point in tail["hill"]:
            print(f"  k={point['k']}: alpha={point['alpha']:.4g}")
```

**Output**

```text
No local S&P cache; empirical diagnostics remain uncomputed.
```
:::

`uv run python demos/incerto/sp500_diagnostics.py` regenerates the synthetic
figures and prints the complete local-cache report when data are present.
It uses disjoint horizons 1, 5, and 21, then repeats with overlapping windows;
incomplete disjoint suffixes are dropped and counted. It also reports
subwindow dates, drawdowns, extrema, record counts, MS shares for powers
1–4, both tail fits, and the
[frozen holdout comparison](incerto-ml-econometrics.md#pipeline-holdout).
At least 44 closes are required to form a 21-return block in each half.
Such a small sample is sufficient to execute the code, not to support a
tail conclusion.

Choose windows and thresholds before interpreting results. Preserve the
data source, retrieval date, checksum, and excluded observations with each
private run. Examine dependence and time-window sensitivity before attaching
uncertainty intervals. These diagnostics are exploratory; testing many
thresholds and reporting only the most striking one changes the inferential
question. No market model or tail exponent is selected by this page.

## Source scope

SCoFT v4 §§10.1–10.3, pp. 193–205 [@taleb2025scoftv4] motivates these
diagnostics. The aggregation and record identities above are proved under
their stated assumptions; the figures are synthetic or analytic examples.
We do not reproduce the book's empirical figures, numerical verdicts, or
claim that moment instability establishes a power law. Definition 10.1
on p. 200 is replaced by the conditional-mean identity stated here, with
the correct sign and conditioning event for each tail.
