# Japan-CPI-USDJPY-Strategy

**Forecast Japan's core CPI inflation, and trade the inflation *surprise* through USD/JPY.**

A compact, end-to-end macro/quant pipeline: from public data → a one-month-ahead
inflation forecast → an investable USD/JPY signal → a realistic, cost-and-timing-aware
backtest. Built around a single, still-unfolding macro story — Japan's exit from two
decades of deflation and the Bank of Japan's ongoing policy normalization.

---

## 1. The idea

Japan spent ~25 years in deflation and near-zero rates. Since exiting negative rates
(NIRP) and Yield Curve Control (YCC) in **March 2024**, the Bank of Japan (BOJ) has been
**normalizing** policy, hiking to a 31-year-high policy rate by 2026 as inflation held
above its 2% target. This is a genuine, live **regime change**, which makes Japanese
inflation an unusually interesting variable to forecast.

**Why trade it through USD/JPY, not the CPI basket?** Trade groceries, rent
and services is not direct. You trade the asset whose price *reacts* to the inflation number — and what
makes an asset react to inflation is the central bank:

> **inflation surprise → BOJ policy repricing → JP–US rate differential → yen**

A CPI print **hotter than expected** makes the BOJ more hawkish, Japanese rate
expectations rise, the JP–US differential narrows, and the yen strengthens (USD/JPY
falls). So a hot inflation surprise is a **short-USD/JPY** signal; a cool surprise is
long. USD/JPY is the most liquid, freely reproducible expression of this chain (rates
would be one step more direct, but less clean on free data).

Crucially, we trade the **surprise** — the forecast minus what is already priced — because
only the unexpected part of a print moves markets.

> **Regime caveat (honest, and central):** the sign of this trade assumes the *post-2024
> "BOJ reacts to inflation"* regime. In **2022–2023** inflation rose but the BOJ stayed
> ultra-dovish, so the yen *weakened* on high inflation — the opposite sign. The strategy
> bets the reaction-function regime persists; the 2022 subperiod is the natural stress test.

---

## 2. Repository structure

The pipeline is sequential — each step consumes the previous step's output.

```
Japan-CPI-USDJPY-Strategy/
├── Data/
│   ├── Data.py              # Q1 — build the dataset from public sources
│   └── cpi_dataset.csv      # output: monthly levels + availability dates
├── Forecasting model/
│   ├── Model.py             # Q2 — ARX forecast of core-CPI MoM (+ YoY reconstruction)
│   ├── mom_oos_dataset.csv  # out-of-sample forecasts (MoM)
│   ├── yoy_oos_dataset.csv  # out-of-sample forecasts (YoY, reconstructed)
│   ├── *_metrics.csv        # RMSE / MAE / skill vs benchmarks
│   └── *_forecast.png       # forecast-vs-realized charts
├── Strategy/
│   ├── Strategy.py          # Q3 — turn forecasts into USD/JPY positions
│   ├── position.csv         # the signal / position series
│   └── position.png
└── Backtest/
    ├── Backtest.py          # Q4 — P&L, costs, Sharpe, drawdown, robustness
    ├── vt_backtest.csv
    ├── vt_metrics.csv
    └── vt_equity_curve.png
```

---

## 3. Data (`Data/Data.py`)

**Target:** `All items, less fresh food` — Japan's official **core CPI**, the BOJ's actual
policy-target measure. (`All items, less fresh food and energy`, "core-core", is pulled too
for context.) Fresh food and energy are volatile and don't drive the inflation *trend* the
BOJ acts on.

**Series and sources (all public, all reproducible):**

| Series | Role | Source |
|---|---|---|
| National core CPI (All Japan) | target | Statistics Bureau of Japan, e-Stat API (2020-base, table `0003427113`) |
| Tokyo core CPI (Ku-area) | **leading indicator** | same e-Stat table |
| USD/JPY spot | traded instrument + lagged feature | FRED `DEXJPUS` |
| Brent crude oil | cost-push feature | FRED `DCOILBRENTEU` |

**Key data-engineering decisions:**

- **Growth rates, not index levels.** The Statistics Bureau rebases 2020-base → 2025-base
  with the Aug 21 2026 release (different basket weights). MoM/YoY growth rates are robust
  to that splice; raw levels are not.
- **Release lag / availability.** Every observation carries an `*_available` date — *when the
  number became public*, kept separate from the month it describes. National CPI for month
  *M* is public ~day 24 of *M+1*; **Tokyo CPI for the same month *M* is public ~3 weeks
  earlier** (end of *M*). Market prices (USD/JPY, oil) are known same-day. This is what lets
  every later step avoid look-ahead bias.
- **The Tokyo lead is the engine of the forecast** — a genuine, official ~3-week head start
  on the national print for the *same* reference month.
- **Missing data:** FRED daily series are resampled to month-end (last value in month),
  short gaps forward-filled up to 5 trading days; anything remaining is reported, not
  silently imputed.

Requires a free e-Stat application ID (see [How to run](#7-how-to-run)); FRED needs no key.

---

## 4. Forecasting model (`Forecasting model/Model.py`)

**Model: ARX** (single-equation autoregression with exogenous regressors), fit by OLS —
deliberately simple, interpretable, and appropriate for ~460 monthly observations (an RNN
would overfit; a VAR/VARMA would waste parameters trying to forecast oil and FX, which are
essentially random walks).

We forecast **month-over-month (MoM)** core CPI one month ahead:

```
CPI_Japan_MoM(M) ~ const
    + CPI_Japan_MoM(M-1..M-3)   # own persistence (AR terms)
    + CPI_Tokyo_MoM(M)          # leading indicator (published ~3 wks before national M)
    + USDJPY_Ret_MoM(M-1)       # lagged FX pass-through (lagged also to avoid circularity,
    + Oil_Brent_Ret_MoM(M-1)    #   since USD/JPY is the traded instrument)
```

**Why MoM and not YoY** (the number the BOJ headlines)? YoY is a 12-month *overlapping*
difference — very persistent, which makes a YoY random-walk benchmark artificially hard to
beat and would *overstate* skill. MoM is the honest test. And by the identity
`YoY(M) = YoY(M-1) + MoM(M) − MoM(M-12)`, the only *new* information in each YoY print **is**
that month's MoM — so a MoM forecast loses nothing. We reconstruct a genuine 1-step YoY
forecast honestly, `1 + YoY_hat(M) = (1 + MoM_hat(M)) · L(M-1)/L(M-12)` (11 realized months + 1 forecast month), and score it against native YoY benchmarks.

**Evaluation:** expanding-window, one-step-ahead, **out-of-sample** — refit on all data
strictly before month *t*, predict *t*, step forward. Benchmarks: a **random walk** (RW,
`MoM(M)=MoM(M-1)`) and the **historical mean** (HM). No look-ahead is *asserted* against the
`*_available` dates (a constant 23-day feature→target gap).

**Results (out-of-sample, test from 2024-06, n≈24):**

| Forecast | RMSE | Skill vs Random Walk | Skill vs Historical Mean |
|---|---|---|---|
| ARX, MoM | **0.144** | **+69%** | +56% |
| ARX, YoY (reconstructed) | **0.147** | **+49%** | +78% |

The model beats both naive benchmarks comfortably. Most of the skill comes from the Tokyo
leading indicator — legitimate, since Tokyo prints before the national figure.

---

## 5. Strategy (`Strategy/Strategy.py`)

Turns each forecast into a position. The tradeable object is the **surprise**:

```
surprise(M)         = ARX_forecast(M) − naive_expectation(M)      # expectation ≈ RW print
USDJPY_position(M)  = − sign(surprise)                            # hot surprise → SHORT USD/JPY
```

Rules (replicable, monthly): rebalance when the forecast is formed (~Tokyo-CPI release);
enter on the sign of the surprise; an optional dead-band stays flat on tiny surprises; unit
sizing `{-1, 0, +1}`; the position is simply replaced by next month's signal. This script
produces **only** signals/positions — P&L lives in Q4.

*(Refinement noted for the deck: the BOJ reacts most to the wage-driven slice of inflation —
services and core-core, ex energy — so keying the surprise on core-core would be a sharper,
more reaction-function-aligned signal.)*

---

## 6. Backtest (`Backtest/Backtest.py`)

**Timing (the anti-look-ahead core):** the position for signal-month *M* is set at the end
of *M* and earns the **following** month's USD/JPY return —
`strategy_return(t) = position(t-1) · fx_return(t)`. You never earn a return in the same
month whose data you used to trade.

**Costs:** every rebalance pays the bid/ask spread, `cost = COST_BPS/1e4 · |Δposition|`
(a flip costs 2 units). **Carry** (the rate differential) is ignored — a stated limitation,
small at a monthly horizon.

**Two sizings** are reported side by side: **unit** (`±1`, constant leverage) and
**volatility-targeted** (scale to a constant 10% annualized vol using *past-only* realized
vol, capped at 3×). Benchmark: **buy-and-hold USD/JPY**.

**Results (2024-07 → 2026-07, n=25, 1 bp/unit cost, Sharpe at rf=0):**

| | Total return | Ann. return | Ann. vol | **Sharpe** | Max drawdown | Hit rate |
|---|---|---|---|---|---|---|
| **Strategy (unit)** | **+17.8%** | 8.2% | 9.9% | **0.85** | −4.7% | 52% |
| Strategy (vol-target 10%) | +13.4% | 6.2% | 9.5% | 0.69 | **−4.0%** | 52% |
| Buy & hold USD/JPY | +1.0% | 0.5% | 10.2% | 0.10 | −9.4% | 48% |

The strategy earns a far higher risk-adjusted return than passively holding the pair, at
similar volatility and roughly half the drawdown. **Robustness:** dropping the single best
month lowers Sharpe from 0.85 to ~0.60 — helped by that month, but not dependent on it.
Vol-targeting here slightly *reduces* Sharpe (the yen's realized vol was already near the
10% target, and scaling trimmed high-vol winning months) while improving drawdown — a
textbook risk-control trade-off, not a free lunch.

---

## 7. How to run

```bash
# 1. install
pip install -r requirements.txt

# 2. e-Stat API key (free, instant, no review): https://www.e-stat.go.jp  (MyPage → Application ID)
export ESTAT_APP_ID="your_app_id_here"      # macOS/Linux
# setx ESTAT_APP_ID "your_app_id_here"      # Windows

# 3. run the pipeline in order
python "Data/Data.py"
python "Forecasting model/Model.py"
python "Strategy/Strategy.py"
python "Backtest/Backtest.py"
```

Dependencies: `pandas`, `numpy`, `requests`, `statsmodels`, `matplotlib`.

---

## 8. Limitations & next steps

- **Regime dependence.** The trade's sign assumes the post-2024 "BOJ reacts to inflation"
  regime; it inverts in a behind-the-curve regime like 2022. *Next:* backtest the 2022–2023
  subperiod explicitly.
- **Short live window.** ~25 out-of-sample months → noisy Sharpe; treat as indicative.
- **Naive expectation.** The "surprise" is measured against a random-walk proxy, not true
  economist consensus (e.g. Bloomberg survey), which isn't freely reproducible.
- **Signal component.** Core CPI (ex fresh food) still contains energy; a **core-core /
  services** signal is closer to what the BOJ's reaction function keys on.
- **Execution timing.** The signal is assumed actionable at month-end close; Tokyo CPI
  prints in the final days of the month, so this is conservative for ~10/12 months and a
  ~1-day approximation for the other two (April, September). Immaterial at monthly frequency.
- **Carry ignored** in P&L; small monthly, but non-zero as the BOJ hikes.
- **Fixed release lags** (24 / 1 days) rather than the exact per-month schedule.

---

## 9. Data sources & references

**Data**

- Statistics Bureau of Japan — Consumer Price Index (2020-base), e-Stat database table `0003427113`: https://www.e-stat.go.jp/en/dbview?sid=0003427113
- e-Stat API guide: https://www.e-stat.go.jp/api/en/api-info/api-guide
- CPI methodology (2020-base): https://www.stat.go.jp/english/data/cpi/1590.html
- CPI release schedule: https://www.stat.go.jp/english/data/cpi/1582.html
- Linking of old and new base indices (Ch. 6): https://www.stat.go.jp/english/data/cpi/pdf/2020base3-6.pdf
- Seasonal adjustment (Ch. 7): https://www.stat.go.jp/english/data/cpi/pdf/2020base3-7.pdf
- 2020-base item/group code list: https://www.stat.go.jp/english/data/cpi/pdf/2020base-list.pdf
- FRED — USD/JPY spot (`DEXJPUS`): https://fred.stlouisfed.org/series/DEXJPUS
- FRED — Brent crude (`DCOILBRENTEU`): https://fred.stlouisfed.org/series/DCOILBRENTEU

**Bank of Japan policy**

- BOJ — Price Stability Target of 2 Percent: https://www.boj.or.jp/en/mopo/outline/target.htm
- BOJ — "Changes in the Monetary Policy Framework," March 19 2024 (NIRP/YCC exit): https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2024/k240319a.pdf
- BOJ — Indicators for core CPI: https://www.boj.or.jp/en/research/research_data/cpi/index.htm
- BOJ Review — measures of underlying inflation (2015): https://www.boj.or.jp/en/research/wps_rev/rev_2015/rev15e06.htm

**Context**

- Bloomberg — "Japan Still Can't Escape the Old Normal" (Jun 9 2026): https://www.bloomberg.com/opinion/newsletters/2026-06-09/takaichi-trade-off-japan-still-can-t-escape-the-old-normal
- Bloomberg — "Japan Has Spent Billions to Prop Up the Yen. Why Isn't It Working?" (Jul 22 2026): https://www.bloomberg.com/news/articles/2026-07-22/jpy-usd-why-is-japan-yen-so-weak-can-government-intervention-help-currency


