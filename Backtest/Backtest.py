"""
Backtest: turn the Q3 USD/JPY positions into P&L, under two sizings
(unit and volatility-targeted), against a buy & hold benchmark.

Timing (this is what avoids look-ahead):
  The position for signal-month M is decided when the forecast is formed (~the
  Tokyo-CPI release, end of month M). It becomes effective at the end of M and
  earns the USD/JPY return of the FOLLOWING month. So:
        strategy_return_t = position_{t-1} * fx_return_t
  i.e. the held position is the signal lagged one month. We never earn a return
  in the same month whose data we used to trade.

  fx_return_t = USDJPY_t / USDJPY_{t-1} - 1   (month-end to month-end)

Sizing — two tracks reported on the same window:
  * Unit       : direction only, position in {-1, 0, +1} (constant leverage).
  * Vol-target : scale the position to a constant target annualized volatility,
        scaled_position_t = direction_t * (TARGET_VOL / realized_vol_t)
    direction_t    : the Q3 sign (-sign(surprise)), in {-1, 0, +1}
    realized_vol_t : trailing annualized vol of USD/JPY returns, estimated with
                     PAST data only (rolling std through t-1, then shifted), so
                     there is no look-ahead in the scaling.
    Leverage is capped at MAX_LEV so a calm-market vol estimate can't blow the
    size up. Constant-risk sizing stops one volatile month from dominating and
    makes the Sharpe comparison cleaner (the professional default); its cost is
    higher turnover, since the size changes every month as vol moves.

Costs: every rebalance pays the bid/ask spread on the traded amount.
        turnover_t = |position_t - position_{t-1}|      (a flip +1->-1 = 2)
        cost_t     = COST_BPS/1e4 * turnover_t
  Net return = gross - cost; the equity curve is the compounded net return.

Benchmark: buy-and-hold USD/JPY (always long, +1). Sharpe uses rf=0 (a
simplification; the JPY cash rate is small). Carry (rate differential) is
ignored — a stated limitation, small at a monthly horizon.

Inputs : position.csv (Strategy), cpi_dataset.csv (Data, for the USDJPY level)
Outputs: equity-curve PNG, backtest CSV, metrics CSV
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IN_NAME = "position.csv"  # Input file name (from Strategy.py)

HERE = Path(__file__).parent
IN_DIR = HERE / "../Strategy"
DATA_DIR = HERE / "../Data"
DATA_PATH = DATA_DIR / "cpi_dataset.csv"
IN_PATH = IN_DIR / IN_NAME
OUT_DIR = HERE
OUT_PATH = OUT_DIR

COL_PERIOD = "Period"
COL_USDJPY = "USDJPY"
COL_POS = "USDJPY_position"

COST_BPS = 1.0        # one-way cost, bps per unit turnover
ANN = 12
TARGET_VOL = 0.10     # target annualized volatility (10%)
VOL_WINDOW = 12       # months of trailing returns for the vol estimate
MAX_LEV = 3.0         # cap on |scaled position|


# ---------------------------------------------------------------------------
# Backtest with both sizings
# ---------------------------------------------------------------------------

def backtest(pos: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    fx = fx[[COL_PERIOD, COL_USDJPY]].copy()
    fx["fx_ret"] = fx[COL_USDJPY].pct_change()

    # Trailing annualized realized vol, using returns through t-1 only.
    fx["realized_vol"] = fx["fx_ret"].rolling(VOL_WINDOW).std(ddof=0).shift(1) * np.sqrt(ANN)

    bt = fx.merge(pos[[COL_PERIOD, COL_POS]], on=COL_PERIOD, how="left").sort_values(COL_PERIOD)
    bt = bt.reset_index(drop=True)

    # Direction (sign) held one month after the signal (no look-ahead).
    bt["dir_held"] = bt[COL_POS].shift(1)

    first_signal = pos[COL_PERIOD].min()
    bt = bt[bt[COL_PERIOD] > first_signal].reset_index(drop=True)
    bt["dir_held"] = bt["dir_held"].fillna(0.0)

    # Vol-target scaling (also lagged: uses the vol known when the position is set).
    scale = (TARGET_VOL / bt["realized_vol"]).clip(upper=MAX_LEV)
    bt["pos_unit"] = bt["dir_held"]
    bt["pos_vt"] = bt["dir_held"] * scale

    cost = COST_BPS / 1e4
    for tag, poscol in [("unit", "pos_unit"), ("vt", "pos_vt")]:
        turn = bt[poscol].diff().abs()
        turn.iloc[0] = abs(bt[poscol].iloc[0])
        bt[f"ret_{tag}"] = bt[poscol] * bt["fx_ret"] - cost * turn
        bt[f"turn_{tag}"] = turn
    bt["ret_bh"] = bt["fx_ret"]

    for tag in ["unit", "vt", "bh"]:
        bt[f"eq_{tag}"] = (1 + bt[f"ret_{tag}"]).cumprod()
    return bt


def perf(r: pd.Series) -> dict:
    r = r.dropna(); n = len(r); eq = (1 + r).cumprod(); sd = r.std(ddof=0)
    return {"months": n, "total_return": eq.iloc[-1] - 1,
            "ann_return": (1 + r).prod() ** (ANN / n) - 1, "ann_vol": sd * np.sqrt(ANN),
            "sharpe": (r.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan,
            "max_drawdown": (eq / eq.cummax() - 1).min(), "hit_rate": (r > 0).mean()}


def metrics_table(bt: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Strategy_unit": perf(bt["ret_unit"]),
        "Strategy_volTarget": perf(bt["ret_vt"]),
        "Buy&Hold_USDJPY": perf(bt["ret_bh"]),
    }).T


def plot_equity(bt: pd.DataFrame, out_path: Path) -> None:

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(bt[COL_PERIOD], bt["eq_vt"], "-", color="tab:blue", lw=2,
            label=f"Vol-target ({int(TARGET_VOL*100)}%)")
    ax.plot(bt[COL_PERIOD], bt["eq_unit"], "--", color="tab:green", lw=1.8, label="Unit (+-1)")
    ax.plot(bt[COL_PERIOD], bt["eq_bh"], "-", color="tab:grey", lw=1.5, label="Buy & hold USD/JPY")
    ax.axhline(1, color="black", lw=0.6)
    ax.set_ylabel("Equity (start = 1)")
    ax.set_title("Backtest — vol-targeted vs unit sizing vs buy & hold")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout();
    fig.savefig(out_path, dpi=200);

    plt.close(fig)


def main():
    pos = pd.read_csv(IN_PATH, parse_dates=[COL_PERIOD])
    fx = pd.read_csv(DATA_PATH, parse_dates=[COL_PERIOD])
    bt = backtest(pos, fx)
    tbl = metrics_table(bt)

    print(f"=== Backtest {bt[COL_PERIOD].min().date()} .. {bt[COL_PERIOD].max().date()} "
          f"(target vol={int(TARGET_VOL*100)}%, cap={MAX_LEV}x, cost={COST_BPS}bp/unit) ===")
    print(tbl.round(4).to_string())
    print(f"\navg |position|: unit={bt['pos_unit'].abs().mean():.2f}, "
          f"vol-target={bt['pos_vt'].abs().mean():.2f}  |  "
          f"avg turnover: unit={bt['turn_unit'].mean():.2f}, vt={bt['turn_vt'].mean():.2f}")

    bt.to_csv(OUT_DIR / "vt_backtest.csv", index=False)
    tbl.to_csv(OUT_DIR / "vt_metrics.csv")
    plot_equity(bt, OUT_DIR / "vt_equity_curve.png")


if __name__ == "__main__":
    main()

