"""
Strategy: turn the Q2 MoM forecasts into USD/JPY positions.

Economic logic (goes in the deck):
  We forecast Japan core-CPI inflation. Inflation drives the BOJ: a print
  HOTTER than the market naively expects makes the BOJ more hawkish -> Japanese
  rate expectations rise -> the JP-US rate differential narrows -> the yen
  strengthens -> USD/JPY falls. So a hot inflation surprise is a SHORT-USD/JPY
  signal (long yen); a cool surprise is long USD/JPY.

  We trade the surprise, not the level:
        surprise_t = ARX_forecast_t - naive_expectation_t
  Naive expectation = the random-walk print (last realized MoM), i.e. what an
  uninformed observer would pencil in. Positive surprise -> short USD/JPY.

Signal / position rules (replicable, monthly):
  * Rebalance monthly, when the new forecast is formed (around the Tokyo-CPI
    release, ~3 weeks before the national print -> the position is set on
    information already public; P&L timing is handled in Q4).
  * Entry / direction: USDJPY_position = -sign(surprise).
  * Dead-band: stay flat (0) when |surprise| <= DEADBAND, to avoid trading noise.
  * Sizing: unit position in {-1, 0, +1}. (A vol-scaled or surprise-scaled
    size is a natural extension -- left for Q4 robustness.)
  * Exit: the position is simply replaced by next month's signal (flip or flat).

This script ONLY produces signals/positions. Returns,
costs, Sharpe and drawdown are computed in Backtest folder.

Input : mom_oos_dataset.csv  (written by Q2_solution.py: Period, Realized, RW, HM, ARX, OOS)
Output: position.csv + position.png
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

IN_NAME = "mom_oos_dataset.csv"  # Input file name (from Model.py)

HERE = Path(__file__).parent
IN_DIR = HERE / "../Forecasting model"
OUT_DIR = HERE
IN_PATH = IN_DIR / IN_NAME
OUT_PATH = OUT_DIR

COL_PERIOD = "Period"
EXPECTATION = "RW"     # naive expectation of the CPI print: "RW" or "HM"
DEADBAND = 0.0         # MoM %-points; |surprise| <= DEADBAND -> stay flat


# ---------------------------------------------------------------------------
# Signal / position construction
# ---------------------------------------------------------------------------

def build_positions(oos: pd.DataFrame) -> pd.DataFrame:
    """
    From the OOS MoM forecasts, produce the USD/JPY position series.

    signal          : +1 hot / -1 cool / 0 flat  (inflation-surprise direction)
    USDJPY_position : -signal  (short USD/JPY on a hot surprise)
    """

    df = oos[oos["OOS"]].copy() if "OOS" in oos.columns else oos.copy()
    df = df.sort_values(COL_PERIOD).reset_index(drop=True)

    df["Expectation"] = df[EXPECTATION]
    df["Surprise"] = df["ARX"] - df["Expectation"]
    
    signal = np.sign(df["Surprise"])
    signal[df["Surprise"].abs() <= DEADBAND] = 0
    df["Signal"] = signal.astype(int)
    df["USDJPY_position"] = -df["Signal"]  # hot surprise -> short USD/JPY

    return df[[COL_PERIOD, "ARX", "Expectation", "Surprise",
               "Signal", "USDJPY_position"]]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_positions(pos: pd.DataFrame, out_path: Path) -> None:
    
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ["tab:green" if p < 0 else "tab:red" if p > 0 else "grey"
              for p in pos["USDJPY_position"]]
    
    ax.bar(pos[COL_PERIOD], pos["USDJPY_position"], width=20, color=colors, alpha=0.7)
    ax.plot(pos[COL_PERIOD], pos["Surprise"], "o-", color="black", lw=1.5,
            label="Inflation surprise (ARX - expectation, MoM %)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("USD/JPY position vs inflation surprise\n"
                 "(green = short USD/JPY on hot surprise, red = long USD/JPY)")
    ax.set_ylabel("Position (bars) / Surprise (line)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    oos = pd.read_csv(IN_PATH, parse_dates=[COL_PERIOD])
    pos = build_positions(oos)

    pos.to_csv(OUT_DIR / "position.csv", index=False)
    plot_positions(pos, OUT_DIR / "position.png")


if __name__ == "__main__":
    main()

