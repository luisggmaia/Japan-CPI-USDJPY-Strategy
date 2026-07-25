"""
Forecasting model

Forecasts Japan's national Core CPI ("All items, less fresh food") Month-over-Month
(MoM) inflation one month ahead (forecast horizon), and shows it beats naive benchmarks
(simple random walk and historical mean) out-of-sample.

Target: national core CPI MoM (%).
Although YoY is the number the BOJ actually targets, it is a 12-month overlapping difference,
which induces strong serial correlation. A MoM forecast can be converted to a YoY forecast and
can be forecasted with a simpler ARX model, rather than a more complex ARMAX model which would
The MoM is also what the strategy (Q3) trades on, so we forecast it directly.

Model: ARX (single-equation autoregression with exogenous regressors), OLS:

    CPI_Japan_MoM(M) ~ const
           + CPI_Japan_MoM(M-1)    # 1 M persistence (AR term)
           + CPI_Japan_MoM(M-2)    # 2 M persistence (AR term)
           + CPI_Japan_MoM(M-3)    # 3 M persistence (AR term)
           + CPI_Tokyo_MoM(M)      # LEADING INDICATOR: Tokyo core CPI for month
                                   #   M is published ~3 weeks BEFORE national M
                                   #   (verified: constant 23-day gap), so it is
                                   #   a legitimate, non-look-ahead predictor.
           + Oil_Brent_MoM(M-1)    # MoM oil log-change, lagged (cost-push
           + USDJPY_MoM(M-1)       #   pass-through from energy / import prices)

Benchmarks.
Both are expressed in MoM, so they are directly comparable.
  * RW : random walk in MoM. Zero mean expectation of previous month,
            MoM(M) = MoM(M-1). A naive benchmark.  
  * HM : historical mean in MoM. Expanding mean of all previous months,
            MoM(M) = mean(MoM(1..M-1)). A naive benchmark.

Evaluation: expanding-window, one-step-ahead, out-of-sample. For each month t in
the test window we refit on all data strictly before t and predict t.

Deliverables: forecast-vs-realized chart (PNG) + metrics table (CSV) +
forecast table (CSV).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend -> save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IN_NAME = "cpi_dataset.csv"  # Input file name (from Data.py)

HERE = Path(__file__).parent
IN_DIR = HERE / "../Data"
OUT_DIR = HERE
IN_PATH = IN_DIR / IN_NAME
OUT_PATH = OUT_DIR

TEST_START = "2024-06-01"   # ~24 monthly OOS points (H2-2024 .. mid-2026)

# Column names as saved by the Q1 step (cpi_dataset.csv).
COL_PERIOD = "Period"
COL_JAPAN_LEVEL = "Japan Core CPI"
COL_TOKYO_LEVEL = "Tokyo Core CPI"
COL_USDJPY = "USDJPY"
COL_OIL = "Oil_Brent"
COL_JAPAN_AV = "Japan Core CPI_available"
COL_TOKYO_AV = "Tokyo Core CPI_available"
COL_USDJPY_AV = "USDJPY_available"
COL_OIL_AV = "Oil_Brent_available"

TARGET = "y"

HAC_MAXLAGS = 12   # Newey-West lag window (monthly data, up to seasonal horizon)

# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def calculations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Growth rates / returns from the level columns (no forward-fill).
    """

    calc_df = df.sort_values(COL_PERIOD).reset_index(drop = True).copy()

    calc_df["CPI_Japan_YoY"] = (calc_df[COL_JAPAN_LEVEL] / calc_df[COL_JAPAN_LEVEL].shift(12) - 1) * 100
    calc_df["CPI_Japan_MoM"] = (calc_df[COL_JAPAN_LEVEL] / calc_df[COL_JAPAN_LEVEL].shift(1) - 1) * 100
    calc_df["CPI_Tokyo_YoY"] = (calc_df[COL_TOKYO_LEVEL] / calc_df[COL_TOKYO_LEVEL].shift(12) - 1) * 100
    calc_df["CPI_Tokyo_MoM"] = (calc_df[COL_TOKYO_LEVEL] / calc_df[COL_TOKYO_LEVEL].shift(1) - 1) * 100
    calc_df["USDJPY_Ret_YoY"] = np.log(calc_df[COL_USDJPY]).diff(12)
    calc_df["USDJPY_Ret_MoM"] = np.log(calc_df[COL_USDJPY]).diff(1)
    calc_df["Oil_Brent_Ret_YoY"] = np.log(calc_df[COL_OIL]).diff(12)
    calc_df["Oil_Brent_Ret_MoM"] = np.log(calc_df[COL_OIL]).diff(1)

    return calc_df


def _availability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the availability of the target and features.
    target_av (national release) and feat_av (latest feature availability).
    """
    Japan_CPI_av = pd.to_datetime(df[COL_JAPAN_AV])
    Tokyo_CPI_av = pd.to_datetime(df[COL_TOKYO_AV])
    usdjpy = pd.to_datetime(df[COL_USDJPY_AV])
    oil = pd.to_datetime(df[COL_OIL_AV])

    out = pd.DataFrame({COL_PERIOD: df[COL_PERIOD]})
    out["target_av"] = Japan_CPI_av
    # Binding feature availability: national lag (>=1), Tokyo month T, market lag-1.
    out["feat_av"] = pd.concat([Japan_CPI_av.shift(1), Tokyo_CPI_av, oil.shift(1), usdjpy.shift(1)], axis=1).max(axis=1)

    return out


def mom_arx(calc_df: pd.DataFrame, av_df: pd.DataFrame) -> pd.DataFrame:
    """
    ARX model dataframe tracking MoM Japan Core CPI.
    """

    model_df = pd.DataFrame({COL_PERIOD: calc_df[COL_PERIOD]})

    # The model
    model_df[TARGET] = calc_df["CPI_Japan_MoM"]
    model_df["AR_T_1"] = calc_df["CPI_Japan_MoM"].shift(1)
    model_df["AR_T_2"] = calc_df["CPI_Japan_MoM"].shift(2)
    model_df["AR_T_3"] = calc_df["CPI_Japan_MoM"].shift(3)
    model_df["Tokyo_MoM_T"] = calc_df["CPI_Tokyo_MoM"]
    # No lag; the information will be known from the current month, due to the previous release of Tokyo cpi
    model_df["USDJPY_MoM_T_1"] = calc_df["USDJPY_Ret_MoM"].shift(1)
    # lag of 1 Month; the information will be known from the previous month's last day
    model_df["Oil_MoM_T_1"] = calc_df["Oil_Brent_Ret_MoM"].shift(1)
    # lag of 1 Month; the information will be known from the previous month's last day

    model_df = model_df.merge(av_df, on=COL_PERIOD, how="left")

    # The benchmarks
    model_df["RW"] = model_df[TARGET].shift(1) # Random walk in MoM
    model_df["HM"] = model_df[TARGET].mean().shift(1) # Historical mean in MoM

    feats = ["AR_T_1", "AR_T_2", "AR_T_3", "Tokyo_MoM_T", "USDJPY_MoM_T_1", "Oil_MoM_T_1"]

    return model_df, feats


def yoy_arx(calc_df: pd.DataFrame, av_df: pd.DataFrame) -> pd.DataFrame:
    """
    ARX model dataframe tracking YoY Japan Core CPI.
    """

    model_df = pd.DataFrame({COL_PERIOD: calc_df[COL_PERIOD]})

    # The model
    model_df[TARGET] = calc_df["CPI_Japan_YoY"]
    model_df["AR_T_1"] = calc_df["CPI_Japan_YoY"].shift(1)
    model_df["AR_T_12"] = calc_df["CPI_Japan_YoY"].shift(12)
    model_df["Tokyo_YoY_T"] = calc_df["CPI_Tokyo_YoY"]
    # No lag; the information will be known from the current month, due to the previous release of Tokyo cpi
    model_df["USDJPY_YoY_T_1"] = calc_df["USDJPY_Ret_YoY"].shift(1)
    # lag of 1 Month; the information will be known from the previous month's last day
    model_df["Oil_YoY_T_1"] = calc_df["Oil_Brent_Ret_YoY"].shift(1)
    # lag of 1 Month; the information will be known from the previous month's last day

    model_df = model_df.merge(av_df, on=COL_PERIOD, how="left")

    # The benchmarks
    model_df["RW"] = model_df[TARGET].shift(1) # Random walk in YoY
    model_df["HM"] = model_df[TARGET].mean() # Historical mean in YoY

    feats = ["AR_T_1", "AR_T_12", "Tokyo_YoY_T", "USDJPY_YoY_T_1", "Oil_YoY_T_1"]

    return model_df, feats


def run_oos(model_df: pd.DataFrame, feats, test_start: str) -> pd.DataFrame:
    """
    Expanding-window one-step-ahead OOS forecasts. Returns all usable months:
    pre-test 'ARX' = in-sample fitted (line only); test rows = true OOS.
    Benchmarks are leak-free: RW = last realized target; HM = expanding mean
    of realized target up to t-1.
    """

    need = [TARGET] + feats
    usable = model_df.dropna(subset = need).reset_index(drop = True)
    test_idx = usable.index[usable[COL_PERIOD] >= pd.Timestamp(test_start)]
    if len(test_idx) == 0:
        raise ValueError(f"No usable test rows on/after {test_start}.")

    oos_df = usable[[COL_PERIOD, TARGET, "RW", "HM"]].copy()
    oos_df = oos_df.rename(columns={TARGET: "Realized"})
    oos_df["ARX"] = np.nan
    oos_df["OOS"] = False

    # Pre-test: one fit on all data before the test window -> in-sample fitted
    # values, just to draw the full ARX line (NOT out-of-sample).
    i_0 = test_idx[0]
    train = usable.iloc[:i_0]
    ols = sm.OLS(train[TARGET], sm.add_constant(train[feats])).fit()
    oos_df.loc[:i_0-1, "ARX"] = ols.fittedvalues.values
    
    # Test: expanding-window, one-step-ahead OOS forecasts.
    records = []
    for i in test_idx:
        train = usable.iloc[:i] # strictly before month t -> no leakage
        row = usable.iloc[i]
        if len(train) < 30:
            continue

        ols = sm.OLS(train[TARGET], sm.add_constant(train[feats])).fit()
        X_te = sm.add_constant(row[feats].to_frame().T, has_constant = "add")

        oos_df.loc[i, "OOS"] = True
        oos_df.loc[i, "ARX"] = float(ols.predict(X_te).iloc[0])
        # oos_df.set_index(COL_PERIOD)

    return oos_df


def mom_oos_to_yoy(mom_oos: pd.DataFrame, calc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Model-1 MoM forecasts to honest 1-step YoY forecasts:
        1 + YoY_hat_t = (1 + MoM_hat_t/100) * L_{t-1} / L_{t-12}
    (11 realized months + 1 forecast month). Benchmarks are the native YoY
    random walk (realized YoY_{t-1}) and expanding YoY mean, so Model 1 and
    Model 2 are compared on the SAME YoY benchmarks.
    """

    L = calc_df.set_index(COL_PERIOD)[COL_JAPAN_LEVEL]
    ryoy = calc_df.set_index(COL_PERIOD)["CPI_Japan_YoY"]
    idx = mom_oos[COL_PERIOD]
    ratio11 = (L.shift(1) / L.shift(12)).reindex(idx).values

    out = mom_oos[[COL_PERIOD, "OOS"]].copy()
    out["Realized"] = ((1 + mom_oos["Realized"].values / 100) * ratio11 - 1) * 100
    out["ARX"] = ((1 + mom_oos["ARX"].values / 100) * ratio11 - 1) * 100
    out["RW"] = ryoy.shift(1).reindex(idx).values                     # native YoY RW
    out["HM"] = ryoy.expanding().mean().shift(1).reindex(idx).values  # native YoY HM

    return out


def metrics_table(oos: pd.DataFrame) -> pd.DataFrame:
    """
    RMSE / MAE per forecaster, plus RMSE skill vs each benchmark.
    """

    def rmse(e): return float(np.sqrt(np.mean(e ** 2)))
    def mae(e):  return float(np.mean(np.abs(e)))

    fil_oos = oos[oos["OOS"]]

    tbl = pd.DataFrame(
        [{"Model": n,
          "RMSE": rmse(fil_oos[n] - fil_oos["Realized"]),
          "MAE": mae(fil_oos[n] - fil_oos["Realized"])}
         for n in ["ARX", "RW", "HM"]]
    ).set_index("Model")

    for bench in ["RW", "HM"]:
        tbl[f"skill_vs_{bench}_%"] = (1 - tbl["RMSE"] / tbl.loc[bench, "RMSE"]) * 100

    return tbl

# Plot

def plot_forecast(oos: pd.DataFrame, params: dict) -> None:

    fil_oos = oos[params.get("idx", oos["OOS"])]

    fig, ax = plt.subplots(figsize = (12, 8))

    ax.plot(fil_oos[COL_PERIOD], fil_oos["Realized"], "o-", color = "black", lw = 2, label = "Realized")
    ax.plot(fil_oos[COL_PERIOD], fil_oos["ARX"], "s--", color = "tab:blue", label = "ARX forecast")
    ax.plot(fil_oos[COL_PERIOD], fil_oos["RW"], "^:", color = "tab:red", alpha = 0.8, label = "Random walk")
    ax.plot(fil_oos[COL_PERIOD], fil_oos["HM"], "v:", color = "tab:orange", alpha = 0.7, label = "Historical mean")
    ax.set_title(params.get("title", "Forecast"))
    ax.set_ylabel(params.get("ylabel", "MoM %"))
    ax.legend()
    ax.grid(alpha = 0.3)

    fig.tight_layout()
    fig.savefig(params.get("fig_name", "forecast.png"), dpi=300)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(IN_PATH)
    df[COL_PERIOD] = pd.to_datetime(df[COL_PERIOD], format="%Y-%m-%d")

    calc_df = calculations(df)
    av_df = _availability(df)

    model, feats = mom_arx(calc_df, av_df)

    mom_oos = run_oos(model, feats, test_start = TEST_START)
    yoy_oos = mom_oos_to_yoy(mom_oos, calc_df)

    plot_params_mom = {
        "fig_name": OUT_DIR / "mom_forecast.png",
        "title": "Out-of-sample forecasts of Japan Core CPI (MoM)",
        "ylabel": "MoM %"
    }
    plot_params_yoy = {
        "fig_name": OUT_DIR / "yoy_forecast.png",
        "title": "Out-of-sample forecasts of Japan Core CPI (YoY)",
        "ylabel": "YoY %"
    }

    plot_forecast(mom_oos, plot_params_mom)
    plot_forecast(yoy_oos, plot_params_yoy)

    mom_tbl = metrics_table(mom_oos)
    yoy_tbl = metrics_table(yoy_oos)

    mom_oos.to_csv(OUT_DIR / "mom_oos_dataset.csv", index = False)
    yoy_oos.to_csv(OUT_DIR / "yoy_oos_dataset.csv", index = False)
    mom_tbl.to_csv(OUT_DIR / "mom_oos_metrics.csv", index = True)
    yoy_tbl.to_csv(OUT_DIR / "yoy_oos_metrics.csv", index = True)


if __name__ == "__main__":
    main()


