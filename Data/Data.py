"""
Data aquisition and cleaning

Builds a monthly dataset for forecasting Japan's Core CPI — the Bank of
Japan's official inflation-target measure, "All items, less fresh food" —
by combining four public series (explanatory variables):

    1. Japan/National Core CPI        — Statistics Bureau of Japan (e-Stat),
                                         2020-base table, "All Japan" area.
    2. Tokyo Core CPI (leading ind.)  — same e-Stat table, "Ku-area of Tokyo".
    3. USD/JPY spot rate              — FRED, series DEXJPUS.
    4. Brent crude oil price          — FRED, series DCOILBRENTEU.

Key design decisions (elaborated in the presentation / README, kept short
here as inline comments):

  * Target = "All items, less fresh food", not headline CPI. This is
    Japan's official "core CPI" and the BOJ's actual policy-target measure,
    which ties the forecast directly to the BOJ-reaction -> USD/JPY.

  * We work in growth rates (MoM / YoY), not raw index levels, because the
    Statistics Bureau is rebasing the series from 2020-base to 2025-base (
    different weights for each base) with the August 21, 2026 release
    (stat.go.jp/english/data/cpi/1582.html). Growth rates are robust to that
    rebase; index levels would not be.

  * Every observation carries an `available_date` — the date the figure
    was (or would have been) publicly known — kept separate from `period`,
    the month it describes. This is what lets the forecasting and trading
    strategy respect the release lag and avoid look-ahead bias.
    Assumptions (fixed lags, for simplicity — a limitation worth stating
    explicitly):
      - Japan/National CPI for month M -> available ~day 24 of month M+1
        (according to the Statistics Bureau release calendar).
      - Tokyo CPI for month M -> available ~day 1 of month M+1 (released a
        few days before month M itself ends) -> a genuine ~3-week lead on
        the national figure for the SAME reference month.
      - USD/JPY and oil are market prices -> available same day, no lag.

  * Missing data: FRED daily series have gaps on weekends/holidays; we
    resample to month-end using the last available value in the month,
    forward-filling gaps of at most MAX_FFILL_DAYS trading days. Any
    remaining gaps are reported via logging, not silently imputed.

Getting the datasets:

  * For Japan and Ku-area of Tokyo CPI, provided by the Statistics Bureau of
    Japan (e-Stat). It is feasable using the e-Stat API, once it provides all
    necessary data in a single dataframe. But an application ID is required.
    Register a free application ID at e-stat.go.jp
    (MyPage > Application ID — issued instantly, no review), then set it
    as the ESTAT_APP_ID environment variable. The script
    calls e-Stat's "Simple" endpoint (getSimpleStatsData).

  * For USD/JPY and Brent crude oil, provided by FRED (Federal Reserve Bank of
    St. Louis). No API key is required; the script calls FRED's public CSV
    endpoint.

Data sources are documented in README.md.
"""

import io
import os
from pathlib import Path

import pandas as pd
import requests


# ------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------

OUT_NAME = "cpi_dataset.csv"  # Final file name

HERE = Path(__file__).parent
OUT_DIR = HERE
DATA_PATH = OUT_DIR / OUT_NAME # Final file path

# e-Stat API endpoint and statistics data ID for Japan's 2020-Base Consumer Price Index data.
ESTAT_SIMPLE_URL = "http://api.e-stat.go.jp/rest/3.0/app/getSimpleStatsData"
ESTAT_STATS_DATA_ID = "0003427113"  # "2020-Base Consumer Price Index" table

# The following information can be found in https://www.stat.go.jp/english/data/cpi/pdf/2020base-list.pdf
# Item and area labels for the 2020-Base Consumer Price Index table.
TARGET_ITEM_LABEL = "All items, less fresh food"                  # Japan's official "core CPI"
HEADLINE_ITEM_LABEL = "All items"                                 # kept for context
TARGET_CORE_ITEM_LABEL = "All items, less fresh food and energy"  # Japan's official "core-core CPI"
JAPAN_AREA_LABEL = "All Japan"
TOKYO_AREA_LABEL = "Ku-area of Tokyo"

ITEM_CODES = {                      # Item codes for the 2020-Base Consumer Price Index table.
    "0001": HEADLINE_ITEM_LABEL,    # All items
    "0161": TARGET_ITEM_LABEL,      # All items, less fresh food (core CPI)
    "0178": TARGET_CORE_ITEM_LABEL, # core-core (optional trend)
}
AREA_CODES = {                      # Area codes for the 2020-Base Consumer Price Index table.
    "00000": JAPAN_AREA_LABEL,      # All Japan
    "13A01": TOKYO_AREA_LABEL,      # Ku-area of Tokyo
}

# FRED public CSV endpoint — no API key required.
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_DAILY_SERIES = {
    "USDJPY": "DEXJPUS", # Daily
    "Oil_Brent": "DCOILBRENTEU", # Daily
}

# Release-lag assumptions, in calendar days from month-end of the reference
# month to the date the figure becomes public. Source: Statistics Bureau
# release calendar, https://www.stat.go.jp/english/data/cpi/1582.html.
# Treated as fixed for simplicity across the sample.
JAPAN_CPI_LAG_DAYS = 24
TOKYO_CPI_LAG_DAYS = 1

MAX_FFILL_DAYS = 5  # forward-fill limit when resampling FRED daily -> monthly


# ------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------

# E-Stat API functions

def fetch_estat_api(app_id: str, stats_data_id: str, estat_simple_url: str = ESTAT_SIMPLE_URL) -> requests.Response:
    """
    Fetch data from the e-Stat API.

    app_id: str: Your e-Stat application ID. Create a free account
    at https://api.e-stat.go.jp/ and request it.
    stats_data_id: str: The ID of the statistics data you want to retrieve
    (e.g., "0003427113" for Japan's 2020-Base Consumer Price Index data)
    """

    params = {
        "appId": app_id, # Personal application ID for the e-Stat API
        "lang": "E", # English
        "statsDataId": stats_data_id, # Statistics data ID for Japan's 2020-Base Consumer Price Index data
        "cdCat01": ",".join(ITEM_CODES), # Desired items
        "cdArea": ",".join(AREA_CODES), # Desired areas
        "metaGetFlg": "N",
        "cntGetFlg": "N",
        "explanationGetFlg": "N",
        "annotationGetFlg": "N",
        "sectionHeaderFlg": "1",
        "replaceSpChars": "0",
    }

    # e-Stat can prepend metadata lines before the real header row -
    # find "tab_code" (the actual data header) and start parsing there.
    response = requests.get(estat_simple_url, params = params, timeout = 60)
    response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
    response.encoding = "utf-8"

    return response


def tidy_estat_cpi(response: requests.Response) -> pd.DataFrame:
    """
    Convert a api response (from the e-Stat API) into a tidy long DataFrame:
    [Area, Item, Period, Value] — monthly rows only.

    time_code encodes YYYY + 2-digit period-type flag + month repeated
    twice, e.g.:
        "2026000606" -> Jun 2026            (kept)
        "2025001212" -> Dec 2025            (kept)
        "2025000000" -> calendar-year 2025  (dropped)
        "2025100000" -> fiscal-year 2025    (dropped)
    """

    lines = response.text.splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith('"tab_code"'))
    raw = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), dtype = str)

    # The table stacks three "tabulated variables" per area/item/month:
    #   tab_code 1 = Index (level)
    #   tab_code 2 = Change from the previous period (MoM/QoQ/YoY)
    #   tab_code 3 = Change over the year
    # Keep only the index level; we derive our own growth rates in
    # build_dataset() (see module docstring on the pending 2025-base rebase).
    # Without this, the extra rows collide on (area, item, period) downstream.
    if "tab_code" in raw.columns:
        raw = raw[raw["tab_code"].astype(str).str.strip() == "1"]

    tc = raw["time_code"].astype(str)
    month, month_dup = tc.str[-2:], tc.str[-4:-2]
    is_monthly = (month == month_dup) & (month != "00") # drop annual/fiscal rows

    # Select only the relevant columns and rows
    df = raw.loc[is_monthly, ["Area(2020-base)", "Items(2020-base)", "value"]].copy()
    df.columns = ["Area", "Item", "Value"] # Rename columns to more convenient names
    df["Period"] = pd.to_datetime(
        tc.loc[is_monthly].str[:4] + month.loc[is_monthly], format = "%Y%m"
    ) + pd.offsets.MonthEnd(0)
    df["Value"] = pd.to_numeric(df["Value"], errors = "coerce")

    return df[["Area", "Item", "Period", "Value"]].sort_values("Period").reset_index(drop = True)


def get_CPI_series(
    long_df: pd.DataFrame,
    area_label: str,
    item_label: str,
    series_name: str,
    lag_days: int,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Extract a tidy CPI series from the long-format e-Stat DataFrame, with columns:
    [Period, series_name, series_name_available], where series_name is the
    CPI index level and series_name_available is the date the figure becomes public.
    """

    df = long_df[(long_df["Area"] == area_label) & (long_df["Item"] == item_label)].copy()
    df = df.dropna(subset = ["Value"]).sort_values("Period").drop_duplicates("Period")
    df["Available_date"] = df["Period"] + pd.Timedelta(days = lag_days)
    df = df.rename(columns = {"Value": series_name, "Available_date": f"{series_name}_available"})
    df = df[(df["Period"] >= start) & (df["Period"] <= end)]

    return df[["Period", series_name, f"{series_name}_available"]]


# FRED functions

def fetch_fred_series(series_id: str, fred_url: str = FRED_CSV_URL) -> pd.DataFrame:
    """
    Pull one FRED series via its public CSV endpoint (no API key needed).
    """

    url = fred_url.format(series_id = series_id)
    raw = pd.read_csv(url)
    raw.columns = ["date", "value"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")  # FRED marks missing as "."

    return raw


def tidy_fred_series(raw: pd.DataFrame, series_name: str, start: str, end: str) -> pd.DataFrame:
    """
    Tidy a FRED series into a DataFrame with columns:
    [period, series_name, series_name_available]
    """

    df = raw.rename(columns = {"date": "Period", "value": series_name})

    df = df.set_index("Period").sort_index()
    df[series_name] = df[series_name].ffill(limit = MAX_FFILL_DAYS)

    monthly = df[series_name].resample("ME").last().to_frame() # "ME" for month-end frequency
    monthly = monthly.reset_index()
    monthly[f"{series_name}_available"] = monthly["Period"]
    monthly = monthly[(monthly["Period"] >= start) & (monthly["Period"] <= end)]

    return monthly


# Build the final dataset

def build_dataset(app_id: str, start: str, end: str):
    """
    Build a dataset with the specified time range.

    returns a DataFrame with columns:
    [Period, Japan Core CPI, Tokyo Core CPI, USDJPY, Oil Brent]
    
    Period: the month-end date of the observation (datetime)
    Japan Core CPI: the level (value) of the Japan Core CPI index (float)
    Japan Core Core CPI: the level (value) of the Japan Core-Core CPI index (float)
    Tokyo Core CPI: the level (value) of the Tokyo Core CPI index (float)
    USDJPY: the level (diff log) of the USDJPY exchange rate (float)
    Oil Brent: the level (diff log) of the Oil Brent price (float)
    """

    cpi_long_df = tidy_estat_cpi(fetch_estat_api(app_id, ESTAT_STATS_DATA_ID, ESTAT_SIMPLE_URL))

    Japan_core_CPI = get_CPI_series(
        cpi_long_df, JAPAN_AREA_LABEL, TARGET_ITEM_LABEL, "Japan Core CPI",
        JAPAN_CPI_LAG_DAYS, start, end
    )
    Japan_core_core_CPI = get_CPI_series(
        cpi_long_df, JAPAN_AREA_LABEL, TARGET_CORE_ITEM_LABEL, "Japan Core-Core CPI",
        JAPAN_CPI_LAG_DAYS, start, end
    )
    Tokyo_core_CPI = get_CPI_series(
        cpi_long_df, TOKYO_AREA_LABEL, TARGET_ITEM_LABEL, "Tokyo Core CPI",
        TOKYO_CPI_LAG_DAYS, start, end
    )

    fred_frames = {}
    for name, series_id in FRED_DAILY_SERIES.items():
        raw = fetch_fred_series(series_id, FRED_CSV_URL)
        fred_frames[name] = tidy_fred_series(raw, name, start, end)

    df = Japan_core_CPI.merge(Tokyo_core_CPI, on = "Period", how = "outer")
    df = df.merge(Japan_core_core_CPI, on = "Period", how = "outer")
    for frame in fred_frames.values():
        df = df.merge(frame, on = "Period", how = "outer")
    df = df.sort_values("Period").reset_index(drop = True)

    # Missing-data handling: report, don't silently impute past the short
    # forward-fill already applied to the FRED series.
    missing_report = df.isna().sum()
    missing_report = missing_report[missing_report > 0]
    if not missing_report.empty:
        print("Missing values by column:\n" + missing_report.to_string())

    return df


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main():
    app_id = os.environ.get("ESTAT_APP_ID", "YOUR_APP_ID_HERE")
    # Replace with your actual e-Stat application ID or set it as an environment variable.
    # In the terminal:
    # export ESTAT_APP_ID="YOUR_APP_ID_HERE"  # For Linux/macOS
    # setx ESTAT_APP_ID "YOUR_APP_ID_HERE"  # For Windows

    start_date = "1970-01-01"
    end_date = "2026-12-31"

    df = build_dataset(app_id, start_date, end_date)

    df.to_csv(DATA_PATH, index = False)  # Save the final dataset to a CSV file


if __name__ == "__main__":
    main()

