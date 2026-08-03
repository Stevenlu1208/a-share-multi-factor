# src/factors.py

import numpy as np
import pandas as pd

from config import FACTOR_DIRECTION, MIN_FACTORS


def get_month_end_trading_dates(index_daily, start_date, end_date):
    df = index_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
    month_ends = df.groupby(df["date"].dt.to_period("M"))["date"].max().tolist()
    return month_ends


def latest_value_before(df, date, col):
    if df is None or df.empty:
        return np.nan
    tmp = df[df["date"] <= date]
    if tmp.empty:
        return np.nan
    return tmp[col].iloc[-1]


def get_momentum(price_df, date, lookback=126):
    if price_df is None or price_df.empty:
        return np.nan
    hist = price_df[price_df["date"] <= date].sort_values("date")
    if len(hist) < lookback + 1:
        return np.nan
    window = hist.tail(lookback + 1)
    return window["close"].iloc[-1] / window["close"].iloc[0] - 1.0


def latest_fin_before(fin_df, date):
    if fin_df is None or fin_df.empty:
        return [np.nan, np.nan, np.nan, np.nan]
    tmp = fin_df[fin_df["available_date"] <= date].sort_values(["available_date", "report_date"])
    if tmp.empty:
        return [np.nan, np.nan, np.nan, np.nan]
    row = tmp.iloc[-1]
    return [
        row.get("roe", np.nan),
        row.get("gross_margin", np.nan),
        row.get("revenue_yoy", np.nan),
        row.get("debt_to_asset", np.nan),
    ]


def build_panel(codes, prices, pbs, fins, rebalance_dates):
    rows = []
    for date in rebalance_dates:
        for code in codes:
            price_df = prices.get(code)
            pb = latest_value_before(pbs.get(code), date, "pb")
            mom_6m = get_momentum(price_df, date, lookback=126)
            roe, gross_margin, revenue_yoy, debt_to_asset = latest_fin_before(fins.get(code), date)
            rows.append({
                "date": date,
                "code": code,
                "pb": pb,
                "roe": roe,
                "gross_margin": gross_margin,
                "revenue_yoy": revenue_yoy,
                "debt_to_asset": debt_to_asset,
                "mom_6m": mom_6m,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "date", "code", "pb", "roe", "gross_margin",
            "revenue_yoy", "debt_to_asset", "mom_6m",
        ])

    return pd.DataFrame(rows)


def winsorize(s, lower=0.025, upper=0.975):
    valid = s.dropna()
    if len(valid) < 5:
        return s
    lo, hi = valid.quantile([lower, upper])
    return s.clip(lo, hi)


def standardize(s):
    valid = s.dropna()
    if len(valid) < 5:
        return s
    std = valid.std()
    if pd.isna(std) or std == 0:
        return s.where(s.isna(), 0.0)
    return (s - valid.mean()) / std


def preprocess_cross_section(df):
    out = df.copy()
    zcols = []
    for raw, direction in FACTOR_DIRECTION.items():
        zcol = f"{raw}_z"
        zcols.append(zcol)
        if raw not in out.columns:
            out[zcol] = np.nan
            continue
        s = winsorize(out[raw])
        s = standardize(s)
        out[zcol] = s * direction

    out["z_count"] = out[zcols].notna().sum(axis=1)
    out["score"] = out[zcols].mean(axis=1, skipna=True)
    out.loc[out["z_count"] < MIN_FACTORS, "score"] = np.nan
    return out


def compute_scores(panel):
    if panel is None or panel.empty or "date" not in panel.columns:
        return panel

    results = []
    for date, df in panel.groupby("date"):
        scored = preprocess_cross_section(df)
        results.append(scored)

    if not results:
        return panel

    return pd.concat(results, ignore_index=True)


def select_top(panel, top_n=30):
    if panel is None or panel.empty or "date" not in panel.columns:
        return {}
    holdings = {}
    for date, df in panel.groupby("date"):
        df = df.dropna(subset=["score"]).sort_values("score", ascending=False)
        holdings[date] = df.head(top_n)["code"].tolist()
    return holdings