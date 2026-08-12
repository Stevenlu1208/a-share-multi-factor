# src/factors.py

import numpy as np
import pandas as pd

from config import FACTOR_DIRECTION, MIN_FACTORS, NEUTRALIZE_INDUSTRY, NEUTRALIZE_MCAP, MAX_PER_INDUSTRY, NO_NEUTRALIZE_FACTORS


def get_month_end_trading_dates(index_daily, start_date, end_date):
    df = index_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
    return df.groupby(df["date"].dt.to_period("M"))["date"].max().tolist()


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

def get_volatility(price_df, date, lookback=20):
    """
    过去1个月波动率：最近约20个交易日日收益率的标准差。
    （是否年化不影响结果，横截面标准化会消除量纲。）
    """
    if price_df is None or price_df.empty:
        return np.nan
    hist = price_df[price_df["date"] <= date].sort_values("date")
    if len(hist) < lookback + 1:
        return np.nan
    window = hist.tail(lookback + 1)
    ret = window["close"].pct_change().dropna()
    if len(ret) < lookback * 0.8:
        return np.nan
    return ret.std()


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


def build_panel(codes, prices, pbs, fins, rebalance_dates, mcaps=None, turnovers=None):
    rows = []
    for date in rebalance_dates:
        for code in codes:
            price_df = prices.get(code)
            pb = latest_value_before(pbs.get(code), date, "pb")
            mom_1m = get_momentum(price_df, date, lookback=20)
            vol_1m = get_volatility(price_df, date, lookback=20)
            roe, gross_margin, revenue_yoy, debt_to_asset = latest_fin_before(fins.get(code), date)
            mcap = latest_value_before(mcaps.get(code), date, "mcap") if mcaps is not None else np.nan
            turnover = latest_value_before(turnovers.get(code), date, "turnover") if turnovers is not None else np.nan
            rows.append({
                "date": date, "code": code, "pb": pb, "roe": roe,
                "gross_margin": gross_margin, "revenue_yoy": revenue_yoy,
                "debt_to_asset": debt_to_asset, "mcap": mcap,
                "turnover": turnover, "mom_1m": mom_1m, "vol_1m": vol_1m,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "date", "code", "pb", "roe", "gross_margin", "revenue_yoy",
            "debt_to_asset", "mcap", "turnover", "mom_1m", "vol_1m",
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


def neutralize_cross_section(df, factors):
    """
    横截面回归剔除市值与行业暴露，残差作为新因子：
    factor = a + b*ln(mcap) + 行业哑变量 + epsilon
    """
    out = df.copy()
    use_mc = NEUTRALIZE_MCAP and "mcap" in out.columns and out["mcap"].notna().sum() >= 10
    use_ind = NEUTRALIZE_INDUSTRY and "industry" in out.columns

    for f in factors:
        if f not in out.columns:
            continue

        if f in NO_NEUTRALIZE_FACTORS:
            yv = pd.to_numeric(out[f], errors="coerce")
            out[f"{f}_n"] = np.log(yv.clip(lower=1e-6)) if f == "mcap" else yv
            continue

        y = pd.to_numeric(out[f], errors="coerce")
        y = winsorize(y)   # 回归前去极值，避免极端值拉动回归线
        mask = y.notna()

        X_parts = []
        if use_mc:
            lm = np.log(out["mcap"].clip(lower=1e-6))
            mask = mask & lm.notna()
            X_parts.append(lm.rename("log_mcap"))
        if use_ind:
            dummies = pd.get_dummies(out["industry"].fillna("unknown"), prefix="ind")
            if dummies.shape[1] > 1:
                dummies = dummies.drop(columns=dummies.columns[0])
                X_parts.append(dummies)

        if not X_parts:
            out[f"{f}_n"] = y
            continue

        X = pd.concat(X_parts, axis=1).astype(float)
        mask = mask & X.notna().all(axis=1)

        if mask.sum() <= X.shape[1] + 3:
            out[f"{f}_n"] = y
            continue

        yv = y.loc[mask].to_numpy(dtype=float)
        Xv = X.loc[mask].to_numpy(dtype=float)
        Xv = np.column_stack([np.ones(len(Xv)), Xv])

        beta, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
        resid = yv - Xv @ beta

        new_col = pd.Series(np.nan, index=out.index)
        new_col.loc[mask] = resid
        out[f"{f}_n"] = new_col

    return out


def preprocess_cross_section(df, weights=None):
    out = neutralize_cross_section(df, list(FACTOR_DIRECTION.keys()))

    zcols = []
    for raw, direction in FACTOR_DIRECTION.items():
        zcol = f"{raw}_z"
        zcols.append(zcol)
        src = f"{raw}_n" if f"{raw}_n" in out.columns else raw
        s = winsorize(out[src])
        s = standardize(s)
        out[zcol] = s * direction

    out["z_count"] = out[zcols].notna().sum(axis=1)

    if weights:
        # 加权合成：缺失的因子按0贡献，并按剩余权重重新归一
        num = pd.Series(0.0, index=out.index)
        den = pd.Series(0.0, index=out.index)
        for raw in FACTOR_DIRECTION.keys():
            zc = f"{raw}_z"
            w = weights.get(raw, 0.0)
            num = num + out[zc].fillna(0.0) * w
            den = den + out[zc].notna().astype(float) * w
        out["score"] = num / den.replace(0, np.nan)
    else:
        out["score"] = out[zcols].mean(axis=1, skipna=True)

    out.loc[out["z_count"] < MIN_FACTORS, "score"] = np.nan
    return out


def compute_scores(panel, weights_by_date=None):
    if panel is None or panel.empty or "date" not in panel.columns:
        return panel

    results = []
    for date, df in panel.groupby("date"):
        w = weights_by_date.get(date) if weights_by_date else None
        results.append(preprocess_cross_section(df, w))

    if not results:
        return panel
    return pd.concat(results, ignore_index=True)


def select_top(panel, top_n=30, suspended=None):
    """
    带停牌约束的选股：
    - 停牌股不可买入；
    - 上期持仓若本期停牌，无法卖出，强制持有顺延。
    """
    if panel is None or panel.empty or "date" not in panel.columns:
        return {}

    holdings = {}
    prev = set()

    for date, df in panel.groupby("date"):
        susp = suspended.get(date, set()) if suspended else set()
        df = df.dropna(subset=["score"]).sort_values("score", ascending=False)

        # 强制持有：上期持仓中本期停牌的（想卖卖不掉）
        forced = [c for c in prev if c in susp]
        chosen = list(forced)
        count = {}

        inds = df["industry"] if "industry" in df.columns else pd.Series("unknown", index=df.index)
        for code, ind in zip(df["code"], inds):
            if len(chosen) >= top_n:
                break
            if code in susp or code in chosen:
                continue                      # 停牌买不进 / 已强制持有
            if MAX_PER_INDUSTRY:
                if count.get(ind, 0) >= MAX_PER_INDUSTRY:
                    continue
                count[ind] = count.get(ind, 0) + 1
            chosen.append(code)

        holdings[date] = chosen
        prev = set(chosen)

    return holdings