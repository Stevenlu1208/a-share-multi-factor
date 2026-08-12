# src/data.py

from pathlib import Path
import akshare as ak
import pandas as pd
import numpy as np
import time

from config import CACHE_DIR, DATA_START_DATE, END_DATE

CACHE_PATH = Path(CACHE_DIR)
CACHE_PATH.mkdir(parents=True, exist_ok=True)


def _cache_file(name):
    return CACHE_PATH / name


def _prefixed(code):
    """6/9 开头 -> sh，其余 -> sz"""
    return ("sh" + code) if code.startswith(("6", "9")) else ("sz" + code)


def _find_col(df, candidates):
    if df is None or df.empty:
        return None
    for candidate in candidates:
        for col in df.columns:
            if candidate.lower() in str(col).lower():
                return col
    return None


# ---------------- 股票池 ----------------
def get_universe(index_symbol, max_stocks=None):
    """
    获取指数成分股（支持多指数取并集），并过滤 ST / 退市风险股。
    """
    symbols = index_symbol if isinstance(index_symbol, (list, tuple)) else [index_symbol]

    frames = []
    for sym in symbols:
        fp = _cache_file(f"universe_{sym}.csv")
        if fp.exists():
            df = pd.read_csv(fp, dtype={"code": str})
        else:
            raw = ak.index_stock_cons_csindex(symbol=sym)
            df = pd.DataFrame({
                "code": raw["成分券代码"].astype(str).str.zfill(6),
                "name": raw["成分券名称"].astype(str),
            })
            df.to_csv(fp, index=False)
        frames.append(df)

    universe = pd.concat(frames, ignore_index=True).drop_duplicates("code")

    # ---- ST / 退市风险过滤（*ST、ST、退市整理） ----
    if "name" in universe.columns:
        mask = universe["name"].str.contains("ST|退", na=False)
        print(f"🚷 股票池 {len(universe)} 只，剔除 ST/退市风险股 {int(mask.sum())} 只")
        universe = universe[~mask]

    universe = universe.reset_index(drop=True)
    if max_stocks:
        universe = universe.head(max_stocks)
    return universe


# ---------------- 指数日线（基准 + 交易日历） ----------------
def get_index_daily(index_symbol="sh000300", start_date=DATA_START_DATE, end_date=END_DATE):
    fp = _cache_file(f"index_{index_symbol}.csv")
    if fp.exists():
        return pd.read_csv(fp, parse_dates=["date"])

    df = None

    # 源1：新浪
    try:
        raw = ak.stock_zh_index_daily(symbol=index_symbol)
        if raw is not None and not raw.empty:
            df = raw[["date", "close"]].copy()
    except Exception as e:
        print(f"新浪指数 error: {e}")

    # 源2：腾讯
    if df is None or df.empty:
        try:
            raw = ak.stock_zh_index_daily_tx(
                symbol=index_symbol,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if raw is not None and not raw.empty:
                df = raw[["date", "close"]].copy()
        except Exception as e:
            print(f"腾讯指数 error: {e}")

    # 源3：东财
    if df is None or df.empty:
        raw = ak.index_zh_a_hist(
            symbol=index_symbol[2:],
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
        df = raw.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]]

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna()
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df = df.sort_values("date").reset_index(drop=True)

    df.to_csv(fp, index=False)
    return df


# ---------------- 个股日线（前复权） ----------------
def get_stock_price(code, start_date=DATA_START_DATE, end_date=END_DATE):
    fp = _cache_file(f"price_{code}.parquet")
    if fp.exists():
        return pd.read_parquet(fp)

    sd = start_date.replace("-", "")
    ed = end_date.replace("-", "")
    sym = _prefixed(code)
    df = None

    # 源1：新浪
    try:
        raw = ak.stock_zh_a_daily(symbol=sym, start_date=sd, end_date=ed, adjust="qfq")
        if raw is not None and not raw.empty:
            df = raw[["date", "close"]].copy()
    except Exception as e:
        print(f"{code} 新浪行情 error: {e}")

    # 源2：腾讯
    if df is None or df.empty:
        try:
            raw = ak.stock_zh_a_hist_tx(symbol=sym, start_date=sd, end_date=ed, adjust="qfq")
            if raw is not None and not raw.empty:
                df = raw[["date", "close"]].copy()
        except Exception as e:
            print(f"{code} 腾讯行情 error: {e}")

    # 源3：东财
    if df is None or df.empty:
        raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=sd, end_date=ed, adjust="qfq")
        df = raw.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]]

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)

    if not df.empty:
        df.to_parquet(fp, index=False)
    return df


# ---------------- 历史市净率（百度源，已验证可用） ----------------
def get_stock_pb(code):
    fp = _cache_file(f"pb_{code}.parquet")
    if fp.exists():
        return pd.read_parquet(fp)

    out = pd.DataFrame(columns=["date", "pb"])

    try:
        raw = ak.stock_zh_valuation_baidu(symbol=code, indicator="市净率", period="全部")
        if raw is not None and not raw.empty:
            raw = raw.rename(columns={"value": "pb"})
            raw["date"] = pd.to_datetime(raw["date"])
            raw["pb"] = pd.to_numeric(raw["pb"], errors="coerce")
            out = raw[["date", "pb"]].dropna()
            out = out[(out["date"] >= DATA_START_DATE) & (out["date"] <= END_DATE)]
            out = out.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"{code} 百度PB error: {e}")

    if not out.empty:
        out.to_parquet(fp, index=False)
    return out


# ---------------- 财务指标（新浪源） ----------------
def _to_numeric(df, col):
    if col is None:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _report_available_date(report_date):
    y, m = report_date.year, report_date.month
    if m == 3:
        return pd.Timestamp(y, 5, 1)
    if m == 6:
        return pd.Timestamp(y, 9, 1)
    if m == 9:
        return pd.Timestamp(y, 11, 1)
    if m == 12:
        return pd.Timestamp(y + 1, 5, 1)
    return pd.Timestamp(y + 100, 1, 1)


def get_financial_indicator(code, start_year="2017"):
    fp = _cache_file(f"fin_{code}.parquet")

    cols = ["report_date", "roe", "gross_margin", "revenue_yoy", "debt_to_asset", "available_date"]

    if fp.exists():
        return pd.read_parquet(fp)

    raw = ak.stock_financial_analysis_indicator(symbol=code, start_year=start_year)

    if raw is None or raw.empty:
        out = pd.DataFrame(columns=cols)
        out.to_parquet(fp, index=False)
        return out

    df = raw.copy()

    date_col = _find_col(df, ["日期", "报告期"])
    roe_col = _find_col(df, ["净资产收益率", "ROE"])
    gm_col = _find_col(df, ["销售毛利率", "毛利率"])
    debt_col = _find_col(df, ["资产负债率"])
    rev_col = _find_col(df, ["主营业务收入增长率", "营业收入增长率", "营收增长率", "收入同比"])

    if date_col is None:
        out = pd.DataFrame(columns=cols)
        out.to_parquet(fp, index=False)
        return out

    out = pd.DataFrame()
    out["report_date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["roe"] = _to_numeric(df, roe_col)
    out["gross_margin"] = _to_numeric(df, gm_col)
    out["revenue_yoy"] = _to_numeric(df, rev_col)
    out["debt_to_asset"] = _to_numeric(df, debt_col)

    out = out.dropna(subset=["report_date"])
    out = out[out["report_date"].dt.month.isin([3, 6, 9, 12])]

    if out.empty:
        out = pd.DataFrame(columns=cols)
        out.to_parquet(fp, index=False)
        return out

    out["available_date"] = out["report_date"].apply(_report_available_date)
    out = out.sort_values(["available_date", "report_date"])
    out = out.drop_duplicates(subset=["available_date"], keep="last")
    out = out.reset_index(drop=True)

    out.to_parquet(fp, index=False)
    return out

# ---------------- 历史总市值（百度源） ----------------
def get_stock_mcap(code):
    fp = _cache_file(f"mcap_{code}.parquet")
    if fp.exists():
        return pd.read_parquet(fp)

    out = pd.DataFrame(columns=["date", "mcap"])
    try:
        raw = ak.stock_zh_valuation_baidu(symbol=code, indicator="总市值", period="全部")
        if raw is not None and not raw.empty:
            raw = raw.rename(columns={"value": "mcap"})
            raw["date"] = pd.to_datetime(raw["date"])
            raw["mcap"] = pd.to_numeric(raw["mcap"], errors="coerce")
            out = raw[["date", "mcap"]].dropna()
            out = out[(out["date"] >= DATA_START_DATE) & (out["date"] <= END_DATE)]
            out = out.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"{code} 百度市值 error: {e}")

    if not out.empty:
        out.to_parquet(fp, index=False)
    return out


# ---------------- 行业分类 ----------------
def _fetch_industry_sina():
    """源1：新浪行业分类（全市场，一次性缓存）"""
    try:
        sectors = ak.stock_sector_spot(indicator="新浪行业")
        rows = []
        for _, s in sectors.iterrows():
            try:
                det = ak.stock_sector_detail(sector=str(s["label"]))
                sub = det[["code"]].copy()
                sub["code"] = sub["code"].astype(str).str.zfill(6)
                sub["industry"] = str(s["板块"])
                rows.append(sub)
                time.sleep(0.2)
            except Exception:
                continue
        if rows:
            full = pd.concat(rows, ignore_index=True).drop_duplicates("code")
            if len(full) > 100:
                return full
    except Exception as e:
        print(f"新浪行业 error: {e}")
    return None


def _fetch_industry_em(codes):
    """源2：东财个股信息（只拉股票池）"""
    rows = []
    try:
        for code in codes:
            info = ak.stock_individual_info_em(symbol=code)
            v = info[info["item"] == "行业"]["value"]
            if len(v) > 0:
                rows.append({"code": code, "industry": str(v.iloc[0])})
            time.sleep(0.1)
    except Exception as e:
        print(f"东财行业 error: {e}")
    return pd.DataFrame(rows) if rows else None


def get_industry_map(codes):
    """返回 {code: 行业名}，失败时全部返回 unknown（自动退化为不做行业中性化）"""
    fp = _cache_file("industry_map.csv")

    if fp.exists():
        full = pd.read_csv(fp, dtype={"code": str})
    else:
        full = _fetch_industry_sina()
        if (full is None or full.empty):
            full = _fetch_industry_em(codes)
        if full is not None and not full.empty:
            full.to_csv(fp, index=False)

    if full is None or full.empty:
        print("警告：行业数据获取失败，本次不做行业中性化")
        return {c: "unknown" for c in codes}

    m = dict(zip(full["code"].astype(str), full["industry"].astype(str)))
    return {c: m.get(c, "unknown") for c in codes}

def get_suspended_map(prices, dates, max_gap_days=7):
    """
    判定每只股票在每个调仓日是否停牌：
    若其最后交易日距调仓日超过 max_gap_days 个自然日，视为停牌。
    返回 {Timestamp: set(codes)}
    """
    last_dates = {}
    for code, df in prices.items():
        if df is None or df.empty:
            last_dates[code] = None
        else:
            last_dates[code] = pd.to_datetime(df["date"]).max()

    result = {}
    for d in dates:
        d = pd.Timestamp(d)
        result[d] = {
            code for code, ld in last_dates.items()
            if ld is None or (d - ld).days > max_gap_days
        }
    return result

def get_stock_turnover(code):
    """历史换手率：新浪日线自带 turnover 列，缓存"""
    fp = _cache_file(f"turnover_{code}.parquet")
    if fp.exists():
        return pd.read_parquet(fp)

    out = pd.DataFrame(columns=["date", "turnover"])
    try:
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        raw = ak.stock_zh_a_daily(symbol=f"{prefix}{code}")
        if raw is not None and not raw.empty and "turnover" in raw.columns:
            raw = raw.copy()
            raw["date"] = pd.to_datetime(raw["date"])
            raw["turnover"] = pd.to_numeric(raw["turnover"], errors="coerce")
            out = raw[["date", "turnover"]].dropna()
            out = out[(out["date"] >= DATA_START_DATE) & (out["date"] <= END_DATE)]
            out = out.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"{code} 新浪换手率 error: {e}")

    if not out.empty:
        out.to_parquet(fp, index=False)
    return out