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
def get_universe(index_symbol="000300", max_stocks=None):
    fp = _cache_file(f"universe_{index_symbol}.csv")

    df = None
    if fp.exists():
        df = pd.read_csv(fp, dtype={"code": str})
        if len(df) < 10:      # 防止读到以前的坏缓存
            df = None

    if df is None:
        try:
            raw = ak.index_stock_cons_csindex(symbol=index_symbol)
        except Exception as e:
            print(f"csindex 成分股 error: {e}")
            raw = ak.index_stock_cons(symbol=index_symbol)

        code_col = _find_col(raw, ["成分券代码", "品种代码", "证券代码", "股票代码"])
        name_col = _find_col(raw, ["成分券名称", "品种名称", "证券名称", "股票名称"])

        df = pd.DataFrame()
        df["code"] = raw[code_col].astype(str).str.zfill(6)
        df["name"] = raw[name_col].astype(str) if name_col else ""
        df = df[df["code"] != index_symbol].drop_duplicates().reset_index(drop=True)
        df.to_csv(fp, index=False)

    if max_stocks is not None:
        df = df.head(max_stocks)
    return df


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