# src/backtest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import USE_POSITION_CONTROL, POSITION_METHOD, POSITION_MA_MONTHS, POSITION_LOW, POSITION_DD_STEPS


def monthly_close_from_prices(prices, codes, month_end_dates, ffill_limit=1):
    """
    将个股日线收盘价转换成月末收盘价矩阵。

    ffill_limit=1：
    如果某个月末停牌，最多用前一个月末价格填充一次。
    这只是简化处理，更严谨的做法需要单独处理停牌。
    """
    idx = pd.DatetimeIndex(month_end_dates)
    data = {}

    for code in codes:
        df = prices.get(code)

        if df is None or df.empty:
            continue

        s = (
            df.drop_duplicates("date")
            .set_index("date")["close"]
            .sort_index()
        )

        s = s.reindex(idx, method="ffill", limit=ffill_limit)
        data[code] = s

    close = pd.DataFrame(data)

    return close


def monthly_returns(close):
    """
    根据月末收盘价计算月度收益率。
    """
    return close.pct_change(fill_method=None)


def index_monthly_returns(index_daily, month_end_dates):
    """
    计算基准指数月度收益率。
    """
    idx = pd.DatetimeIndex(month_end_dates)

    s = (
        index_daily.drop_duplicates("date")
        .set_index("date")["close"]
        .sort_index()
    )

    s = s.reindex(idx, method="ffill", limit=1)

    return s.pct_change()

def compute_positions(index_daily, rebalance_dates):
    """
    简易仓位管理（只用调仓日已知的基准数据，防未来函数）。
    """
    if not USE_POSITION_CONTROL:
        return {pd.Timestamp(d): 1.0 for d in rebalance_dates}

    df = index_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    g = df.groupby(df["date"].dt.to_period("M"))
    monthly = pd.Series(
        g["close"].last().values,
        index=pd.DatetimeIndex(g["date"].max().values),
    ).sort_index()

    positions = {}
    for d in rebalance_dates:
        d = pd.Timestamp(d)
        hist = monthly[monthly.index <= d]

        if len(hist) < 3:
            positions[d] = 1.0
            continue

        close = hist.iloc[-1]

        if POSITION_METHOD == "ma":
            ma = hist.tail(POSITION_MA_MONTHS).mean()
            positions[d] = 1.0 if close > ma else POSITION_LOW
        elif POSITION_METHOD == "drawdown":
            dd = close / hist.max() - 1
            level = 1.0
            for th, lv in POSITION_DD_STEPS:
                if dd <= th:
                    level = lv
            positions[d] = level
        else:
            positions[d] = 1.0

    return positions


def portfolio_returns(holdings, monthly_ret, rebalance_dates, cost_rate=0.003, positions=None):
    """
    计算组合月度收益并扣除交易成本。
    positions: {调仓日: 仓位0~1}，None 表示永远满仓。
    """
    rets = []
    dates = []
    turnovers = []
    prev_w = {}

    for i in range(len(rebalance_dates) - 1):
        d0 = pd.Timestamp(rebalance_dates[i])
        d1 = pd.Timestamp(rebalance_dates[i + 1])

        stocks = holdings.get(d0, [])
        p = positions.get(d0, 1.0) if positions else 1.0

        # 目标权重：权益部分等权，总仓位 = p
        w = {c: p / len(stocks) for c in stocks} if stocks else {}

        # 交易量 = 权重变动绝对值之和（含选股换手 + 仓位变动），成本 = 交易量 × 单边成本
        all_codes = set(w) | set(prev_w)
        traded = sum(abs(w.get(c, 0.0) - prev_w.get(c, 0.0)) for c in all_codes)
        cost = traded * cost_rate

        if d1 in monthly_ret.index and stocks:
            valid = [c for c in stocks
                     if c in monthly_ret.columns and not pd.isna(monthly_ret.loc[d1, c])]
            r = p * monthly_ret.loc[d1, valid].mean(skipna=True) if valid else 0.0
        else:
            r = 0.0

        rets.append(float(r - cost))
        dates.append(d1)
        turnovers.append(traded / 2)   # 单边换手率
        prev_w = w

    return pd.Series(rets, index=pd.DatetimeIndex(dates), name="strategy"), turnovers


def performance_metrics(ret, periods_per_year=12):
    """
    计算绩效指标。
    """
    ret = ret.dropna()

    if len(ret) == 0:
        return {}

    nav = (1 + ret).cumprod()

    years = len(ret) / periods_per_year

    total_return = nav.iloc[-1] - 1

    if years > 0:
        cagr = nav.iloc[-1] ** (1 / years) - 1
    else:
        cagr = np.nan

    vol = ret.std() * np.sqrt(periods_per_year)

    if vol != 0:
        sharpe = (ret.mean() * periods_per_year) / vol
    else:
        sharpe = np.nan

    max_drawdown = (nav / nav.cummax() - 1).min()

    return {
        "累计收益": float(total_return),
        "年化收益": float(cagr),
        "年化波动": float(vol),
        "夏普比率": float(sharpe),
        "最大回撤": float(max_drawdown)
    }


def plot_nav(strategy_ret, benchmark_ret, output_path):
    """
    画净值曲线。
    """
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    strategy_ret = strategy_ret.fillna(0)
    nav_s = (1 + strategy_ret).cumprod()

    bench = benchmark_ret.reindex(nav_s.index).fillna(0)
    nav_b = (1 + bench).cumprod()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(nav_s.index, nav_s.values, label="Strategy")
    ax.plot(nav_b.index, nav_b.values, label="Benchmark")

    ax.set_title("Multi-factor Strategy NAV")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
