# src/backtest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def portfolio_returns(holdings, monthly_ret, rebalance_dates):
    """
    计算组合月度收益率。

    逻辑：
    在 d0 月末选股；
    持有到 d1 月末；
    组合收益为选中股票 d1 月收益的均值。
    """
    rets = []
    dates = []

    for i in range(len(rebalance_dates) - 1):
        d0 = pd.Timestamp(rebalance_dates[i])
        d1 = pd.Timestamp(rebalance_dates[i + 1])

        stocks = holdings.get(d0, [])

        if d1 not in monthly_ret.index:
            continue

        if len(stocks) == 0:
            r = 0.0
        else:
            valid_cols = [c for c in stocks if c in monthly_ret.columns]

            if len(valid_cols) == 0:
                r = 0.0
            else:
                r = monthly_ret.loc[d1, valid_cols].mean(skipna=True)

                if pd.isna(r):
                    r = 0.0

        rets.append(float(r))
        dates.append(d1)

    return pd.Series(rets, index=pd.DatetimeIndex(dates), name="strategy")


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
