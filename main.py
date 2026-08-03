# main.py

import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import *
from src.data import (
    get_universe,
    get_index_daily,
    get_stock_price,
    get_stock_pb,
    get_financial_indicator,
)
from src.factors import (
    get_month_end_trading_dates,
    build_panel,
    compute_scores,
    select_top,
)
from src.backtest import (
    monthly_close_from_prices,
    monthly_returns,
    index_monthly_returns,
    portfolio_returns,
    performance_metrics,
    plot_nav,
)


def load_stock_data(codes):
    prices, pbs, fins = {}, {}, {}
    start_year = str(int(DATA_START_DATE[:4]) - 1)

    for code in tqdm(codes, desc="Download data"):
        try:
            prices[code] = get_stock_price(code, DATA_START_DATE, END_DATE)
            time.sleep(0.15)
        except Exception as e:
            print(f"{code} price error: {e}")

        try:
            pbs[code] = get_stock_pb(code)
            time.sleep(0.15)
        except Exception as e:
            print(f"{code} pb error: {e}")

        try:
            fins[code] = get_financial_indicator(code, start_year=start_year)
            time.sleep(0.15)
        except Exception as e:
            print(f"{code} financial error: {e}")

    return prices, pbs, fins


def save_holdings(holdings, path):
    rows = []
    for date, stocks in holdings.items():
        rows.append({
            "date": date,
            "stock_count": len(stocks),
            "stocks": "|".join(stocks),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    universe = get_universe(UNIVERSE_INDEX, MAX_STOCKS)
    codes = universe["code"].tolist()
    print(f"股票池数量：{len(codes)}")

    if len(codes) == 0:
        print("股票池为空，请删除 data/cache 并检查网络。")
        return

    index_daily = get_index_daily(INDEX_SYMBOL, DATA_START_DATE, END_DATE)
    print(f"指数日线行数：{len(index_daily)}")

    if index_daily.empty:
        print("指数日线为空，请检查网络或数据源。")
        return

    all_month_ends = get_month_end_trading_dates(index_daily, DATA_START_DATE, END_DATE)
    rebalance_dates = get_month_end_trading_dates(index_daily, START_DATE, END_DATE)
    print(f"月末交易日总数：{len(all_month_ends)}，调仓月数：{len(rebalance_dates)}")

    if len(rebalance_dates) == 0:
        print("调仓日期为空，请检查 config 中的日期范围。")
        return

    print("开始下载个股数据，第一次会比较慢，请耐心等待。")
    prices, pbs, fins = load_stock_data(codes)

    print("开始构建因子面板。")
    panel = build_panel(codes, prices, pbs, fins, rebalance_dates)
    print(f"因子面板行数：{len(panel)}")

    print("开始因子标准化与打分。")
    panel = compute_scores(panel)

    print("开始选股。")
    holdings = select_top(panel, TOP_N)

    if not holdings:
        print("没有有效持仓，请检查数据完整性。")
        return

    close = monthly_close_from_prices(prices, codes, all_month_ends, ffill_limit=1)
    stock_ret = monthly_returns(close)
    bench_ret = index_monthly_returns(index_daily, all_month_ends)

    strategy_ret = portfolio_returns(holdings, stock_ret, rebalance_dates).sort_index()

    metrics = performance_metrics(strategy_ret, periods_per_year=12)
    bench_metrics = performance_metrics(
        bench_ret.reindex(strategy_ret.index).dropna(), periods_per_year=12
    )

    panel.to_csv(Path(OUTPUT_DIR) / "factor_panel.csv", index=False)
    save_holdings(holdings, Path(OUTPUT_DIR) / "holdings.csv")

    if len(strategy_ret) > 0:
        plot_nav(strategy_ret, bench_ret, Path(OUTPUT_DIR) / "nav.png")

    result = {"strategy": metrics, "benchmark": bench_metrics}
    with open(Path(OUTPUT_DIR) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print("运行完成。")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()