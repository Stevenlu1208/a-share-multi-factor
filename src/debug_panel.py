import pandas as pd
from config import *
from src.data import get_universe, get_index_daily, get_stock_price, get_stock_pb, get_financial_indicator
from src.factors import get_month_end_trading_dates, build_panel, compute_scores

print("pandas version:", pd.__version__)

# 只拿 3 只股票和最后 3 个月来做极速测试
codes = get_universe(UNIVERSE_INDEX, MAX_STOCKS)["code"].tolist()[:3]
idx = get_index_daily(INDEX_SYMBOL, DATA_START_DATE, END_DATE)
rd = get_month_end_trading_dates(idx, START_DATE, END_DATE)[-3:]

prices = {c: get_stock_price(c, DATA_START_DATE, END_DATE) for c in codes}
pbs = {c: get_stock_pb(c) for c in codes}
fins = {c: get_financial_indicator(c, "2017") for c in codes}

print("\n--- 1. 原始数据行数 ---")
for c in codes: 
    print(f"{c} | price: {len(prices[c])} | pb: {len(pbs[c])} | fin: {len(fins[c])}")

panel = build_panel(codes, prices, pbs, fins, rd)
print("\n--- 2. panel 各列非空数量 ---")
print(panel.notna().sum())

scored = compute_scores(panel)
print("\n--- 3. 打分后检查 ---")
print("scored 的列:", scored.columns.tolist())
if "score" in scored.columns: 
    print("score 非空数量:", scored["score"].notna().sum())
else: 
    print("没有 score 列！")