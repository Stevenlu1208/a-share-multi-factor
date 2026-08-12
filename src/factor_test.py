# src/factor_test.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from config import FACTOR_DIRECTION, OUTPUT_DIR
from src.factors import neutralize_cross_section

OUT_DIR = Path(OUTPUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _setup_plot():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

def calculate_forward_returns(close_df):
    """
    计算未来一个月的收益率。
    close_df: 月末收盘价矩阵 (index=date, columns=codes)
    """
    ret = close_df.pct_change(fill_method=None).shift(-1) # shift(-1) 表示下个月的收益
    ret = ret.stack().reset_index()
    ret.columns = ["date", "code", "fwd_ret"]
    return ret

def run_ic_and_quintile_tests(panel, fwd_ret_df):
    """
    对所有配置的因子进行 IC 检验和五分位分组回测。
    """
    _setup_plot()
    
    # 合并因子面板和未来收益
    merged = pd.merge(panel, fwd_ret_df, on=["date", "code"], how="inner")
    merged = merged.dropna(subset=["fwd_ret"])
    
    base = [f for f in FACTOR_DIRECTION.keys() if f in merged.columns]
    neut = [f"{f}_n" for f in FACTOR_DIRECTION.keys() if f"{f}_n" in merged.columns]
    factors_to_test = base + neut
    
    ic_summary = []
    
    for factor in factors_to_test:
        print(f"\n--- 正在检验因子: {factor} ---")
        
        # 按日期分组计算 Rank IC (Spearman相关系数)
        ic_list = []
        for date, group in merged.groupby("date"):
            valid = group.dropna(subset=[factor])
            if len(valid) < 10:
                continue
            ic, _ = stats.spearmanr(valid[factor], valid["fwd_ret"])
            ic_list.append({"date": date, "ic": ic})
            
        ic_df = pd.DataFrame(ic_list)
        
        if ic_df.empty:
            continue
            
        # 统计 IC 序列
        ic_mean = ic_df["ic"].mean()
        ic_std = ic_df["ic"].std()
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ic_win_rate = (ic_df["ic"] > 0).mean()
        
        # 考虑因子方向
        direction = FACTOR_DIRECTION.get(factor, FACTOR_DIRECTION.get(factor[:-2], 1))
        effective_ic = ic_mean * direction 
        
        print(f"IC均值: {ic_mean:.4f} | 方向调整后IC: {effective_ic:.4f} | ICIR: {icir:.4f} | IC胜率: {ic_win_rate:.2%}")
        
        ic_summary.append({
            "factor": factor,
            "IC_Mean": ic_mean,
            "Effective_IC": effective_ic,
            "ICIR": icir,
            "Win_Rate": ic_win_rate
        })
        
        # 1. 画 IC 序列图
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(ic_df["date"], ic_df["ic"], color="skyblue", alpha=0.6, label="Rank IC")
        ax1.axhline(0, color="black", linewidth=1)
        
        ax2 = ax1.twinx()
        cum_ic = ic_df["ic"].cumsum()
        ax2.plot(ic_df["date"], cum_ic, color="red", linewidth=2, label="累计 IC")
        
        plt.title(f"因子 {factor} 的 Rank IC 序列 (方向:{'+' if direction==1 else '-'})")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ic_series_{factor}.png", dpi=150)
        plt.close(fig)
        
        # 2. 五分位分组回测
        quintile_rets = []
        for date, group in merged.groupby("date"):
            valid = group.dropna(subset=[factor])
            if len(valid) < 20:
                continue
            
            # 按因子值从小到大排序，分成 5 组
            # 按"方向调整后"值排序，保证 Q1=理论最差、Q5=理论最好（正负因子统一）
            direction = FACTOR_DIRECTION.get(factor, FACTOR_DIRECTION.get(factor[:-2], 1))
            sorted_g = valid.assign(_adj=valid[factor] * direction).sort_values("_adj")
            q_size = len(sorted_g) // 5
            for q in range(1, 6):
                if q < 5:
                    q_group = sorted_g.iloc[(q-1)*q_size : q*q_size]
                else:
                    q_group = sorted_g.iloc[(q-1)*q_size : ]
                
                q_ret = q_group["fwd_ret"].mean()
                quintile_rets.append({"date": date, "quintile": f"Q{q}", "ret": q_ret})
                
        q_df = pd.DataFrame(quintile_rets)
        if not q_df.empty:
            nav_df = pd.DataFrame()
            for q in [f"Q{i}" for i in range(1, 6)]:
                q_rets = q_df[q_df["quintile"] == q].set_index("date")["ret"].sort_index()
                nav_df[q] = (1 + q_rets).cumprod()
                
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["#d73027", "#fc8d59", "#fee08b", "#91bfdb", "#4575b4"]
            for i, q in enumerate([f"Q{i}" for i in range(1, 6)]):
                label = f"{q} ({'最差' if q=='Q1' else '最好' if q=='Q5' else '中'})"
                ax.plot(nav_df.index, nav_df[q], label=label, color=colors[i], linewidth=2)
                
            plt.title(f"因子 {factor} 五分位分组净值 (Q1最低, Q5最高)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"quintile_{factor}.png", dpi=150)
            plt.close(fig)

    # 保存 IC 汇总表
    if ic_summary:
        summary_df = pd.DataFrame(ic_summary)
        summary_df.to_csv(OUT_DIR / "ic_summary.csv", index=False)
        print("\n=== 所有因子 IC 汇总表 ===")
        print(summary_df.to_string(index=False))


def rolling_icir_weights(panel, fwd_ret, dates, window=12, min_periods=6):
    """
    滚动 ICIR 加权：
    对每个调仓日，只用"之前" window 个月的 IC 计算各因子权重。
    IC 基于"中性化后"的因子值计算，严格避免未来函数。
    """
    base = [f for f in FACTOR_DIRECTION.keys() if f in panel.columns]

    merged = pd.merge(panel, fwd_ret, on=["date", "code"], how="inner")
    merged = merged.dropna(subset=["fwd_ret"])

    # 1. 每月先中性化，再算每个因子的 IC（方向调整后）
    ic_rows = []
    for date, g in merged.groupby("date"):
        gn = neutralize_cross_section(g, base)
        row = {"date": date}
        for f in base:
            col = f"{f}_n" if f"{f}_n" in gn.columns else f
            valid = gn[[col, "fwd_ret"]].dropna()
            if len(valid) >= 10:
                ic, _ = stats.spearmanr(valid[col] * FACTOR_DIRECTION[f], valid["fwd_ret"])
                row[f] = ic
        ic_rows.append(row)
    ic_df = pd.DataFrame(ic_rows).set_index("date").sort_index()

    # 2. 每个调仓日，用"之前" window 个月的 IC 算 ICIR 权重
    weights_by_date = {}
    for date in dates:
        past = ic_df[ic_df.index < pd.Timestamp(date)].tail(window)
        w = {}
        if len(past) >= min_periods:
            for f in base:
                if f not in past.columns:      # 容错：因子IC全缺失时跳过
                    continue
                s = past[f].dropna()
                if len(s) >= min_periods and s.std() > 0:
                    w[f] = max(s.mean() / s.std(), 0.0)

        if not w or sum(w.values()) == 0:
            w = {f: 1.0 for f in base}         # 热身期 -> 等权

        total = sum(w.values())
        weights_by_date[pd.Timestamp(date)] = {f: v / total for f, v in w.items()}

    return weights_by_date

def subperiod_ic_report(panel, fwd_ret, periods=None):
    """
    分阶段因子稳定性报告：检验因子IC在不同市场regime下是否方向一致。
    """
    if periods is None:
        periods = [
            ("2014-2017", "2014-01-01", "2017-12-31"),
            ("2018-2021", "2018-01-01", "2021-12-31"),
            ("2022-2024", "2022-01-01", "2024-12-31"),
        ]

    base = [f for f in FACTOR_DIRECTION.keys() if f in panel.columns]
    neut = [f"{f}_n" for f in FACTOR_DIRECTION.keys() if f"{f}_n" in panel.columns]
    factor_cols = base + neut

    merged = pd.merge(
        panel[["date", "code"] + factor_cols],
        fwd_ret, on=["date", "code"], how="inner",
    ).dropna(subset=["fwd_ret"])
    merged["date"] = pd.to_datetime(merged["date"])

    rows = []
    for name, s, e in periods:
        sub = merged[(merged["date"] >= s) & (merged["date"] <= e)]
        for f in factor_cols:
            direction = FACTOR_DIRECTION.get(f, FACTOR_DIRECTION.get(f[:-2], 1))
            ic_list = []
            for date, g in sub.groupby("date"):
                valid = g[[f, "fwd_ret"]].dropna()
                if len(valid) >= 10:
                    ic, _ = stats.spearmanr(valid[f] * direction, valid["fwd_ret"])
                    ic_list.append(ic)
            if ic_list:
                ic_s = pd.Series(ic_list)
                rows.append({
                    "period": name,
                    "factor": f,
                    "Effective_IC": round(ic_s.mean(), 4),
                    "ICIR": round(ic_s.mean() / ic_s.std(), 3) if ic_s.std() > 0 else 0,
                    "WinRate": f"{(ic_s > 0).mean():.0%}",
                })

    report = pd.DataFrame(rows)
    print("\n=== 因子分阶段稳定性（方向调整后） ===")
    print(report.to_string(index=False))
    report.to_csv(OUT_DIR / "subperiod_ic.csv", index=False)
    return report