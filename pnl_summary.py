"""
多空平均 PnL 汇总脚本。

用法
----
python pnl_summary.py

输出
----
pnl_single.csv  ：单因子各实验设置的多空 avg PnL（bps/tick）
pnl_multi.csv   ：多因子各实验设置的多空 avg PnL（bps/tick）
同时在终端打印两张汇总表。
"""

import os
import sys

import pandas as pd

import config

# ── 工具 ───────────────────────────────────────────────────────────────────────

def _avg_pnl(cum_daily: pd.DataFrame, ls_col: str = "long_short") -> float:
    """
    从 _cum_daily.csv 的最后一行读取总累计多空收益和总 tick 数，
    返回每 tick 平均 PnL（单位：bps）。
    """
    if cum_daily.empty or ls_col not in cum_daily.columns or "n_ticks" not in cum_daily.columns:
        return float("nan")
    last = cum_daily.iloc[-1]
    n    = last["n_ticks"]
    ls   = last[ls_col]
    if pd.isna(n) or pd.isna(ls) or n == 0:
        return float("nan")
    return ls / n * 1e4


# ── 单因子 ─────────────────────────────────────────────────────────────────────

def collect_single(cs_quantile_root: str) -> pd.DataFrame:
    """
    遍历 cs_quantile/{factor_name}/ret{h}/_cum_daily.csv，
    每个 factor_col 取最后一行计算 avg_pnl_bps。

    列：factor_name, factor_col, ret_horizon, avg_pnl_bps
    """
    rows = []
    if not os.path.isdir(cs_quantile_root):
        return pd.DataFrame(rows)

    for factor_name in sorted(os.listdir(cs_quantile_root)):
        fn_dir = os.path.join(cs_quantile_root, factor_name)
        if not os.path.isdir(fn_dir):
            continue
        for ret_h in sorted(os.listdir(fn_dir)):
            rh_dir = os.path.join(fn_dir, ret_h)
            csv    = os.path.join(rh_dir, "_cum_daily.csv")
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv, dtype={"Date": str})
            if df.empty or "factor_col" not in df.columns:
                continue
            for fc, sub in df.groupby("factor_col"):
                sub = sub.sort_values("Date").reset_index(drop=True)
                rows.append({
                    "factor_name":  factor_name,
                    "factor_col":   fc,
                    "ret_horizon":  ret_h,
                    "avg_pnl_bps":  _avg_pnl(sub),
                })

    return (pd.DataFrame(rows)
            .sort_values(["factor_name", "factor_col", "ret_horizon"])
            .reset_index(drop=True))


# ── 多因子 ─────────────────────────────────────────────────────────────────────

def collect_multi(mfq_root: str) -> pd.DataFrame:
    """
    遍历 multi_factor_quantile/g{n}/{factor_pool}/{score_method}/ret{h}/_cum_daily.csv，
    取最后一行计算 avg_pnl_bps。

    列：n_groups, factor_pool, score_method, ret_horizon, avg_pnl_bps
    """
    rows = []
    if not os.path.isdir(mfq_root):
        return pd.DataFrame(rows)

    for g_dir in sorted(os.listdir(mfq_root)):            # g10 / g20 / ...
        if not g_dir.startswith("g"):
            continue
        try:
            n_groups = int(g_dir[1:])
        except ValueError:
            continue
        g_path = os.path.join(mfq_root, g_dir)
        if not os.path.isdir(g_path):
            continue

        for pool in sorted(os.listdir(g_path)):            # threshold / union / intersection
            pool_path = os.path.join(g_path, pool)
            if not os.path.isdir(pool_path):
                continue

            for method in sorted(os.listdir(pool_path)):   # rank / zscore / minmax
                method_path = os.path.join(pool_path, method)
                if not os.path.isdir(method_path):
                    continue

                for ret_h in sorted(os.listdir(method_path)):  # ret100 / ret200 / ret300
                    rh_dir = os.path.join(method_path, ret_h)
                    csv    = os.path.join(rh_dir, "_cum_daily.csv")
                    if not os.path.exists(csv):
                        continue
                    df = pd.read_csv(csv, dtype={"Date": str})
                    df = df.sort_values("Date").reset_index(drop=True)
                    rows.append({
                        "n_groups":    n_groups,
                        "factor_pool": pool,
                        "score_method": method,
                        "ret_horizon":  ret_h,
                        "avg_pnl_bps":  _avg_pnl(df),
                    })

    return (pd.DataFrame(rows)
            .sort_values(["n_groups", "factor_pool", "score_method", "ret_horizon"])
            .reset_index(drop=True))


# ── 主入口 ─────────────────────────────────────────────────────────────────────

def main():
    cs_root  = os.path.join(config.EVAL_ROOT, "cs_quantile")
    mfq_root = os.path.join(config.EVAL_ROOT, "multi_factor_quantile")

    print("正在扫描单因子结果...")
    single = collect_single(cs_root)
    print("正在扫描多因子结果...")
    multi  = collect_multi(mfq_root)

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    out_single = os.path.join(config.ROOT, "pnl_single.csv")
    out_multi  = os.path.join(config.ROOT, "pnl_multi.csv")
    single.to_csv(out_single, index=False)
    multi.to_csv(out_multi,  index=False)
    print(f"\n已保存：{out_single}")
    print(f"已保存：{out_multi}")

    # ── 打印单因子汇总（pivot：行=factor_col，列=ret_horizon）─────────────────
    if not single.empty:
        print("\n=== 单因子多空平均 PnL（bps/tick）===")
        pivot = (
            single.pivot_table(
                index=["factor_name", "factor_col"],
                columns="ret_horizon",
                values="avg_pnl_bps",
                aggfunc="first",
            )
            .round(4)
        )
        # 列排序
        col_order = [c for c in ["ret100", "ret200", "ret300"] if c in pivot.columns]
        pivot = pivot[col_order]
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.width", 160)
        print(pivot.to_string())
    else:
        print("\n未找到单因子结果（cs_quantile 目录为空或尚未运行）。")

    # ── 打印多因子汇总（pivot：行=实验设置，列=ret_horizon）──────────────────
    if not multi.empty:
        print("\n=== 多因子多空平均 PnL（bps/tick）===")
        multi["experiment"] = (
            "g" + multi["n_groups"].astype(str)
            + " | " + multi["factor_pool"]
            + " | " + multi["score_method"]
        )
        pivot_m = (
            multi.pivot_table(
                index="experiment",
                columns="ret_horizon",
                values="avg_pnl_bps",
                aggfunc="first",
            )
            .round(4)
        )
        col_order = [c for c in ["ret100", "ret200", "ret300"] if c in pivot_m.columns]
        pivot_m = pivot_m[col_order]
        print(pivot_m.to_string())
    else:
        print("\n未找到多因子结果（multi_factor_quantile 目录为空或尚未运行）。")


if __name__ == "__main__":
    main()
